import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np

from models.main_network import MISRConvLSTM
from dataset import SRDataset
from loss import CombinedLoss
from metrics import calculate_psnr, calculate_ssim, calculate_lpips, crop_edges
from config import config
import lpips
import torch.nn.functional as F
import torch.nn as nn

def train_unified_model():
    """Train the unified model using configuration"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MISRConvLSTM(config.get_model_config()).to(device)
    pretrained_path = config.paths.pretrained_model_path
    model_path = os.path.join(config.paths.model_checkpoint_dir,
                              config.paths.pretrained_model_path)

    if os.path.exists(model_path):
        print(f"Loading pretrained model: {model_path}")
        pretrained_dict = torch.load(model_path, map_location=device)
        model_dict = model.state_dict()

        # Filter parameters with matching dimensions
        matched_dict = {}
        for k, v in pretrained_dict.items():
            if k in model_dict:
                if v.shape == model_dict[k].shape:
                    matched_dict[k] = v
                    print(f"✓ Loading layer: {k} ({tuple(v.shape)})")
                else:
                    print(
                        f"✗ Skipping layer: {k} (pretrained: {tuple(v.shape)}, current: {tuple(model_dict[k].shape)})")
                    # Try compatible processing for partial layers
                    if 'bias' in k or 'weight' in k:
                        # For convolutional weights, attempt to crop center part
                        if len(v.shape) == 4 and len(model_dict[k].shape) == 4:
                            if v.shape[0] == model_dict[k].shape[0] and v.shape[1] == model_dict[k].shape[1]:
                                # Only spatial dimensions differ, crop center part
                                min_kernel = min(v.shape[2], model_dict[k].shape[2])
                                start_v = (v.shape[2] - min_kernel) // 2
                                start_m = (model_dict[k].shape[2] - min_kernel) // 2

                                # Crop weights
                                v_cropped = v[:, :, start_v:start_v + min_kernel, start_v:start_v + min_kernel]
                                matched_dict[k] = v_cropped
                                print(f"  Cropped weights: {k} ({tuple(v.shape)} -> {tuple(v_cropped.shape)})")

        # Load matched parameters
        model_dict.update(matched_dict)
        model.load_state_dict(model_dict, strict=False)

        # Print loading statistics
        print(f"\nLoading statistics:")
        print(f"  Matched parameters: {len(matched_dict)} / {len(pretrained_dict)}")
        print(f"  Total parameters: {len(model_dict)}")
    else:
        print("No pretrained model found, starting training from scratch")
    # criterion = CombinedLoss(
    #     lpips_net_type='alex',
    #     alpha=config.training.loss_alpha,
    #     beta=config.training.loss_beta,
    #     gamma=config.training.loss_gamma
    # ).to(device)
    criterion = nn.L1Loss().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=config.training.learning_rate,
                           weight_decay=1e-5)

    # train dataset
    train_dataset = SRDataset(
        burst_dir=config.paths.train_lr_dir,
        hr_dir=config.paths.train_hr_dir,
        low_res_size=config.training.low_res_size,
        high_res_size=config.training.high_res_size
    )
    train_loader = DataLoader(train_dataset,
                              batch_size=config.training.batch_size,
                              shuffle=True,
                              num_workers=0)

    # val dataset
    val_dataset = SRDataset(
        burst_dir=config.paths.val_lr_dir,
        hr_dir=config.paths.val_hr_dir,
        low_res_size=config.training.low_res_size,
        high_res_size=config.training.high_res_size
    )
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    lpips_model = lpips.LPIPS(net='alex')
    train_losses = []
    val_psnr_values = []
    val_ssim_values = []
    val_lpips_values = []
    best_train_loss = float('inf')
    no_improvement_count = 0
    current_lr = config.training.learning_rate

    for epoch in range(config.training.start_epoch, config.training.num_epochs):
        model.train()
        epoch_loss = 0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{config.training.num_epochs} [Train]', leave=False)

        for batch_idx, (z_rgb_batch, x_hr_gt_batch) in enumerate(pbar):
            z_rgb_batch = z_rgb_batch.to(device)
            x_hr_gt_batch = x_hr_gt_batch.to(device)

            batch_size = z_rgb_batch.shape[0]

            optimizer.zero_grad()

            outputs = []
            for i in range(batch_size):
                output, _ = model(z_rgb_batch[i:i + 1], alphas=None)
                outputs.append(output)

            outputs = torch.cat(outputs, dim=0)

            outputs_cropped = crop_edges(outputs, config.training.crop_pixels)
            x_hr_gt_cropped = crop_edges(x_hr_gt_batch, config.training.crop_pixels)

            loss = criterion(outputs_cropped, x_hr_gt_cropped)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           max_norm=config.training.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg Loss': f'{epoch_loss / (batch_idx + 1):.6f}'
            })

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        if avg_loss < best_train_loss:
            best_train_loss = avg_loss
            no_improvement_count = 0
        else:
            no_improvement_count += 1

            if no_improvement_count >= config.training.patience:
                current_lr *= config.training.lr_reduction_factor
                for param_group in optimizer.param_groups:
                    param_group['lr'] = current_lr
                no_improvement_count = 0

        model.eval()
        val_l1_losses = []
        val_psnr = []
        val_ssim = []
        val_lpips = []

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{config.training.num_epochs} [Val]', leave=False)
            for i, (z_rgb, x_hr_gt) in enumerate(val_pbar):
                z_rgb = z_rgb.to(device)
                x_hr_gt = x_hr_gt.to(device)

                output, _ = model(z_rgb, alphas=None)

                output_cropped = crop_edges(output, config.training.crop_pixels)
                x_hr_gt_cropped = crop_edges(x_hr_gt, config.training.crop_pixels)

                l1_loss = F.l1_loss(output_cropped, x_hr_gt_cropped).item()
                val_l1_losses.append(l1_loss)

                output_np = output_cropped.squeeze().permute(1, 2, 0).cpu().numpy()
                gt_np = x_hr_gt_cropped.squeeze().permute(1, 2, 0).cpu().numpy()

                output_np = np.clip(output_np, 0, 1)
                gt_np = np.clip(gt_np, 0, 1)

                psnr_val = calculate_psnr(output_np, gt_np)
                ssim_val = calculate_ssim(output_np, gt_np)
                lpips_val = calculate_lpips(output_np, gt_np, lpips_model)

                val_pbar.set_postfix({
                    'PSNR': f'{psnr_val:.2f}',
                    'SSIM': f'{ssim_val:.4f}',
                    'LPIPS': f'{lpips_val:.4f}'
                })

                val_psnr.append(psnr_val)
                val_ssim.append(ssim_val)
                val_lpips.append(lpips_val)

        avg_val_l1 = np.mean(val_l1_losses)
        avg_val_psnr = np.mean(val_psnr)
        avg_val_ssim = np.mean(val_ssim)
        avg_val_lpips = np.mean(val_lpips)

        val_psnr_values.append(avg_val_psnr)
        val_ssim_values.append(avg_val_ssim)
        val_lpips_values.append(avg_val_lpips)

        print(f'Epoch {epoch + 1}/{config.training.num_epochs}')
        print(f'  Train Loss: {avg_loss:.6f}, LR: {current_lr:.6f}')
        print(
            f'  Val Metrics - L1: {avg_val_l1:.6f}, PSNR: {avg_val_psnr:.2f}, SSIM: {avg_val_ssim:.4f}, LPIPS: {avg_val_lpips:.4f}')
        print('-' * 80)

        if (epoch + 1) % 1 == 0:
            checkpoint_path = os.path.join(config.paths.model_checkpoint_dir,
                                           f'checkpoint_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    final_model_path = os.path.join(config.paths.model_checkpoint_dir,
                                    config.paths.final_model_path)
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    print("Unified model training completed!")