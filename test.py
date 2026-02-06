import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image
import os
import time
import lpips

from models.main_network import MISRConvLSTM
from dataset import SRDataset
from metrics import calculate_psnr, calculate_ssim, calculate_lpips, crop_edges
from config import config


def test_unified_model(model_path=None):
    """Test the unified model using configuration"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MISRConvLSTM(config.get_model_config()).to(device)

    # check model
    if model_path is None:
        model_path = os.path.join(config.paths.model_checkpoint_dir,
                                  config.paths.pretrained_model_path)

    if os.path.exists(model_path):
        print(f"Loading model: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print("Model loaded successfully")

    test_dataset = SRDataset(
        burst_dir=config.paths.test_lr_dir,
        hr_dir=config.paths.test_hr_dir,
        low_res_size=config.training.low_res_size,
        high_res_size=config.training.high_res_size
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    print(f"Test samples: {len(test_dataset)}")

    lpips_calculator = lpips.LPIPS(net='alex').to(device)
    for param in lpips_calculator.parameters():
        param.requires_grad = False

    # paths
    output_dir = config.paths.test_results_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    all_l1_losses = []
    all_psnr_values = []
    all_ssim_values = []
    all_lpips_values = []

    pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc='Testing progress')

    with torch.no_grad():
        for i, (z_rgb, x_hr_gt) in pbar:
            z_rgb = z_rgb.to(device)
            x_hr_gt = x_hr_gt.to(device)

            output, _ = model(z_rgb, alphas=None)

            output_cropped = crop_edges(output, config.training.crop_pixels)
            x_hr_gt_cropped = crop_edges(x_hr_gt, config.training.crop_pixels)

            l1_loss = F.l1_loss(output_cropped, x_hr_gt_cropped).item()
            all_l1_losses.append(l1_loss)

            output_np = output_cropped.squeeze().permute(1, 2, 0).cpu().numpy()
            gt_np = x_hr_gt_cropped.squeeze().permute(1, 2, 0).cpu().numpy()

            output_np = np.clip(output_np, 0, 1)
            gt_np = np.clip(gt_np, 0, 1)

            psnr_val = calculate_psnr(output_np, gt_np)
            ssim_val = calculate_ssim(output_np, gt_np)
            all_psnr_values.append(psnr_val)
            all_ssim_values.append(ssim_val)

            lpips_val = calculate_lpips(output_np, gt_np, lpips_calculator)
            all_lpips_values.append(lpips_val)

            pbar.set_postfix({
                'L1': f'{l1_loss:.4f}',
                'PSNR': f'{psnr_val:.2f}',
                'SSIM': f'{ssim_val:.4f}',
                'LPIPS': f'{lpips_val:.4f}'
            })

            # save hr
            output_full = output.squeeze().permute(1, 2, 0).cpu().numpy()
            output_full = np.clip(output_full, 0, 1) * 255
            output_img = Image.fromarray(output_full.astype(np.uint8))
            output_img.save(os.path.join(output_dir, f'test_{i:03d}_output.bmp'))

            # save gt
            gt_full = x_hr_gt.squeeze().permute(1, 2, 0).cpu().numpy()
            gt_full = np.clip(gt_full, 0, 1) * 255
            gt_img = Image.fromarray(gt_full.astype(np.uint8))
            gt_img.save(os.path.join(output_dir, f'test_{i:03d}_gt.bmp'))

            # if (i + 1) % 5 == 0:
            #     print(f"Sample {i + 1}/{len(test_dataset)}:")
            #     print(f"  L1 loss: {l1_loss:.6f}")
            #     print(f"  PSNR: {psnr_val:.2f} dB")
            #     print(f"  SSIM: {ssim_val:.4f}")
            #     print(f"  LPIPS: {lpips_val:.4f}")

    avg_l1 = np.mean(all_l1_losses)
    avg_psnr = np.mean(all_psnr_values)
    avg_ssim = np.mean(all_ssim_values)
    avg_lpips = np.mean(all_lpips_values)

    std_psnr = np.std(all_psnr_values)
    std_ssim = np.std(all_ssim_values)
    std_lpips = np.std(all_lpips_values)

    print("\n" + "=" * 80)
    print("Testing completed! Model performance statistics:")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"\nPerformance metrics:")
    print(f"  Average L1 loss: {avg_l1:.6f}")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    print(f"  Average LPIPS: {avg_lpips:.4f}")
    print(f"\nStandard deviations:")
    print(f"  PSNR: {std_psnr:.2f} dB")
    print(f"  SSIM: {std_ssim:.4f}")
    print(f"  LPIPS: {std_lpips:.4f}")

    if len(all_psnr_values) > 0:
        best_psnr_idx = np.argmax(all_psnr_values)
        worst_psnr_idx = np.argmin(all_psnr_values)

        print(f"\nBest PSNR sample (index {best_psnr_idx}):")
        print(f"  PSNR: {all_psnr_values[best_psnr_idx]:.2f} dB")
        print(f"  SSIM: {all_ssim_values[best_psnr_idx]:.4f}")
        print(f"  LPIPS: {all_lpips_values[best_psnr_idx]:.4f}")

        print(f"\nWorst PSNR sample (index {worst_psnr_idx}):")
        print(f"  PSNR: {all_psnr_values[worst_psnr_idx]:.2f} dB")
        print(f"  SSIM: {all_ssim_values[worst_psnr_idx]:.4f}")
        print(f"  LPIPS: {all_lpips_values[worst_psnr_idx]:.4f}")

    # save result
    model_name = os.path.basename(model_path).replace('.pth', '')
    stats_file = os.path.join(output_dir, f"{model_name}_statistics.txt")

    with open(stats_file, 'w') as f:
        f.write("Model Test Statistics\n")
        f.write("=" * 50 + "\n")
        f.write(f"Test time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Test samples: {len(test_dataset)}\n\n")

        f.write("Performance metrics:\n")
        f.write(f"  Average L1 loss: {avg_l1:.6f}\n")
        f.write(f"  Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"  Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"  Average LPIPS: {avg_lpips:.4f}\n\n")

        f.write("Standard deviations:\n")
        f.write(f"  PSNR: {std_psnr:.2f} dB\n")
        f.write(f"  SSIM: {std_ssim:.4f}\n")
        f.write(f"  LPIPS: {std_lpips:.4f}\n\n")

        f.write("Detailed sample results:\n")
        for i in range(len(all_psnr_values)):
            f.write(f"  Sample{i:03d}: ")
            f.write(f"PSNR={all_psnr_values[i]:.2f}dB, ")
            f.write(f"SSIM={all_ssim_values[i]:.4f}, ")
            f.write(f"LPIPS={all_lpips_values[i]:.4f}\n")

    print(f"\nDetailed statistics saved to: {stats_file}")
    print(f"Result images saved to: {output_dir}")
    print("=" * 80)