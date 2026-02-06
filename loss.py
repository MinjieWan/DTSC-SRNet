import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips


class CombinedLoss(nn.Module):
    def __init__(self, lpips_net_type='alex', alpha=1.0, beta=1.0, gamma=1.0):
        super().__init__()
        self.lpips_loss = lpips.LPIPS(net=lpips_net_type, spatial=False)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lpips_loss.to(self.device)

        for param in self.lpips_loss.parameters():
            param.requires_grad = False

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        pred = pred.to(self.device)
        target = target.to(self.device)

        mse_loss = F.mse_loss(pred, target)
        psnr_loss = 10 * torch.log10(1.0 / (mse_loss + 1e-8))
        psnr_component = -psnr_loss

        ssim_value = self.calculate_ssim_loss(pred, target)
        ssim_component = 1.0 - ssim_value

        pred_norm = (pred - 0.5) * 2
        target_norm = (target - 0.5) * 2
        lpips_component = self.lpips_loss(pred_norm, target_norm, normalize=False)

        combined_loss = (self.alpha * psnr_component +
                         self.beta * ssim_component +
                         self.gamma * lpips_component)

        return combined_loss

    def calculate_ssim_loss(self, pred, target):
        """Calculate differentiable SSIM loss"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = F.avg_pool2d(pred, 3, 1, 1)
        mu_y = F.avg_pool2d(target, 3, 1, 1)

        sigma_x = F.avg_pool2d(pred ** 2, 3, 1, 1) - mu_x ** 2
        sigma_y = F.avg_pool2d(target ** 2, 3, 1, 1) - mu_y ** 2
        sigma_xy = F.avg_pool2d(pred * target, 3, 1, 1) - mu_x * mu_y

        ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        ssim_value = ssim_n / ssim_d
        return torch.mean(ssim_value)