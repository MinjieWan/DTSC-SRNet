import numpy as np
import torch
import torch.nn.functional as F
import lpips
from skimage.metrics import structural_similarity as ssim
from math import log10


def calculate_psnr(img1, img2):
    """
    Calculate PSNR metric
    Args:
        img1, img2: numpy arrays (H, W, C) in range [0, 1]
    Returns:
        psnr: float
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * log10(1.0 / np.sqrt(mse))


def calculate_ssim(img1, img2):
    """
    Calculate SSIM metric for color images
    Args:
        img1, img2: numpy arrays (H, W, C) in range [0, 1]
    Returns:
        ssim_value: float
    """
    return ssim(img1, img2, channel_axis=2, data_range=1.0)


def calculate_lpips(img1, img2, lpips_model):
    """
    Calculate LPIPS metric
    Args:
        img1, img2: numpy arrays (H, W, C) in range [0, 1]
        lpips_model: LPIPS model
    Returns:
        lpips_value: float
    """
    with torch.no_grad():
        img1_tensor = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
        img2_tensor = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()

        if next(lpips_model.parameters()).is_cuda:
            img1_tensor = img1_tensor.cuda()
            img2_tensor = img2_tensor.cuda()

        lpips_value = lpips_model(img1_tensor, img2_tensor).item()

    return lpips_value


def crop_edges(tensor, crop_pixels=4):
    """
    Crop edge pixels from tensor
    Args:
        tensor: tensor with shape [B, C, H, W] or [C, H, W]
        crop_pixels: number of pixels to crop from each edge
    Returns:
        cropped tensor
    """
    if tensor.dim() == 4:
        return tensor[:, :, crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
    elif tensor.dim() == 3:
        return tensor[:, crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")


def crop_numpy_edges(array, crop_pixels=4):
    """
    Crop edge pixels from numpy array
    Args:
        array: numpy array with shape [H, W, C] or [H, W]
        crop_pixels: number of pixels to crop from each edge
    Returns:
        cropped array
    """
    if array.ndim == 3:
        return array[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels, :]
    elif array.ndim == 2:
        return array[crop_pixels:-crop_pixels, crop_pixels:-crop_pixels]
    else:
        raise ValueError(f"Unsupported array dimension: {array.ndim}")