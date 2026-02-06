import torch
import torch.nn as nn


def get_activation(activation, activation_params=None, num_channels=None):
    """Get activation function"""
    if activation_params is None:
        activation_params = {}

    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'lrelu':
        return nn.LeakyReLU(negative_slope=activation_params.get('negative_slope', 0.1), inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=num_channels)
    elif activation == 'none':
        return None
    else:
        raise Exception('Unknown activation {}'.format(activation))


def save_image(tensor, filename):
    """Save tensor as image file"""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    import numpy as np
    from PIL import Image

    img_array = tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255.0
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)

    img = Image.fromarray(img_array)
    img.save(filename)
    return img_array


def calculate_psnr_ssim(img1, img2):
    """Calculate PSNR and SSIM between two images"""
    if img1 is None or img2 is None:
        raise ValueError("Cannot read images, please check file paths")

    import numpy as np
    from skimage.metrics import structural_similarity as ssim

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr_value = float('inf')
    else:
        max_pixel = 255.0
        psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))

    if len(img1.shape) == 3 and img1.shape[2] == 3:
        ssim_values = []
        for i in range(3):
            channel_ssim = ssim(img1[:, :, i], img2[:, :, i],
                                data_range=255, win_size=7)
            ssim_values.append(channel_ssim)
        ssim_value = np.mean(ssim_values)
    else:
        ssim_value = ssim(img1, img2, data_range=255, win_size=7)

    return psnr_value, ssim_value