# burst_synthesis_v2.py
import torch
import random
import cv2
import numpy as np
import os
import math
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


def generate_psf_kernel(kernel_size=15, psf_type='gaussian', **kwargs):
    """Generate Point Spread Function (PSF) kernel"""
    if psf_type == 'gaussian':
        sigma = kwargs.get('sigma', 3.2)  # 8x downsampling corresponds to high blur
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        for i in range(kernel_size):
            for j in range(kernel_size):
                x = i - center
                y = j - center
                kernel[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
        kernel /= kernel.sum()
    elif psf_type == 'motion':
        length = kwargs.get('length', 15)  # Motion blur length
        angle = kwargs.get('angle', 0) * np.pi / 180  # Horizontal direction
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        for i in range(length):
            x = int(center + cos_val * (i - length / 2))
            y = int(center + sin_val * (i - length / 2))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[x, y] = 1.0
        if kernel.sum() > 0:
            kernel /= kernel.sum()
        else:
            kernel[center, center] = 1.0
    else:
        kernel = np.eye(kernel_size) / kernel_size
    return kernel


def apply_psf_blur(image, psf_kernel):
    """Apply PSF blur"""
    if len(image.shape) == 3:  # Color image
        blurred = np.zeros_like(image)
        for c in range(image.shape[2]):
            blurred[..., c] = convolve2d(image[..., c], psf_kernel, mode='same')
    else:  # Grayscale image
        blurred = convolve2d(image, psf_kernel, mode='same')
    return blurred


def get_tmat(image_shape, translation, theta, shear_values, scale_factors):
    """Generate transformation matrix"""
    im_h, im_w = image_shape
    t_mat = np.identity(3)
    t_mat[0, 2] = translation[0]
    t_mat[1, 2] = translation[1]
    t_rot = cv2.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
    t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))
    t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                        [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                        [0.0, 0.0, 1.0]])
    t_scale = np.array([[scale_factors[0], 0.0, 0.0],
                        [0.0, scale_factors[1], 0.0],
                        [0.0, 0.0, 1.0]])
    t_mat = t_scale @ t_rot @ t_shear @ t_mat
    return t_mat[:2, :]


def generate_normal_motion_vectors(burst_size, base_translation=0.5):
    """Generate normal motion vectors (similar to Brownian motion)"""
    vectors = [(0.0, 0.0)]  # First frame has no motion

    # Generate random walk path
    x, y = 0.0, 0.0
    for i in range(1, burst_size):
        # Normal motion: small continuous changes
        dx = random.uniform(-base_translation, base_translation)
        dy = random.uniform(-base_translation, base_translation)
        x += dx
        y += dy

        # Limit maximum offset (3x base translation)
        max_offset = base_translation * 3
        x = max(-max_offset, min(max_offset, x))
        y = max(-max_offset, min(max_offset, y))

        vectors.append((x, y))

    return vectors


def single2rgb_burst(image, burst_size=16, downsample_factor=8,
                     psf_kernel=None, noise_level=0.01):
    """Generate RGB burst from single RGB image"""

    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose(1, 2, 0) if image.shape[0] == 3 else image.numpy()

    if image.max() <= 1.0:
        image = (image * 255.0).astype(np.uint8)

    h, w = image.shape[:2]
    burst = []
    motion_vectors = []

    # Generate normal motion vectors
    base_translation = 1.0 * downsample_factor  # Base translation in HR space
    hr_vectors = generate_normal_motion_vectors(burst_size, base_translation)

    for i in range(burst_size):
        # Get current frame's motion vector
        tx, ty = hr_vectors[i]

        # Add moderate scaling and rotation
        if i == 0:
            scale = 1.0
            rotation = 0.0
        else:
            scale = 1.0 + random.uniform(-0.03, 0.03)  # Moderate scaling ±1%
            rotation = random.uniform(-1.5, 1.5)  # Moderate rotation ±0.5 degrees

        # Build transformation matrix (with moderate shear)
        shear_x = random.uniform(-0.001, 0.001) if i > 0 else 0.0
        shear_y = random.uniform(-0.001, 0.001) if i > 0 else 0.0

        # Generate transformation matrix
        t_mat = get_tmat(
            image_shape=(h, w),
            translation=(tx, ty),
            theta=rotation,
            shear_values=(shear_x, shear_y),
            scale_factors=(scale, scale)
        )

        # Apply transformation
        if len(image.shape) == 3:  # Color
            transformed = cv2.warpAffine(
                image, t_mat, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )
        else:  # Grayscale
            transformed = cv2.warpAffine(
                image, t_mat, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

        # Apply high blur PSF
        if psf_kernel is not None:
            transformed = apply_psf_blur(transformed, psf_kernel)

        # 8x downsampling
        new_h, new_w = h // downsample_factor, w // downsample_factor
        downsampled = cv2.resize(
            transformed,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        # Add moderate noise
        if noise_level > 0:
            noise = np.random.randn(*downsampled.shape) * noise_level * 255
            downsampled = downsampled.astype(np.float32) + noise
            downsampled = np.clip(downsampled, 0, 255).astype(np.uint8)

        burst.append(downsampled)

        # Calculate LR space motion vectors
        lr_tx = tx / downsample_factor
        lr_ty = ty / downsample_factor
        motion_vectors.append((lr_tx, lr_ty))

    # Convert to tensor format
    burst_tensor = []
    for frame in burst:
        if len(frame.shape) == 3:  # Color
            frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float() / 255.0
        else:  # Grayscale
            frame_tensor = torch.from_numpy(frame).unsqueeze(0).float() / 255.0
        burst_tensor.append(frame_tensor)

    burst_tensor = torch.stack(burst_tensor)

    return burst_tensor, motion_vectors, hr_vectors


def process_gt_image(input_path, output_dir, burst_size=16,
                     downsample_factor=8, psf_config=None):
    """Process single GT image"""

    # Read image
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Image does not exist: {input_path}")

    img = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {input_path}")

    # Ensure image is 384x384
    if img.shape[0] != 384 or img.shape[1] != 384:
        print(f"Resizing image: {img.shape} -> (384, 384)")
        img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)

    # Generate PSF kernel (high blur)
    psf_kernel = None
    if psf_config:
        psf_kernel = generate_psf_kernel(**psf_config)

    # Generate burst sequence
    burst_tensor, motion_vectors, hr_vectors = single2rgb_burst(
        image=img,
        burst_size=burst_size,
        downsample_factor=downsample_factor,
        psf_kernel=psf_kernel,
        noise_level=0.01  # Moderate noise
    )

    # Create output directory
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_subdir = os.path.join(output_dir, base_name)
    os.makedirs(output_subdir, exist_ok=True)

    # Save LR images
    for i in range(burst_size):
        frame = burst_tensor[i].numpy()
        if frame.shape[0] == 3:  # Color
            frame = (frame.transpose(1, 2, 0) * 255.0).astype(np.uint8)
        else:  # Grayscale
            frame = (frame.squeeze() * 255.0).astype(np.uint8)

        cv2.imwrite(
            os.path.join(output_subdir, f"lr_frame_{i:02d}.bmp"),
            frame
        )

    # Save motion vectors (LR space)
    with open(os.path.join(output_subdir, "motion_vectors.txt"), "w") as f:
        for tx, ty in motion_vectors:
            f.write(f"({tx:.4f}, {ty:.4f})\n")

    # Save HR space motion vectors (for reference)
    with open(os.path.join(output_subdir, "hr_motion_vectors.txt"), "w") as f:
        for tx, ty in hr_vectors:
            f.write(f"({tx:.4f}, {ty:.4f})\n")

    # Save PSF information
    if psf_kernel is not None:
        psf_info_path = os.path.join(output_subdir, "psf_info.txt")
        with open(psf_info_path, "w") as f:
            f.write(f"PSF type: {psf_config.get('psf_type', 'gaussian')}\n")
            f.write(f"Kernel size: {psf_config.get('kernel_size', 15)}\n")
            if 'sigma' in psf_config:
                f.write(f"Sigma: {psf_config['sigma']:.2f}\n")

        # Visualize PSF kernel
        psf_visual = (psf_kernel / psf_kernel.max() * 255).astype(np.uint8)
        psf_visual = cv2.resize(psf_visual, (384, 384), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(output_subdir, "psf_kernel.bmp"), psf_visual)

    # Generate reference image (upsampled first frame)
    first_frame = burst_tensor[0].numpy()
    if first_frame.shape[0] == 3:
        first_frame = (first_frame.transpose(1, 2, 0) * 255.0).astype(np.uint8)
    else:
        first_frame = (first_frame.squeeze() * 255.0).astype(np.uint8)

    upsampled = cv2.resize(
        first_frame,
        (384, 384),
        interpolation=cv2.INTER_CUBIC
    )
    cv2.imwrite(os.path.join(output_subdir, "reference_upsampled.bmp"), upsampled)

    print(f"Processed: {base_name} -> {output_subdir}")
    print(f"  Output: {burst_size} frames, Size: {burst_tensor.shape[-2]}x{burst_tensor.shape[-1]}")

    return output_subdir


def process_gt_folder(gt_folder="gt", output_dir="lr_output",
                      burst_size=16, downsample_factor=8,
                      psf_config=None):
    """Process entire folder of GT images"""

    if not os.path.isdir(gt_folder):
        raise ValueError(f"GT folder does not exist: {gt_folder}")

    os.makedirs(output_dir, exist_ok=True)

    # Process all image files
    image_extensions = ['.bmp', '.png', '.jpg', '.jpeg', '.tiff']
    processed_count = 0

    for filename in sorted(os.listdir(gt_folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext in image_extensions:
            input_path = os.path.join(gt_folder, filename)
            try:
                process_gt_image(
                    input_path=input_path,
                    output_dir=output_dir,
                    burst_size=burst_size,
                    downsample_factor=downsample_factor,
                    psf_config=psf_config
                )
                processed_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    print(f"\nProcessing completed! Total processed images: {processed_count}")
    print(f"Output directory: {output_dir}")


def main():
    """Main function: configure parameters and process"""

    # Configure PSF parameters (high blur)
    psf_config = {
        'psf_type': 'gaussian',
        'kernel_size': 31,  # Large kernel for high blur
        'sigma': 3.2,  # High blur sigma for 8x downsampling
    }

    # Or use motion blur PSF
    # psf_config = {
    #     'psf_type': 'motion',
    #     'kernel_size': 31,
    #     'length': 15,
    #     'angle': 0,
    # }

    # Input and output paths
    gt_folder = "gtlargetest384"  # Input folder
    output_dir = "lr_output_testv8+"  # Output folder

    # Processing parameters
    burst_size = 16
    downsample_factor = 8

    # Check input folder
    if not os.path.exists(gt_folder):
        print(f"Warning: Input folder {gt_folder} does not exist")
        print("Please ensure GT images are in the correct folder")
        return

    # Process entire folder
    process_gt_folder(
        gt_folder=gt_folder,
        output_dir=output_dir,
        burst_size=burst_size,
        downsample_factor=downsample_factor,
        psf_config=psf_config
    )

    # Print configuration information
    print("\nConfiguration parameters:")
    print(f"  Input folder: {gt_folder}")
    print(f"  Output folder: {output_dir}")
    print(f"  Burst size: {burst_size}")
    print(f"  Downsample factor: {downsample_factor}")
    print(f"  PSF configuration: {psf_config}")


if __name__ == "__main__":
    main()