import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


def load_burst_images(burst_folder):
    """Load burst images from folder"""
    image_files = sorted(glob.glob(os.path.join(burst_folder, "lr_frame_*.bmp")))

    if len(image_files) != 16:
        print(f"Warning: {burst_folder} has only {len(image_files)} images, expected 16")

    images = []
    for file_path in image_files[:16]:
        img = Image.open(file_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        images.append(img_array)

    images_tensor = torch.tensor(np.array(images), dtype=torch.float32)
    images_tensor = images_tensor.permute(0, 3, 1, 2)

    return images_tensor


class SRDataset(Dataset):
    def __init__(self, burst_dir, hr_dir, low_res_size=(96, 96), high_res_size=(384, 384), n_frames=16):
        self.burst_dir = burst_dir
        self.hr_dir = hr_dir
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size
        self.n_frames = n_frames

        self.burst_folders = sorted(glob.glob(os.path.join(burst_dir, "*")))
        self.hr_files = sorted(glob.glob(os.path.join(hr_dir, "*.bmp")))

        self.burst_to_hr = {}
        for burst_folder in self.burst_folders:
            folder_name = os.path.basename(burst_folder)
            folder_id = folder_name.split('-')[0]
            hr_path = os.path.join(hr_dir, f"{folder_id}.bmp")

            if os.path.exists(hr_path):
                self.burst_to_hr[burst_folder] = hr_path
            else:
                print(f"Warning: Cannot find corresponding HR image {hr_path}")

    def __len__(self):
        return len(self.burst_to_hr)

    def __getitem__(self, idx):
        burst_folder = list(self.burst_to_hr.keys())[idx]
        hr_path = self.burst_to_hr[burst_folder]

        lr_images = load_burst_images(burst_folder)

        hr_img = Image.open(hr_path).convert('RGB')
        hr_img = hr_img.resize(self.high_res_size, Image.Resampling.LANCZOS)
        hr_array = np.array(hr_img, dtype=np.float32) / 255.0
        hr_array = hr_array.transpose(2, 0, 1)

        lr_tensor = torch.tensor(np.array(lr_images), dtype=torch.float32)
        hr_tensor = torch.tensor(hr_array, dtype=torch.float32)

        return lr_tensor, hr_tensor