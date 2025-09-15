"""
Dataset classes for F1TENTH dashcam to costmap translation.

This module provides dataset implementations for loading paired dashcam images
and their corresponding costmaps for training and evaluation.
"""

import os
import torch
import cv2
from torch.utils.data import Dataset
from PIL import Image


class ImageToImageDataset(Dataset):
    """
    Dataset for paired image-to-image translation.

    Loads dashcam images and their corresponding costmaps for training
    image-to-image translation models.

    Args:
        input_folder (str): Path to folder containing input dashcam images
        target_folder (str): Path to folder containing target costmap images
        transform_input (callable, optional): Transform to apply to input images
        transform_target (callable, optional): Transform to apply to target images
    """

    def __init__(
        self, input_folder, target_folder, transform_input=None, transform_target=None
    ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.input_files = sorted(os.listdir(input_folder))
        self.target_files = sorted(os.listdir(target_folder))
        self.transform_input = transform_input
        self.transform_target = transform_target

        # Ensure equal number of input and target images
        assert len(self.input_files) == len(self.target_files), (
            f"Mismatch: {len(self.input_files)} inputs vs {len(self.target_files)} targets"
        )

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load input image
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        input_image = Image.open(input_path)

        # Load target image
        target_path = os.path.join(self.target_folder, self.target_files[idx])
        target_image = Image.open(target_path)

        # Apply transforms if provided
        if self.transform_input:
            input_image = self.transform_input(input_image)
        if self.transform_target:
            target_image = self.transform_target(target_image)

        return input_image, target_image


class EnhancedImageToImageDataset(Dataset):
    """
    Enhanced dataset with contrast enhancement and morphological operations.

    This dataset includes additional preprocessing steps like contrast enhancement
    and morphological operations for improved training.

    Args:
        input_folder (str): Path to folder containing input dashcam images
        target_folder (str): Path to folder containing target costmap images
        transform_input (callable, optional): Transform to apply to input images
        transform_target (callable, optional): Transform to apply to target images
        threshold (int): Threshold for binary operations (default: 30)
        filter_size (int): Size of morphological filter (default: 26)
    """

    def __init__(
        self,
        input_folder,
        target_folder,
        transform_input=None,
        transform_target=None,
        threshold=30,
        filter_size=26,
    ):
        self.input_folder = input_folder
        self.target_folder = target_folder
        self.input_files = sorted(os.listdir(input_folder))
        self.target_files = sorted(os.listdir(target_folder))
        self.transform_input = transform_input
        self.transform_target = transform_target
        self.threshold = threshold
        self.filter_size = filter_size

        # Ensure equal number of input and target images
        assert len(self.input_files) == len(self.target_files), (
            f"Mismatch: {len(self.input_files)} inputs vs {len(self.target_files)} targets"
        )

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load input image as grayscale
        input_path = os.path.join(self.input_folder, self.input_files[idx])
        input_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        # Load target image as grayscale
        target_path = os.path.join(self.target_folder, self.target_files[idx])
        target_image = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

        # Generate mask using OpenCV morphological operations
        _, binary_image = cv2.threshold(
            target_image, self.threshold, 255, cv2.THRESH_BINARY_INV
        )
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.filter_size, self.filter_size)
        )
        mask = cv2.dilate(binary_image, kernel, iterations=1)

        # Convert to tensors and normalize to [0, 1]
        input_tensor = torch.from_numpy(input_image / 255.0).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target_image / 255.0).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        return input_tensor, target_tensor, mask_tensor
