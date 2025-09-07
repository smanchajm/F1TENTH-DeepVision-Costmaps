"""
Evaluation script for trained models on F1TENTH dashcam to costmap translation.

This script evaluates trained models on test data and generates visualizations
comparing input images, predicted costmaps, and ground truth targets.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import UNet150, ContextNetwork
from data import ImageToImageDataset, EnhancedImageToImageDataset
from utils import get_device, load_model


def evaluate_unet(model_path: str = "models/unet_best.pth"):
    """
    Evaluate U-Net model and generate visualizations.
    
    Args:
        model_path (str): Path to saved U-Net model
    """
    print("Evaluating U-Net model...")
    
    # Configuration
    BATCH_SIZE = 4
    COMPLEXITY_MULTIPLIER = 4
    
    # Data paths
    INPUT_FOLDER = "Data/Dashcams"
    TARGET_FOLDER = "Data/Costmaps"
    
    # Initialize device
    device = get_device()
    print(f"Using device: {device}")
    
    # Data transforms
    transform_input = transforms.Compose([transforms.ToTensor()])
    transform_target = transforms.Compose([transforms.ToTensor()])
    
    # Dataset and DataLoader
    dataset = ImageToImageDataset(
        input_folder=INPUT_FOLDER,
        target_folder=TARGET_FOLDER,
        transform_input=transform_input,
        transform_target=transform_target
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize and load model
    model = UNet150(
        in_channels=1,
        out_channels=1,
        complexity_multiplier=COMPLEXITY_MULTIPLIER
    )
    
    if os.path.exists(model_path):
        model = load_model(model, model_path, device)
    else:
        print(f"Model file {model_path} not found. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # Generate predictions and visualizations
    with torch.no_grad():
        batch_inputs, batch_targets = next(iter(data_loader))
        batch_inputs = batch_inputs.to(device)
        predictions = model(batch_inputs)
        
        # Move tensors back to CPU for visualization
        batch_inputs = batch_inputs.cpu()
        batch_targets = batch_targets.cpu()
        predictions = predictions.cpu()
        
        # Create visualization
        fig, axes = plt.subplots(3, min(4, BATCH_SIZE), figsize=(16, 12))
        if BATCH_SIZE == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(min(4, BATCH_SIZE)):
            # Input image
            axes[0, i].imshow(batch_inputs[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Ground truth
            axes[1, i].imshow(batch_targets[i].squeeze(), cmap='gray')
            axes[1, i].set_title(f"Ground Truth {i+1}")
            axes[1, i].axis('off')
            
            # Prediction
            axes[2, i].imshow(predictions[i].squeeze(), cmap='gray')
            axes[2, i].set_title(f"Prediction {i+1}")
            axes[2, i].axis('off')
        
        plt.suptitle("U-Net Evaluation Results", fontsize=16)
        plt.tight_layout()
        plt.savefig("unet_evaluation.png", dpi=300, bbox_inches='tight')
        plt.show()


def evaluate_context_network(model_path: str = "models/context_net_best.pth"):
    """
    Evaluate Context Network model and generate visualizations.
    
    Args:
        model_path (str): Path to saved Context Network model
    """
    print("Evaluating Context Network...")
    
    # Configuration
    BATCH_SIZE = 4
    MEAN = 0.2335
    STD = 0.1712
    THRESHOLD = 150
    FILTER_SIZE = 50
    
    # Data paths
    INPUT_FOLDER = "Data/Dashcams"
    TARGET_FOLDER = "Data/Costmaps"
    
    # Initialize device
    device = get_device()
    print(f"Using device: {device}")
    
    # Data transforms
    transform_input = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    transform_target = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    # Enhanced dataset
    dataset = EnhancedImageToImageDataset(
        input_folder=INPUT_FOLDER,
        target_folder=TARGET_FOLDER,
        transform_input=transform_input,
        transform_target=transform_target,
        threshold=THRESHOLD,
        filter_size=FILTER_SIZE
    )
    
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Initialize and load model
    model = ContextNetwork(
        in_channels=1,
        out_channels=1,
        dilation_factors=[1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    
    if os.path.exists(model_path):
        model = load_model(model, model_path, device)
    else:
        print(f"Model file {model_path} not found. Using random weights.")
    
    model.to(device)
    model.eval()
    
    # Generate predictions and visualizations
    with torch.no_grad():
        batch_inputs, batch_targets, masks = next(iter(data_loader))
        batch_inputs = batch_inputs.to(device)
        predictions = model(batch_inputs)
        
        # Move tensors back to CPU for visualization
        batch_inputs = batch_inputs.cpu()
        batch_targets = batch_targets.cpu()
        predictions = predictions.cpu()
        masks = masks.cpu()
        
        # Denormalize inputs for visualization
        batch_inputs_vis = batch_inputs * STD + MEAN
        
        # Create visualization
        fig, axes = plt.subplots(4, min(4, BATCH_SIZE), figsize=(16, 16))
        if BATCH_SIZE == 1:
            axes = axes.reshape(-1, 1)
        
        for i in range(min(4, BATCH_SIZE)):
            # Input image (denormalized)
            axes[0, i].imshow(batch_inputs_vis[i].squeeze(), cmap='gray')
            axes[0, i].set_title(f"Input {i+1}")
            axes[0, i].axis('off')
            
            # Ground truth
            axes[1, i].imshow(batch_targets[i].squeeze(), cmap='gray')
            axes[1, i].set_title(f"Ground Truth {i+1}")
            axes[1, i].axis('off')
            
            # Prediction
            axes[2, i].imshow(predictions[i].squeeze(), cmap='gray')
            axes[2, i].set_title(f"Prediction {i+1}")
            axes[2, i].axis('off')
            
            # Mask
            axes[3, i].imshow(masks[i].squeeze(), cmap='binary')
            axes[3, i].set_title(f"Mask {i+1}")
            axes[3, i].axis('off')
        
        plt.suptitle("Context Network Evaluation Results", fontsize=16)
        plt.tight_layout()
        plt.savefig("context_net_evaluation.png", dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main evaluation function."""
    print("F1TENTH Model Evaluation")
    print("=" * 50)
    
    # Evaluate U-Net
    if os.path.exists("models/unet_best.pth") or os.path.exists("params0.pth"):
        model_path = "models/unet_best.pth" if os.path.exists("models/unet_best.pth") else "params0.pth"
        evaluate_unet(model_path)
    else:
        print("No U-Net model found, skipping evaluation.")
    
    print()
    
    # Evaluate Context Network
    if os.path.exists("models/context_net_best.pth") or os.path.exists("params1.pth"):
        model_path = "models/context_net_best.pth" if os.path.exists("models/context_net_best.pth") else "params1.pth"
        evaluate_context_network(model_path)
    else:
        print("No Context Network model found, skipping evaluation.")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()