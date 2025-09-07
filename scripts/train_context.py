"""
Training script for Context Network on F1TENTH dashcam to costmap translation.

This script trains the ContextNetwork model for image-to-image translation from
dashcam images to navigation costmaps with enhanced contrast preprocessing.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models import ContextNetwork
from data import EnhancedImageToImageDataset
from utils import get_device, save_model, init_wandb, log_metrics


def train_context_network():
    """Main training function for Context Network."""
    
    # Configuration
    BATCH_SIZE = 1
    NUM_EPOCHS = 3
    LEARNING_RATE = 0.001
    
    # Dataset configuration
    MEAN = 0.2335
    STD = 0.1712
    THRESHOLD = 150
    FILTER_SIZE = 50
    
    # Data paths
    INPUT_FOLDER = "Data/Dashcams"
    TARGET_FOLDER = "Data/Costmaps"
    MODEL_SAVE_PATH = "models/context_net_best.pth"
    
    # Initialize device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize W&B logging
    init_wandb(
        name="context_net_training",
        config={
            "model": "ContextNetwork",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "threshold": THRESHOLD,
            "filter_size": FILTER_SIZE,
            "normalization_mean": MEAN,
            "normalization_std": STD
        }
    )
    
    # Data transforms (with normalization for enhanced dataset)
    transform_input = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    transform_target = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()
    ])
    
    # Enhanced dataset with morphological operations
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
        shuffle=True
    )
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Initialize model with identity initialization
    model = ContextNetwork(
        in_channels=1,
        out_channels=1,
        dilation_factors=[1, 1, 1, 1, 1, 1, 1, 1, 1]
    )
    model.to(device)
    
    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    model.train()
    
    for epoch in range(NUM_EPOCHS):
        print(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")
        running_loss = 0.0
        
        for batch_inputs, batch_targets, masks in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            # Move data to device
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            masks = masks.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(batch_inputs)
            
            # Calculate loss (using targets directly, masks available if needed)
            loss = criterion(outputs, batch_targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Accumulate loss
            running_loss += loss.item()
            
            # Log to W&B
            log_metrics({"loss": loss.item()})
        
        # Print epoch statistics
        avg_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Average Loss: {avg_loss:.4f}")
        
        # Log epoch metrics
        log_metrics({"epoch_avg_loss": avg_loss, "epoch": epoch + 1})
    
    # Save final model
    save_model(model, MODEL_SAVE_PATH, epoch=NUM_EPOCHS)
    
    print("Training completed!")


if __name__ == "__main__":
    train_context_network()