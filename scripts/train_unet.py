"""
Training script for U-Net model on F1TENTH dashcam to costmap translation.

This script trains the UNet150 model for image-to-image translation from
dashcam images to navigation costmaps.
"""

import os
import sys
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models import UNet150
from data import ImageToImageDataset
from utils import get_device, save_model, init_wandb, log_metrics


def train_unet():
    """Main training function for U-Net model."""

    # Configuration
    BATCH_SIZE = 5
    NUM_EPOCHS = 1
    LEARNING_RATE = 0.001
    COMPLEXITY_MULTIPLIER = 4

    # Data paths
    INPUT_FOLDER = "Data/Dashcams"
    TARGET_FOLDER = "Data/Costmaps"
    MODEL_SAVE_PATH = "models/unet_best.pth"

    # Initialize device
    device = get_device()
    print(f"Using device: {device}")

    # Initialize W&B logging
    init_wandb(
        name="unet_training",
        config={
            "model": "UNet150",
            "batch_size": BATCH_SIZE,
            "epochs": NUM_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "complexity_multiplier": COMPLEXITY_MULTIPLIER,
        },
    )

    # Data transforms
    transform_input = transforms.Compose([transforms.ToTensor()])
    transform_target = transforms.Compose([transforms.ToTensor()])

    # Dataset and DataLoader
    dataset = ImageToImageDataset(
        input_folder=INPUT_FOLDER,
        target_folder=TARGET_FOLDER,
        transform_input=transform_input,
        transform_target=transform_target,
    )

    data_loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"Dataset size: {len(dataset)} samples")

    # Initialize model
    model = UNet150(
        in_channels=1, out_channels=1, complexity_multiplier=COMPLEXITY_MULTIPLIER
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

        for batch_inputs, batch_targets in tqdm(data_loader, desc=f"Epoch {epoch + 1}"):
            # Move data to device
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_inputs)

            # Calculate loss
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
    train_unet()
