"""
Training utilities and helper functions.

This module provides common training utilities including device detection,
model saving/loading, and training loops.
"""

import os
import torch
import torch.nn as nn
from typing import Optional
import wandb


def get_device() -> str:
    """
    Get the best available device (CUDA if available, otherwise CPU).
    
    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def save_model(model: nn.Module, path: str, epoch: Optional[int] = None) -> None:
    """
    Save model state dict to file.
    
    Args:
        model (nn.Module): PyTorch model to save
        path (str): Path to save the model
        epoch (int, optional): Current epoch number
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model state
    save_dict = {'state_dict': model.state_dict()}
    if epoch is not None:
        save_dict['epoch'] = epoch
    
    torch.save(save_dict, path)
    print(f"Model saved to {path}")


def load_model(model: nn.Module, path: str, device: str = 'cpu') -> nn.Module:
    """
    Load model state dict from file.
    
    Args:
        model (nn.Module): PyTorch model to load weights into
        path (str): Path to saved model
        device (str): Device to load model on
        
    Returns:
        nn.Module: Model with loaded weights
    """
    checkpoint = torch.load(path, map_location=device)
    
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"Model loaded from {path}")
    return model


def init_wandb(project_name: str = "F1TENTH", **kwargs) -> None:
    """
    Initialize Weights & Biases logging.
    
    Args:
        project_name (str): W&B project name
        **kwargs: Additional arguments for wandb.init()
    """
    wandb.init(project=project_name, **kwargs)


def log_metrics(metrics: dict) -> None:
    """
    Log metrics to Weights & Biases.
    
    Args:
        metrics (dict): Dictionary of metric names and values
    """
    wandb.log(metrics)