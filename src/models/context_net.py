"""
Context Network implementation for F1TENTH dashcam to costmap translation.

This module contains a dilated convolution network designed to capture 
contextual information for navigation costmap generation.
"""

import torch
import torch.nn as nn


def identity_initialization(layer):
    """
    Apply identity initialization to a Conv2d layer.
    
    This initialization sets the network to approximate an identity function
    initially, which can help with training stability.
    
    Args:
        layer (nn.Conv2d): Convolution layer to initialize
    """
    if isinstance(layer, nn.Conv2d):
        # Initialize all weights to zero
        nn.init.constant_(layer.weight, 0)
        
        # Find center of kernel
        kernel_size = layer.kernel_size[0]
        mid = kernel_size // 2
        
        # Set diagonal elements to 1 for identity mapping
        for i in range(min(layer.in_channels, layer.out_channels)):
            layer.weight.data[i, i, mid, mid] = 1
        
        # Initialize biases to zero
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class ContextNetwork(nn.Module):
    """
    Context Network using dilated convolutions for semantic segmentation.
    
    This network uses a series of dilated convolutions to capture multi-scale
    contextual information while preserving spatial resolution.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels  
        dilation_factors (list): List of dilation factors for each layer
        output_channels (list): List of output channels for each layer
    """
    
    def __init__(
        self, 
        in_channels=1, 
        out_channels=1,
        dilation_factors=[1, 1, 1, 2, 4, 8, 16, 32, 1],
        output_channels=[48, 64, 128, 32, 32, 32, 32, 32, 1]
    ):
        super(ContextNetwork, self).__init__()
        
        layers = []
        prev_channels = in_channels
        
        for i, (dilation, out_ch) in enumerate(zip(dilation_factors, output_channels)):
            # Use 3x3 kernel for all layers except the last one (1x1)
            kernel_size = 3 if i < len(dilation_factors) - 1 else 1
            
            # Calculate padding to maintain spatial dimensions
            padding = (dilation * (kernel_size - 1)) // 2
            
            # Create convolution layer
            conv_layer = nn.Conv2d(
                in_channels=prev_channels,
                out_channels=out_ch,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
            
            # Apply identity initialization
            identity_initialization(conv_layer)
            layers.append(conv_layer)
            
            # Add ReLU activation for all layers except the last
            if i < len(dilation_factors) - 1:
                layers.append(nn.ReLU(inplace=True))
            
            prev_channels = out_ch
        
        # Add final sigmoid activation
        layers.append(nn.Sigmoid())
        
        self.context_module = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass through the context network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        return self.context_module(x)