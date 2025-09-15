"""
U-Net implementation for F1TENTH dashcam to costmap translation.

This module contains a modified U-Net architecture optimized for 150x150 pixel images.
The model performs image-to-image translation from dashcam images to navigation costmaps.
"""

import torch
import torch.nn as nn


class UNet150(nn.Module):
    """
    U-Net architecture designed for 150x150 pixel image-to-image translation.

    The network follows the classic U-Net structure with encoder-decoder paths
    and skip connections for preserving spatial information.

    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        out_channels (int): Number of output channels (default: 1 for costmap)
        complexity_multiplier (int): Multiplier for feature maps complexity (default: 1)
    """

    def __init__(self, in_channels=1, out_channels=1, complexity_multiplier=1):
        super(UNet150, self).__init__()
        self.cm = complexity_multiplier

        # Encoder path
        self.encoder1 = self._double_conv(in_channels, self.cm * 16)
        self.pool1 = nn.MaxPool2d(2)  # 150 -> 75

        self.encoder2 = self._double_conv(self.cm * 16, self.cm * 32)
        self.pool2 = nn.MaxPool2d(2)  # 75 -> 37

        # Bottleneck
        self.bottleneck = self._double_conv(self.cm * 32, self.cm * 64)

        # Decoder path
        self.upconv2 = nn.ConvTranspose2d(
            self.cm * 64, self.cm * 32, kernel_size=2, stride=2, output_padding=1
        )  # 37 -> 75
        self.decoder2 = self._double_conv(self.cm * 64, self.cm * 32)

        self.upconv1 = nn.ConvTranspose2d(
            self.cm * 32, self.cm * 16, kernel_size=2, stride=2, output_padding=0
        )  # 75 -> 150
        self.decoder1 = self._double_conv(self.cm * 32, self.cm * 16)

        # Output layer
        self.final_conv = nn.Conv2d(self.cm * 16, out_channels, kernel_size=1)

    def _double_conv(self, in_channels, out_channels):
        """
        Double convolution block with ReLU activation.

        Args:
            in_channels (int): Input channels
            out_channels (int): Output channels

        Returns:
            nn.Sequential: Two consecutive conv-relu blocks
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Forward pass through the U-Net.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, 150, 150)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, 150, 150)
        """
        # Encoder path
        enc1 = self.encoder1(x)  # 150x150
        pool1 = self.pool1(enc1)  # 75x75

        enc2 = self.encoder2(pool1)  # 75x75
        pool2 = self.pool2(enc2)  # 37x37

        # Bottleneck
        bottleneck = self.bottleneck(pool2)  # 37x37

        # Decoder path with skip connections
        up2 = self.upconv2(bottleneck)  # 37 -> 75
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))

        up1 = self.upconv1(dec2)  # 75 -> 150
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))

        # Output
        output = self.final_conv(dec1)  # 150x150
        return output
