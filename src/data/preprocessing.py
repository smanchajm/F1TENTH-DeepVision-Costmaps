"""
Preprocessing utilities for F1TENTH track data and costmap generation.

This module contains functions to process track maps, GPS coordinates,
and generate costmaps from raw sensor data.
"""

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from scipy import ndimage
from typing import Tuple, List


def to_float(unit: str, decimal: str) -> float:
    """
    Convert unit and decimal parts from CSV to float.
    
    Args:
        unit (str): Unit part of the number
        decimal (str): Decimal part of the number
        
    Returns:
        float: Combined float value
    """
    return float(f"{unit}.{decimal}")


def rotate_image(img: np.ndarray, angle: float, pivot: Tuple[int, int]) -> np.ndarray:
    """
    Rotate image around a specified pivot point.
    
    Args:
        img (np.ndarray): Input image array
        angle (float): Rotation angle in degrees
        pivot (Tuple[int, int]): Pivot point (x, y)
        
    Returns:
        np.ndarray: Rotated image
    """
    pad_x = [img.shape[1] - pivot[0], pivot[0]]
    pad_y = [img.shape[0] - pivot[1], pivot[1]]
    
    # Pad image to avoid cropping during rotation
    img_padded = np.pad(
        img, [pad_y, pad_x, [0, 0]], 
        'constant', constant_values=img[0, 0, 0]
    )
    
    # Rotate and crop back to original region
    img_rotated = ndimage.rotate(img_padded, angle, reshape=False)
    return img_rotated[pad_y[0]:-pad_y[1], pad_x[0]:-pad_x[1], :]


def crop_image(img: np.ndarray, width: int, height: int, 
               point: Tuple[int, int]) -> np.ndarray:
    """
    Crop rectangular region from image centered on a point.
    
    Args:
        img (np.ndarray): Input image
        width (int): Width of crop region
        height (int): Height of crop region  
        point (Tuple[int, int]): Center point (x, y)
        
    Returns:
        np.ndarray: Cropped image region
    """
    x_center, y_center = point
    
    # Extract rectangle above the point, centered on x-axis
    cropped = img[
        y_center - height:y_center,
        x_center - width // 2:x_center + width // 2
    ]
    
    return cropped


def point_angle_to_map(point: Tuple[int, int], angle: float, 
                      img: np.ndarray, size: int = 150) -> np.ndarray:
    """
    Extract oriented rectangular region from map at specified point and angle.
    
    Args:
        point (Tuple[int, int]): Center point coordinates
        angle (float): Orientation angle in radians
        img (np.ndarray): Input map image
        size (int): Size of square region to extract
        
    Returns:
        np.ndarray: Extracted and oriented image region
    """
    # Convert angle to degrees and rotate image
    img_rotated = rotate_image(img, np.rad2deg(angle), point)
    
    # Crop square region
    img_cropped = crop_image(img_rotated, size, size, point)
    
    return img_cropped


def coordinates_to_pixel(x_abs: float, y_abs: float) -> Tuple[int, int]:
    """
    Convert GPS coordinates to pixel coordinates on track map.
    
    These parameters are manually calibrated for the specific track map.
    
    Args:
        x_abs (float): X coordinate in GPS system
        y_abs (float): Y coordinate in GPS system
        
    Returns:
        Tuple[int, int]: Pixel coordinates (x, y)
    """
    # Calibration parameters for track map
    ZERO_X = 785
    ZERO_Y = 359.5
    UNIT_X = -85
    UNIT_Y = -85
    
    pixel_x = int(ZERO_X + x_abs * UNIT_X)
    pixel_y = int(ZERO_Y + y_abs * UNIT_Y)
    
    return pixel_x, pixel_y


def load_sensor_data(csv_path: str, useful_columns: List[int] = None) -> pd.DataFrame:
    """
    Load sensor data from CSV file.
    
    Args:
        csv_path (str): Path to CSV file
        useful_columns (List[int], optional): Column indices to load
        
    Returns:
        pd.DataFrame: Loaded sensor data
    """
    if useful_columns is None:
        useful_columns = list(range(21))  # Default first 21 columns
    
    return pd.read_csv(
        csv_path, 
        usecols=useful_columns, 
        header=None, 
        dtype=str  # Keep as string to preserve leading zeros
    )


def enhance_contrast(image: np.ndarray, alpha: float = 1.5, beta: int = 0) -> np.ndarray:
    """
    Enhance image contrast using linear transformation.
    
    Args:
        image (np.ndarray): Input image
        alpha (float): Contrast control (1.0-3.0)
        beta (int): Brightness control (0-100)
        
    Returns:
        np.ndarray: Contrast enhanced image
    """
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)