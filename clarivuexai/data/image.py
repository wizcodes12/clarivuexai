"""
Image data handler for ClarivueXAI.

This module provides a data handler for image data.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np

from clarivuexai.core.base import BaseDataHandler
from clarivuexai.core.registry import register_data_handler


@register_data_handler('image')
class ImageDataHandler(BaseDataHandler):
    """
    Data handler for image data.
    
    This class provides methods for preprocessing and transforming
    image data for explanation.
    """
    
    def __init__(
        self, 
        target_size: Optional[Tuple[int, int]] = None,
        grayscale: bool = False,
        normalize: bool = True,
        data_format: str = 'channels_last'
    ):
        """
        Initialize an ImageDataHandler.
        
        Args:
            target_size: Target size for images (height, width)
            grayscale: Whether to convert images to grayscale
            normalize: Whether to normalize pixel values to [0, 1]
            data_format: Image data format ('channels_last' or 'channels_first')
        """
        super().__init__('image')
        self.target_size = target_size
        self.grayscale = grayscale
        self.normalize = normalize
        self.data_format = data_format
    
    def preprocess(self, data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Preprocess the input image data.
        
        Args:
            data: Input image data
            
        Returns:
            Preprocessed image data
        """
        # Convert list of images to numpy array
        if isinstance(data, list):
            data = np.array(data)
        
        # Ensure data is a numpy array
        if not isinstance(data, np.ndarray):
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Handle single image
        if len(data.shape) == 2 or (len(data.shape) == 3 and (data.shape[2] == 1 or data.shape[2] == 3)):
            data = np.expand_dims(data, axis=0)
        
        # Resize images if target_size is specified
        if self.target_size is not None:
            try:
                from PIL import Image
                resized_images = []
                for i in range(data.shape[0]):
                    img = Image.fromarray(data[i].astype('uint8'))
                    img = img.resize(self.target_size)
                    resized_images.append(np.array(img))
                data = np.array(resized_images)
            except ImportError:
                raise ImportError("PIL is required for image resizing. "
                                 "Install it with 'pip install pillow'.")
        
        # Convert to grayscale if specified
        if self.grayscale and data.shape[-1] == 3:
            # Simple grayscale conversion: average of RGB channels
            data = np.mean(data, axis=-1, keepdims=True)
        
        # Normalize pixel values if specified
        if self.normalize:
            data = data.astype('float32') / 255.0
        
        # Convert data format if necessary
        if self.data_format == 'channels_first' and data.shape[-1] in [1, 3]:
            # Convert from channels_last to channels_first
            data = np.transpose(data, (0, 3, 1, 2))
        elif self.data_format == 'channels_last' and data.shape[1] in [1, 3]:
            # Convert from channels_first to channels_last
            data = np.transpose(data, (0, 2, 3, 1))
        
        return data
    
    def transform(self, data: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
        """
        Transform the input image data into a format suitable for explanation.
        
        Args:
            data: Input image data
            
        Returns:
            Transformed image data
        """
        # Preprocess the data
        preprocessed_data = self.preprocess(data)
        
        # Flatten images for some explainers
        if len(preprocessed_data.shape) == 4:
            # Keep batch dimension, flatten the rest
            return preprocessed_data.reshape(preprocessed_data.shape[0], -1)
        else:
            return preprocessed_data.flatten()
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """
        Transform data back to image format.
        
        Args:
            data: Transformed data
            
        Returns:
            Data in image format
        """
        # If data is flattened, reshape it back to image format
        if len(data.shape) == 2:
            # Batch of flattened images
            if self.target_size is not None:
                height, width = self.target_size
                channels = 1 if self.grayscale else 3
                
                if self.data_format == 'channels_last':
                    return data.reshape(data.shape[0], height, width, channels)
                else:
                    return data.reshape(data.shape[0], channels, height, width)
        
        # If data is already in image format, return as is
        return data
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for image data.
        
        For images, feature names are pixel coordinates.
        
        Returns:
            List of feature names
        """
        if self.target_size is not None:
            height, width = self.target_size
            channels = 1 if self.grayscale else 3
            
            feature_names = []
            for h in range(height):
                for w in range(width):
                    if channels == 1:
                        feature_names.append(f"pixel_{h}_{w}")
                    else:
                        for c in range(channels):
                            feature_names.append(f"pixel_{h}_{w}_{c}")
            
            return feature_names
        
        return []