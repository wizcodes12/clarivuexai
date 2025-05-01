"""
Data handlers for ClarivueXAI.

This module contains data handlers for different data types
to preprocess and transform data for explanation.
"""

from typing import Dict, Type, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseDataHandler
from clarivuexai.core.registry import registry
from clarivuexai.data.tabular import TabularDataHandler
from clarivuexai.data.text import TextDataHandler
from clarivuexai.data.image import ImageDataHandler
from clarivuexai.data.timeseries import TimeSeriesDataHandler

# Export all data handlers
__all__ = [
    'TabularDataHandler',
    'TextDataHandler',
    'ImageDataHandler',
    'TimeSeriesDataHandler',
    'get_data_handler',
    'infer_data_type',
    'create_data_handler'
]

def get_data_handler(data_type: str) -> Type[BaseDataHandler]:
    """
    Get a data handler by type.
    
    Args:
        data_type: Type of data handler to get
        
    Returns:
        Data handler class
        
    Raises:
        ValueError: If data handler is not found
    """
    try:
        return registry.get_data_handler(data_type)
    except KeyError:
        raise ValueError(f"Data handler for '{data_type}' not found.")

def infer_data_type(data: Union[np.ndarray, pd.DataFrame, str, list]) -> str:
    """
    Infer the data type from the data.
    
    Args:
        data: Input data
        
    Returns:
        Inferred data type
        
    Raises:
        ValueError: If data type cannot be inferred
    """
    # Check if data is a string or list of strings (likely text)
    if isinstance(data, str):
        return 'text'
    elif isinstance(data, list) and all(isinstance(x, str) for x in data):
        return 'text'
    
    # Convert to numpy array for further checks
    if isinstance(data, pd.DataFrame):
        array_data = data.values
    else:
        array_data = np.array(data)
    
    # Check dimensions and content to determine type
    if len(array_data.shape) == 1:
        # 1D array, likely tabular with single feature or text
        return 'tabular'
    elif len(array_data.shape) == 2:
        # 2D array, could be tabular or grayscale image
        if array_data.shape[1] > 100:  # Arbitrary threshold for distinguishing images
            return 'image'
        else:
            return 'tabular'
    elif len(array_data.shape) == 3:
        # 3D array, could be RGB image or time series
        if array_data.shape[2] in [1, 3, 4]:  # Common channel counts for images
            return 'image'
        else:
            return 'timeseries'
    elif len(array_data.shape) == 4:
        # 4D array, likely batch of images
        return 'image'
    
    raise ValueError("Could not infer data type from input data.")

def create_data_handler(data_type: str = None, data: Union[np.ndarray, pd.DataFrame, str, list] = None, **kwargs) -> BaseDataHandler:
    """
    Create a data handler for the given data type.
    
    Either data_type or data must be provided. If only data is provided,
    the data type will be inferred.
    
    Args:
        data_type: Type of data handler to create
        data: Input data for type inference
        **kwargs: Additional arguments to pass to the data handler constructor
        
    Returns:
        Initialized data handler
        
    Raises:
        ValueError: If neither data_type nor data is provided
    """
    if data_type is None:
        if data is None:
            raise ValueError("Either data_type or data must be provided.")
        data_type = infer_data_type(data)
    
    handler_class = get_data_handler(data_type)
    return handler_class(**kwargs)