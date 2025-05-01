"""
Time series data handler for ClarivueXAI.

This module provides a data handler for time series data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseDataHandler
from clarivuexai.core.registry import register_data_handler


@register_data_handler('timeseries')
class TimeSeriesDataHandler(BaseDataHandler):
    """
    Data handler for time series data.
    
    This class provides methods for preprocessing and transforming
    time series data for explanation.
    """
    
    def __init__(
        self, 
        sequence_length: int = 10,
        stride: int = 1,
        normalize: bool = True,
        features: Optional[List[str]] = None
    ):
        """
        Initialize a TimeSeriesDataHandler.
        
        Args:
            sequence_length: Length of sequences to create
            stride: Stride for sequence creation
            normalize: Whether to normalize the data
            features: List of feature names
        """
        super().__init__('timeseries')
        self.sequence_length = sequence_length
        self.stride = stride
        self.normalize = normalize
        self.features = features
        
        # Initialize scalers
        self._scaler = None
        self._num_features = None
    
    def preprocess(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Preprocess the input time series data.
        
        Args:
            data: Input time series data
            
        Returns:
            Preprocessed time series data
        """
        # Convert to DataFrame if necessary
        if isinstance(data, np.ndarray):
            if self.features is not None:
                data = pd.DataFrame(data, columns=self.features)
            else:
                data = pd.DataFrame(data)
        
        # Handle missing values
        data = data.fillna(method='ffill').fillna(method='bfill')
        
        # Store num_features for feature_names generation
        self._num_features = data.shape[1]
        
        # Normalize data if specified
        if self.normalize:
            try:
                from sklearn.preprocessing import StandardScaler
                if self._scaler is None:
                    self._scaler = StandardScaler()
                    data_values = self._scaler.fit_transform(data.values)
                else:
                    data_values = self._scaler.transform(data.values)
                
                data = pd.DataFrame(data_values, columns=data.columns, index=data.index)
            except ImportError:
                # If scikit-learn is not available, just normalize manually
                for column in data.columns:
                    mean = data[column].mean()
                    std = data[column].std()
                    if std > 0:
                        data[column] = (data[column] - mean) / std
        
        return data
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Transform the input time series data into a format suitable for explanation.
        
        Args:
            data: Input time series data
            
        Returns:
            Transformed time series data
        """
        # Preprocess the data
        data = self.preprocess(data)
        
        # Convert to numpy array
        if isinstance(data, pd.DataFrame):
            values = data.values
        else:
            values = data
        
        # Create sequences
        sequences = []
        for i in range(0, len(values) - self.sequence_length + 1, self.stride):
            sequences.append(values[i:i+self.sequence_length])
        
        return np.array(sequences)
    
    def inverse_transform(self, data: np.ndarray) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform data back to its original format.
        
        Args:
            data: Transformed data
            
        Returns:
            Data in original format
        """
        # If data is in sequence format, take the last step of each sequence
        if len(data.shape) == 3:
            data = data[:, -1, :]
        
        # Inverse normalize if necessary
        if self.normalize and self._scaler is not None:
            data = self._scaler.inverse_transform(data)
        
        # Convert to DataFrame if features are available
        if self.features is not None:
            data = pd.DataFrame(data, columns=self.features)
        
        return data
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names for time series data.
        
        Returns:
            List of feature names
        """
        if self.features is not None:
            # For each time step, create a feature name
            feature_names = []
            for t in range(self.sequence_length):
                for feature in self.features:
                    feature_names.append(f"{feature}_t-{self.sequence_length - t}")
            return feature_names
        elif self._num_features is not None:
            # Generate generic feature names
            return [f"feature_{i}_t-{t}" for t in range(self.sequence_length) 
                   for i in range(self._num_features)]
        else:
            return []