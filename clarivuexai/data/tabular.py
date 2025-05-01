"""
Tabular data handler for ClarivueXAI.

This module provides a data handler for tabular data.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseDataHandler
from clarivuexai.core.registry import register_data_handler


@register_data_handler('tabular')
class TabularDataHandler(BaseDataHandler):
    """
    Data handler for tabular data.
    
    This class provides methods for preprocessing and transforming
    tabular data for explanation.
    """
    
    def __init__(
        self, 
        categorical_features: Optional[List[str]] = None,
        numerical_features: Optional[List[str]] = None,
        feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
    ):
        """
        Initialize a TabularDataHandler.
        
        Args:
            categorical_features: List of categorical feature names
            numerical_features: List of numerical feature names
            feature_ranges: Dictionary mapping feature names to (min, max) tuples
        """
        super().__init__('tabular')
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.feature_ranges = feature_ranges or {}
        
        # Initialize scalers and encoders
        self._scaler = None
        self._encoder = None
        self._feature_names = []
        
    def preprocess(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Preprocess the input data.
        
        Args:
            data: Input data
            
        Returns:
            Preprocessed data
        """
        # Convert to DataFrame if necessary
        if isinstance(data, np.ndarray):
            if self.numerical_features or self.categorical_features:
                # Use provided feature names
                feature_names = self.numerical_features + self.categorical_features
                data = pd.DataFrame(data, columns=feature_names)
            else:
                # Generate feature names
                feature_names = [f"feature_{i}" for i in range(data.shape[1])]
                data = pd.DataFrame(data, columns=feature_names)
        
        # Infer numerical and categorical features if not provided
        if not self.numerical_features and not self.categorical_features:
            self.numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
            self.categorical_features = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        # Handle missing values
        for feature in self.numerical_features:
            if feature in data.columns and data[feature].isnull().any():
                data[feature] = data[feature].fillna(data[feature].mean())
        
        for feature in self.categorical_features:
            if feature in data.columns and data[feature].isnull().any():
                data[feature] = data[feature].fillna(data[feature].mode()[0])
        
        return data
    
    def transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform the input data into a format suitable for explanation.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        # Preprocess the data
        data = self.preprocess(data)
        
        # Scale numerical features
        if self.numerical_features:
            try:
                from sklearn.preprocessing import StandardScaler
                if self._scaler is None:
                    self._scaler = StandardScaler()
                    data[self.numerical_features] = self._scaler.fit_transform(data[self.numerical_features])
                else:
                    data[self.numerical_features] = self._scaler.transform(data[self.numerical_features])
            except ImportError:
                # If scikit-learn is not available, just normalize
                for feature in self.numerical_features:
                    if feature in data.columns:
                        min_val = data[feature].min()
                        max_val = data[feature].max()
                        if max_val > min_val:
                            data[feature] = (data[feature] - min_val) / (max_val - min_val)
        
        # Encode categorical features
        if self.categorical_features:
            try:
                from sklearn.preprocessing import OneHotEncoder
                if self._encoder is None:
                    self._encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = self._encoder.fit_transform(data[self.categorical_features])
                else:
                    encoded = self._encoder.transform(data[self.categorical_features])
                
                # Create DataFrame with encoded features
                encoded_df = pd.DataFrame(
                    encoded,
                    columns=self._encoder.get_feature_names_out(self.categorical_features),
                    index=data.index
                )
                
                # Drop original categorical features and add encoded ones
                data = data.drop(self.categorical_features, axis=1)
                data = pd.concat([data, encoded_df], axis=1)
            except ImportError:
                # If scikit-learn is not available, just use pandas get_dummies
                for feature in self.categorical_features:
                    if feature in data.columns:
                        dummies = pd.get_dummies(data[feature], prefix=feature)
                        data = pd.concat([data.drop(feature, axis=1), dummies], axis=1)
        
        # Store feature names for later use
        if isinstance(data, pd.DataFrame):
            self._feature_names = list(data.columns)
        
        return data
    
    def inverse_transform(self, data: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
        """
        Transform data back to its original format.
        
        Args:
            data: Transformed data
            
        Returns:
            Data in original format
        """
        # Convert to DataFrame if necessary
        if isinstance(data, np.ndarray):
            if self._feature_names:
                data = pd.DataFrame(data, columns=self._feature_names)
            else:
                data = pd.DataFrame(data)
        
        # Inverse transform numerical features
        if self.numerical_features and self._scaler is not None:
            try:
                # Get the numerical columns that are in the data
                num_cols = [col for col in self.numerical_features if col in data.columns]
                if num_cols:
                    data[num_cols] = self._scaler.inverse_transform(data[num_cols])
            except:
                pass
        
        # Inverse transform categorical features
        if self.categorical_features and self._encoder is not None:
            try:
                # Get the encoded column names
                encoded_cols = [col for col in data.columns if any(col.startswith(f"{cat}_") for cat in self.categorical_features)]
                
                if encoded_cols:
                    # Extract the encoded features
                    encoded_data = data[encoded_cols].values
                    
                    # Inverse transform
                    original_cats = self._encoder.inverse_transform(encoded_data)
                    
                    # Create DataFrame with original categorical features
                    original_df = pd.DataFrame(
                        original_cats,
                        columns=self.categorical_features,
                        index=data.index
                    )
                    
                    # Drop encoded features and add original ones
                    data = data.drop(encoded_cols, axis=1)
                    data = pd.concat([data, original_df], axis=1)
            except:
                pass
        
        return data
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names from the data.
        
        Returns:
            List of feature names
        """
        if self._feature_names:
            return self._feature_names
        elif self.numerical_features or self.categorical_features:
            return self.numerical_features + self.categorical_features
        else:
            return []