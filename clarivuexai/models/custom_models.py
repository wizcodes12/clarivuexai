"""
Custom model wrappers for ClarivueXAI.

This module provides wrappers for custom models to make them
compatible with the ClarivueXAI framework.
"""

from typing import Any, Callable, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseModel
from clarivuexai.core.registry import register_model
from clarivuexai.core.utils import convert_to_numpy


@register_model('custom')
class CustomModel(BaseModel):
    """
    Wrapper for custom models.
    
    This class wraps custom model implementations to make them compatible with
    the ClarivueXAI framework.
    """
    
    def __init__(
        self, 
        model: Any, 
        predict_fn: Optional[Callable] = None,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'classifier'
    ):
        """
        Initialize a CustomModel.
        
        Args:
            model: Custom model object
            predict_fn: Function to use for predictions (if None, uses model.predict)
            feature_names: Optional list of feature names
            model_type: Type of model ('classifier' or 'regressor')
        """
        super().__init__(model, model_type, feature_names)
        self._framework = 'custom'
        
        # Set prediction function
        self._predict_fn = predict_fn if predict_fn is not None else getattr(model, 'predict', None)
        if self._predict_fn is None:
            raise ValueError("Model must have a predict method or a predict_fn must be provided")
        
        # Store additional metadata
        self.is_classifier = model_type == 'classifier'
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the wrapped model.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        # Make predictions
        predictions = self._predict_fn(X_array)
        
        # Ensure the output is a numpy array
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        
        return predictions
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Input data
            
        Returns:
            Probability estimates
        """
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classifiers")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not have a predict_proba method")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        # Get probabilities
        probabilities = self.model.predict_proba(X_array)
        
        # Ensure the output is a numpy array
        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)
        
        return probabilities
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances from the model if available.
        
        Returns:
            Feature importances or None if not available
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            if self.model.coef_.ndim == 1:
                return np.abs(self.model.coef_)
            else:
                # For multi-class models, return the mean absolute coefficient across classes
                return np.mean(np.abs(self.model.coef_), axis=0)
        else:
            return None


@register_model('enhanced_custom')
class EnhancedCustomModel(BaseModel):
    """
    Enhanced wrapper for custom models.
    
    This class wraps custom models with improved flexibility and validation.
    """
    
    def __init__(
        self, 
        model: Any, 
        predict_fn: Optional[Callable] = None,
        predict_proba_fn: Optional[Callable] = None,
        feature_importance_fn: Optional[Callable] = None,
        gradient_fn: Optional[Callable] = None,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'classifier',
        additional_metadata: Optional[dict] = None
    ):
        """
        Initialize an EnhancedCustomModel.
        
        Args:
            model: Custom model object
            predict_fn: Function to use for predictions (if None, uses model.predict)
            predict_proba_fn: Function to use for probability predictions (if None, uses model.predict_proba)
            feature_importance_fn: Function to get feature importances (if None, tries model attributes)
            gradient_fn: Function to compute gradients (if None, not available)
            feature_names: Optional list of feature names
            model_type: Type of model ('classifier' or 'regressor')
            additional_metadata: Additional model metadata as a dictionary
        """
        super().__init__(model, model_type, feature_names)
        self._framework = 'custom'
        
        # Set prediction functions with validation
        self._predict_fn = self._validate_predict_fn(predict_fn, model)
        self._predict_proba_fn = predict_proba_fn if predict_proba_fn is not None else getattr(model, 'predict_proba', None)
        self._feature_importance_fn = feature_importance_fn
        self._gradient_fn = gradient_fn
        
        # Store additional metadata
        self.is_classifier = model_type == 'classifier'
        self.additional_metadata = additional_metadata or {}
        
    def _validate_predict_fn(self, predict_fn, model):
        """Validate and return a prediction function"""
        if predict_fn is not None:
            return predict_fn
        
        if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
            return model.predict
        
        raise ValueError("Model must have a predict method or a predict_fn must be provided")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the wrapped model with error handling.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        try:
            # Make predictions
            predictions = self._predict_fn(X_array)
            
            # Ensure the output is a numpy array
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)
            
            return predictions
        except Exception as e:
            raise RuntimeError(f"Error in custom predict function: {str(e)}")
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get probability estimates for each class with error handling.
        
        Args:
            X: Input data
            
        Returns:
            Probability estimates
        """
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classifiers")
        
        if self._predict_proba_fn is None:
            raise ValueError("Model does not have a predict_proba method and no predict_proba_fn was provided")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        try:
            # Get probabilities
            probabilities = self._predict_proba_fn(X_array)
            
            # Ensure the output is a numpy array
            if not isinstance(probabilities, np.ndarray):
                probabilities = np.array(probabilities)
            
            # Validate shape for classifiers
            if len(probabilities.shape) < 2 or probabilities.shape[1] < 2:
                raise ValueError(f"predict_proba should return a 2D array with shape (n_samples, n_classes), got {probabilities.shape}")
            
            return probabilities
        except Exception as e:
            if "got {probabilities.shape}" in str(e):
                # Re-raise our validation error
                raise
            raise RuntimeError(f"Error in custom predict_proba function: {str(e)}")
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances from the model if available.
        
        Returns:
            Feature importances or None if not available
        """
        if self._feature_importance_fn is not None:
            try:
                return self._feature_importance_fn(self.model)
            except Exception as e:
                print(f"Warning: Error getting feature importances with custom function: {str(e)}")
        
        # Try standard attributes
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            if self.model.coef_.ndim == 1:
                return np.abs(self.model.coef_)
            else:
                # For multi-class models, return the mean absolute coefficient across classes
                return np.mean(np.abs(self.model.coef_), axis=0)
        else:
            return None
    
    def get_gradients(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """
        Compute gradients of the model output with respect to the input.
        
        Args:
            X: Input data
            y: Target labels (for classification, used to select the class)
            
        Returns:
            Gradients or None if not available
        """
        if self._gradient_fn is None:
            return None
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        try:
            return self._gradient_fn(self.model, X_array, y)
        except Exception as e:
            print(f"Warning: Error computing gradients with custom function: {str(e)}")
            return None