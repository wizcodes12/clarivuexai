"""
Scikit-learn model wrappers for ClarivueXAI.

This module provides wrappers for scikit-learn models to make them
compatible with the ClarivueXAI framework.
"""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseModel
from clarivuexai.core.registry import register_model
from clarivuexai.core.utils import convert_to_numpy, get_feature_names


@register_model('sklearn')
class SklearnModel(BaseModel):
    """
    Wrapper for scikit-learn models.
    
    This class wraps scikit-learn models to make them compatible with
    the ClarivueXAI framework.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize a SklearnModel.
        
        Args:
            model: Scikit-learn model object
            feature_names: Optional list of feature names
        """
        try:
            import sklearn
        except ImportError:
            raise ImportError("scikit-learn is required to use SklearnModel. "
                             "Install it with 'pip install scikit-learn'.")
            
        # Check if the model is a scikit-learn model
        if not hasattr(model, 'fit') or not hasattr(model, 'predict'):
            raise ValueError("Model must be a scikit-learn model with fit and predict methods")
        
        # Determine model type with proper validation
        model_type = self._detect_sklearn_model_type(model)
        
        super().__init__(model, model_type, feature_names)
        self._framework = 'sklearn'
        
        # Check if the model is fitted
        self.is_fitted = self._check_is_fitted(model)
        
        # Store additional metadata
        self.is_classifier = model_type in ['classifier', 'binary_classifier']
        self.classes_ = getattr(model, 'classes_', None) if self.is_classifier else None
        self.n_classes_ = len(self.classes_) if self.classes_ is not None else None
    
    def _detect_sklearn_model_type(self, model):
        """
        Detect the type of scikit-learn model based on its attributes.
        
        Args:
            model: scikit-learn model
            
        Returns:
            String indicating model type ('classifier', 'regressor', or 'unknown')
        """
        # Check for _estimator_type attribute (most reliable)
        if hasattr(model, '_estimator_type'):
            if model._estimator_type == 'classifier':
                return 'classifier'
            elif model._estimator_type == 'regressor':
                return 'regressor'
        
        # Try to infer from model class name
        model_name = model.__class__.__name__.lower()
        if any(name in model_name for name in ['classifier', 'sgdc', 'logistic', 'svc', 'nbc', 'knn']):
            return 'classifier'
        elif any(name in model_name for name in ['regressor', 'svr', 'gbr', 'forest', 'ridge', 'lasso']):
            return 'regressor'
            
        # Check for specific methods or attributes
        if hasattr(model, 'predict_proba') or hasattr(model, 'classes_'):
            return 'classifier'
            
        # Default case
        return 'unknown'
    
    def _check_is_fitted(self, model):
        """
        Check if a scikit-learn model is fitted.
        
        Args:
            model: scikit-learn model
            
        Returns:
            Boolean indicating if the model is fitted
        """
        try:
            from sklearn.utils.validation import check_is_fitted as sklearn_check
            try:
                sklearn_check(model)
                return True
            except Exception:
                return False
        except ImportError:
            # Fallback check if sklearn.utils.validation is not available
            try:
                # Check for common attributes of fitted models
                if hasattr(model, 'coef_') or hasattr(model, 'feature_importances_') or \
                   hasattr(model, 'n_features_in_') or hasattr(model, 'classes_'):
                    return True
                
                # Try getting a small prediction to check
                if hasattr(model, 'n_features_in_'):
                    try:
                        dummy_input = np.zeros((1, model.n_features_in_))
                        model.predict(dummy_input)
                        return True
                    except:
                        pass
                return False
            except:
                return False
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the wrapped model.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        try:
            return self.model.predict(X_array)
        except Exception as e:
            raise RuntimeError(f"Error making predictions with scikit-learn model: {str(e)}")
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Input data
            
        Returns:
            Probability estimates
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted")
        
        if not self.is_classifier:
            raise ValueError("predict_proba is only available for classifiers")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not have predict_proba method")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        try:
            return self.model.predict_proba(X_array)
        except Exception as e:
            raise RuntimeError(f"Error getting probabilities with scikit-learn model: {str(e)}")
    
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
    
    def get_params(self) -> dict:
        """
        Get the parameters of the wrapped model.
        
        Returns:
            Dictionary of model parameters
        """
        if hasattr(self.model, 'get_params'):
            try:
                return self.model.get_params()
            except Exception:
                return {}
        else:
            return {}