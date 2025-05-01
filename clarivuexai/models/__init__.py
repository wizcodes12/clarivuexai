"""
Model wrappers for ClarivueXAI.
This module contains wrappers for different types of models
to make them compatible with the ClarivueXAI framework.
"""
from typing import Any, List, Optional, Union

# Import model wrappers
from clarivuexai.models.sklearn_models import SklearnModel
from clarivuexai.models.tensorflow_models import TensorflowModel
from clarivuexai.models.pytorch_models import PytorchModel
from clarivuexai.models.custom_models import CustomModel, EnhancedCustomModel

# Simplified imports for end users
__all__ = ['SklearnModel', 'TensorflowModel', 'PytorchModel', 'CustomModel', 
           'EnhancedCustomModel', 'detect_framework', 'wrap_model']

def detect_framework(model: Any) -> str:
    """
    Detect the framework of the given model.
    
    Args:
        model: Model object
        
    Returns:
        String indicating the framework ('sklearn', 'tensorflow', 'pytorch', 'custom', or 'unknown')
    """
    import importlib.util
    
    # Check if required packages are available
    sklearn_available = importlib.util.find_spec('sklearn') is not None
    tf_available = importlib.util.find_spec('tensorflow') is not None
    torch_available = importlib.util.find_spec('torch') is not None
    
    # Try to detect the framework from the model's class module
    module_name = model.__class__.__module__
    
    # Check for scikit-learn models
    if sklearn_available and any(name in module_name for name in ['sklearn', 'skl']):
        # Additional check for common scikit-learn model attributes
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            return 'sklearn'
    
    # Check for TensorFlow models
    if tf_available:
        try:
            import tensorflow as tf
            if isinstance(model, tf.keras.Model) or module_name.startswith('tensorflow') or module_name.startswith('keras'):
                return 'tensorflow'
        except (ImportError, AttributeError):
            pass
    
    # Check for PyTorch models
    if torch_available:
        try:
            import torch
            if isinstance(model, torch.nn.Module) or module_name.startswith('torch'):
                return 'pytorch'
        except (ImportError, AttributeError):
            pass
    
    # Additional checks for scikit-learn models (in case module name check failed)
    if sklearn_available and hasattr(model, 'fit') and hasattr(model, 'predict') and hasattr(model, 'get_params'):
        try:
            # Check if get_params follows scikit-learn's convention
            params = model.get_params()
            if isinstance(params, dict):
                return 'sklearn'
        except:
            pass
    
    # If still unknown, try to infer from available methods
    if hasattr(model, 'predict') and callable(getattr(model, 'predict')):
        if hasattr(model, 'fit') and callable(getattr(model, 'fit')):
            # Most likely a scikit-learn-compatible model
            return 'sklearn'
        else:
            # Generic model with prediction capability
            return 'custom'
    
    # If we couldn't identify the framework
    return 'unknown'


def wrap_model(model, feature_names=None, **kwargs):
    """
    Automatically detect the model type and wrap it with the appropriate wrapper.
    
    Args:
        model: Model object to wrap
        feature_names: Optional list of feature names
        **kwargs: Additional arguments to pass to the wrapper
        
    Returns:
        Wrapped model instance
        
    Raises:
        ValueError: If model type cannot be detected
    """
    # Using the detect_framework function defined in this module
    framework = detect_framework(model)
    
    if framework == 'sklearn':
        return SklearnModel(model, feature_names=feature_names)
    elif framework == 'tensorflow':
        return TensorflowModel(model, feature_names=feature_names)
    elif framework == 'pytorch':
        return PytorchModel(model, feature_names=feature_names, **kwargs)
    elif framework == 'custom':
        # Use enhanced custom model if gradient_fn is provided
        if 'gradient_fn' in kwargs or 'predict_proba_fn' in kwargs or 'feature_importance_fn' in kwargs:
            return EnhancedCustomModel(model, feature_names=feature_names, **kwargs)
        else:
            return CustomModel(model, feature_names=feature_names, **kwargs)
    else:
        # If framework can't be detected, try using the custom model wrapper
        try:
            return CustomModel(model, feature_names=feature_names, **kwargs)
        except Exception as e:
            raise ValueError(f"Could not detect model type. Please use a specific model wrapper. Error: {str(e)}")