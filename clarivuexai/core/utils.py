"""
Utility functions for ClarivueXAI.

This module provides utility functions for data handling, model inspection,
and other common tasks in the ClarivueXAI framework.
"""

import inspect
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


def get_feature_names(X):
    """
    Extract feature names from different data formats.
    
    Args:
        X: Input data (numpy array, pandas DataFrame, list, etc.)
        
    Returns:
        List of feature names
    """
    import numpy as np
    import pandas as pd
    
    if isinstance(X, pd.DataFrame):
        return X.columns
    elif isinstance(X, np.ndarray):
        return [f"feature_{i}" for i in range(X.shape[1])]
    elif isinstance(X, list):
        # Check if it's a list of strings (text data)
        if all(isinstance(item, str) for item in X):
            # For text data, we don't have explicit feature names
            return ["text_content"]
        else:
            # For other types of lists, generate generic feature names
            try:
                # Try to determine the number of features
                if isinstance(X[0], (list, np.ndarray)):
                    return [f"feature_{i}" for i in range(len(X[0]))]
                else:
                    return ["feature_0"]  # Single feature case
            except (IndexError, TypeError):
                return ["feature_0"]  # Default for empty or problematic lists
    else:
        raise TypeError(f"Unsupported data type: {type(X)}")

def check_model_compatibility(model: Any, framework: str) -> bool:
    """
    Check if a model is compatible with a given framework.
    
    Args:
        model: Model object to check
        framework: Framework name to check compatibility with
        
    Returns:
        True if the model is compatible with the framework, False otherwise
    """
    if framework == 'sklearn':
        return hasattr(model, 'fit') and hasattr(model, 'predict')
    elif framework == 'tensorflow':
        try:
            import tensorflow as tf
            return isinstance(model, tf.keras.Model)
        except ImportError:
            return False
    elif framework == 'pytorch':
        try:
            import torch
            return isinstance(model, torch.nn.Module)
        except ImportError:
            return False
    else:
        return False


def detect_framework(model: Any) -> str:
    """
    Detect the framework of a model.
    
    Args:
        model: Model object to detect the framework of
        
    Returns:
        Framework name
        
    Raises:
        ValueError: If the framework cannot be detected
    """
    # Check for scikit-learn models
    if hasattr(model, 'fit') and hasattr(model, 'predict'):
        module = inspect.getmodule(model.__class__)
        if module and module.__name__.startswith('sklearn'):
            return 'sklearn'
    
    # Check for TensorFlow models
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return 'tensorflow'
    except ImportError:
        pass
    
    # Check for PyTorch models
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            return 'pytorch'
    except ImportError:
        pass
    
    # Check for custom models
    if hasattr(model, 'predict') and callable(model.predict):
        return 'custom'
    
    raise ValueError(f"Could not detect framework for model: {model}")


def convert_to_numpy(X: Any) -> np.ndarray:
    """
    Convert input data to a numpy array.
    
    Args:
        X: Input data
        
    Returns:
        Numpy array
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        return X.values
    elif isinstance(X, np.ndarray):
        return X
    elif isinstance(X, list):
        return np.array(X)
    else:
        try:
            # Try to convert to numpy array
            return np.array(X)
        except:
            raise TypeError(f"Cannot convert {type(X)} to numpy array")


def is_classifier(model: Any) -> bool:
    """
    Check if a model is a classifier.
    
    Args:
        model: Model object to check
        
    Returns:
        True if the model is a classifier, False otherwise
    """
    # Check for explicit classifier attribute
    if hasattr(model, '_estimator_type'):
        return model._estimator_type == 'classifier'
    
    # Check for scikit-learn classifier naming conventions
    if hasattr(model, '__class__') and hasattr(model.__class__, '__name__'):
        class_name = model.__class__.__name__.lower()
        if 'classifier' in class_name:
            return True
    
    # Check for predict_proba method (common in classifiers)
    if hasattr(model, 'predict_proba') and callable(model.predict_proba):
        return True
    
    # Check for classes_ attribute (common in sklearn classifiers)
    if hasattr(model, 'classes_'):
        return True
    
    # Check for TensorFlow models
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            # Check if the last layer has a softmax or sigmoid activation
            last_layer = model.layers[-1]
            if hasattr(last_layer, 'activation'):
                activation_name = last_layer.activation.__name__ if hasattr(last_layer.activation, '__name__') else str(last_layer.activation)
                return activation_name in ['softmax', 'sigmoid']
            
            # Check output shape for multi-class or binary classification
            if hasattr(model, 'output_shape'):
                # If the output dimension is 1 or >1, it's likely a classifier
                output_dim = model.output_shape[-1] if isinstance(model.output_shape, tuple) else 1
                return output_dim == 1 or output_dim > 1
    except (ImportError, AttributeError, IndexError):
        pass
    
    # Check for PyTorch models
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            # Check if the model has common classifier components
            for module in model.modules():
                if isinstance(module, torch.nn.Softmax) or isinstance(module, torch.nn.Sigmoid):
                    return True
                    
            # Check if the model has a forward method that returns logits
            if hasattr(model, 'forward'):
                sig = inspect.signature(model.forward)
                return 'logits' in sig.return_annotation.__annotations__ if hasattr(sig, 'return_annotation') else False
    except (ImportError, AttributeError):
        pass
    
    # Default to False if we can't determine
    return False


def get_model_metadata(model: Any) -> Dict[str, Any]:
    """
    Extract metadata from a model.
    
    Args:
        model: Model object to extract metadata from
        
    Returns:
        Dictionary of metadata
    """
    metadata = {
        'framework': detect_framework(model),
        'is_classifier': is_classifier(model),
    }
    
    # Add framework-specific metadata
    if metadata['framework'] == 'sklearn':
        metadata['model_type'] = getattr(model, '_estimator_type', 'unknown')
        metadata['params'] = getattr(model, 'get_params', lambda: {})()
    elif metadata['framework'] == 'tensorflow':
        try:
            metadata['layers'] = [layer.name for layer in model.layers]
            metadata['trainable_params'] = model.count_params()
        except:
            pass
    elif metadata['framework'] == 'pytorch':
        try:
            metadata['modules'] = [name for name, _ in model.named_modules()]
            metadata['trainable_params'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        except:
            pass
    
    return metadata


def check_array(X: Any, allow_nd: bool = False) -> np.ndarray:
    """
    Input validation for standard estimators.
    
    Args:
        X: Input data
        allow_nd: Whether to allow multi-dimensional arrays
        
    Returns:
        Validated input data
        
    Raises:
        ValueError: If the input data is invalid
    """
    if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
        X_array = X.values
    else:
        X_array = np.asarray(X)
    
    if not allow_nd and X_array.ndim > 2:
        raise ValueError(f"Expected 2D array, got {X_array.ndim}D array")
    
    return X_array


def check_is_fitted(model: Any, attributes: Optional[List[str]] = None) -> bool:
    """
    Check if a model is fitted.
    
    Args:
        model: Model object to check
        attributes: List of attributes to check for
        
    Returns:
        True if the model is fitted, False otherwise
    """
    if attributes is None:
        attributes = ['coef_', 'estimator_', 'feature_importances_', 'n_iter_']
    
    # Check for scikit-learn models
    if hasattr(model, 'check_is_fitted'):
        try:
            model.check_is_fitted()
            return True
        except:
            return False
    
    # Check for custom attributes
    for attr in attributes:
        if hasattr(model, attr):
            return True
    
    # Check for TensorFlow models
    try:
        import tensorflow as tf
        if isinstance(model, tf.keras.Model):
            return model.built
    except (ImportError, AttributeError):
        pass
    
    # Check for PyTorch models
    try:
        import torch
        if isinstance(model, torch.nn.Module):
            # Check if any parameter requires grad
            return any(p.requires_grad for p in model.parameters())
    except (ImportError, AttributeError):
        pass
    
    return False