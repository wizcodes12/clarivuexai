"""
TensorFlow model wrappers for ClarivueXAI.

This module provides wrappers for TensorFlow models to make them
compatible with the ClarivueXAI framework.
"""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseModel
from clarivuexai.core.registry import register_model
from clarivuexai.core.utils import convert_to_numpy, get_feature_names


@register_model('tensorflow')
class TensorflowModel(BaseModel):
    """
    Wrapper for TensorFlow models.
    
    This class wraps TensorFlow models to make them compatible with
    the ClarivueXAI framework.
    """
    
    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        """
        Initialize a TensorflowModel.
        
        Args:
            model: TensorFlow model object
            feature_names: Optional list of feature names
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required to use TensorflowModel. "
                             "Install it with 'pip install tensorflow'.")
        
        # Check if the model is a TensorFlow model
        if not isinstance(model, tf.keras.Model):
            raise ValueError("Model must be a TensorFlow Keras model")
        
        # Determine model type using improved detection logic
        model_type = self._detect_tf_model_type(model)
        
        super().__init__(model, model_type, feature_names)
        self._framework = 'tensorflow'
        
        # Store additional metadata
        self.is_classifier = model_type in ['classifier', 'binary_classifier']
        self.n_classes_ = model.output_shape[-1] if self.is_classifier else None
        
    def _detect_tf_model_type(self, model):
        """
        Detect the type of TensorFlow model based on its architecture.
        
        Args:
            model: TensorFlow Keras model
            
        Returns:
            String indicating model type ('classifier', 'binary_classifier', or 'regressor')
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required. Install it with 'pip install tensorflow'.")
        
        # Check the shape of the output layer
        output_shape = model.output_shape
        
        # Get the last layer
        last_layer = model.layers[-1]
        
        # Check if the model has a specific loss that indicates its type
        model_loss = getattr(model, 'loss', None)
        if isinstance(model_loss, str):
            if 'categorical' in model_loss or 'sparse' in model_loss:
                return 'classifier'
            elif 'binary' in model_loss:
                return 'binary_classifier'
            elif 'mse' in model_loss or 'mae' in model_loss:
                return 'regressor'
        
        # Check based on activation function
        if hasattr(last_layer, 'activation'):
            activation_name = last_layer.activation.__name__ if callable(last_layer.activation) else str(last_layer.activation)
            
            if 'softmax' in activation_name:
                return 'classifier'
            elif 'sigmoid' in activation_name:
                if isinstance(output_shape, tuple) and output_shape[-1] == 1:
                    return 'binary_classifier'
                else:
                    # Multi-label classification
                    return 'classifier'
            elif 'linear' in activation_name or activation_name == '<function linear at' or 'relu' in activation_name:
                return 'regressor'
        
        # Check based on output dimension
        if isinstance(output_shape, tuple):
            if len(output_shape) >= 2 and output_shape[-1] > 1:
                # Multiple output units often indicate classification
                return 'classifier'
            elif len(output_shape) >= 2 and output_shape[-1] == 1:
                # Single output unit could be binary classification or regression
                # Default to regression as the safer option
                return 'regressor'
        
        # Default case
        return 'unknown'
        
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
        try:
            predictions = self.model.predict(X_array)
        except Exception as e:
            raise RuntimeError(f"Error making predictions with TensorFlow model: {str(e)}")
        
        # For classifiers, return class indices
        if self.is_classifier:
            return np.argmax(predictions, axis=1)
        else:
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
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        # Get probabilities
        try:
            return self.model.predict(X_array)
        except Exception as e:
            raise RuntimeError(f"Error getting probabilities with TensorFlow model: {str(e)}")
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances from the model if available.
        
        For TensorFlow models, feature importances are not directly available.
        This method returns None, and users should use explainers like
        integrated gradients or SHAP to get feature importances.
        
        Returns:
            None (feature importances not directly available)
        """
        return None
    
    def get_gradients(self, X: Union[np.ndarray, pd.DataFrame], y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute gradients of the model output with respect to the input.
        
        Args:
            X: Input data
            y: Target labels (for classification, used to select the class)
            
        Returns:
            Gradients
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required to use get_gradients. "
                             "Install it with 'pip install tensorflow'.")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        try:
            # Convert to TensorFlow tensor
            X_tensor = tf.convert_to_tensor(X_array, dtype=tf.float32)
            
            # Compute gradients
            with tf.GradientTape() as tape:
                tape.watch(X_tensor)
                predictions = self.model(X_tensor)
                
                if self.is_classifier and y is not None:
                    # For classifiers, compute gradients for the target class
                    if len(predictions.shape) == 2:
                        # One-hot encode y if needed
                        if len(y.shape) == 1:
                            y_one_hot = tf.one_hot(y, depth=predictions.shape[1])
                        else:
                            y_one_hot = y
                        
                        # Compute loss
                        loss = tf.reduce_sum(predictions * y_one_hot, axis=1)
                    else:
                        loss = predictions
                else:
                    # For regressors, compute gradients for the output
                    loss = predictions
            
            # Get gradients
            gradients = tape.gradient(loss, X_tensor)
            
            return gradients.numpy()
        except Exception as e:
            raise RuntimeError(f"Error computing gradients with TensorFlow model: {str(e)}")
    
    def get_layer_outputs(self, X: Union[np.ndarray, pd.DataFrame], layer_name: Optional[str] = None) -> dict:
        """
        Get the outputs of a specific layer or all layers for the given input.
        
        Args:
            X: Input data
            layer_name: Name of the layer to get outputs for (None for all layers)
            
        Returns:
            Dictionary mapping layer names to their outputs
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow is required to use get_layer_outputs. "
                             "Install it with 'pip install tensorflow'.")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        try:
            # Create a new model that outputs the layer activations
            if layer_name is not None:
                # Get a specific layer
                layer = self.model.get_layer(layer_name)
                intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
                return {layer_name: intermediate_model.predict(X_array)}
            else:
                # Get all layers
                outputs = {}
                for layer in self.model.layers:
                    try:
                        intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
                        outputs[layer.name] = intermediate_model.predict(X_array)
                    except:
                        # Skip layers that can't be used as outputs
                        continue
                return outputs
        except Exception as e:
            raise RuntimeError(f"Error getting layer outputs with TensorFlow model: {str(e)}")