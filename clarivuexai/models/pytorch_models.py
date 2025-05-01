"""
PyTorch model wrappers for ClarivueXAI.

This module provides wrappers for PyTorch models to make them
compatible with the ClarivueXAI framework.
"""

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseModel
from clarivuexai.core.registry import register_model
from clarivuexai.core.utils import convert_to_numpy, get_feature_names


@register_model('pytorch')
class PytorchModel(BaseModel):
    """
    Wrapper for PyTorch models.
    
    This class wraps PyTorch models to make them compatible with
    the ClarivueXAI framework.
    """
    
    def __init__(
        self, 
        model: Any, 
        feature_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        model_type: str = 'classifier'
    ):
        """
        Initialize a PytorchModel.
        
        Args:
            model: PyTorch model object
            feature_names: Optional list of feature names
            device: Device to run the model on ('cpu' or 'cuda')
            model_type: Type of model ('classifier' or 'regressor')
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to use PytorchModel. "
                             "Install it with 'pip install torch'.")
        
        # Check if the model is a PyTorch model
        if not isinstance(model, torch.nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module")
        
        super().__init__(model, model_type, feature_names)
        self._framework = 'pytorch'
        
        # Set device with proper error handling
        self.device = self._safe_set_device(device)
        
        # Move model to device safely
        self.model = self._safe_to_device(model, self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Store additional metadata
        self.is_classifier = model_type == 'classifier'
    
    def _safe_set_device(self, device=None):
        """
        Safely set the device with proper error handling.
        
        Args:
            device: Target device string or None
            
        Returns:
            torch.device object
        """
        try:
            import torch
            
            if device is None:
                # Try to use CUDA if available
                if torch.cuda.is_available():
                    return torch.device('cuda')
                else:
                    return torch.device('cpu')
            else:
                # Try to use the specified device
                try:
                    return torch.device(device)
                except RuntimeError:
                    print(f"Warning: Device '{device}' not available. Falling back to CPU.")
                    return torch.device('cpu')
        except Exception as e:
            print(f"Error setting device: {str(e)}. Using CPU.")
            import torch
            return torch.device('cpu')
    
    def _safe_to_device(self, model, device):
        """
        Safely move a PyTorch model to the specified device.
        
        Args:
            model: PyTorch model
            device: Target device
            
        Returns:
            Model on the target device
        """
        try:
            import torch
            return model.to(device)
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"Warning: Could not move model to {device} due to CUDA error. "
                     f"Using CPU instead. Error: {str(e)}")
                return model.to('cpu')
            raise
        except Exception as e:
            print(f"Unknown error moving model to device: {str(e)}. Using original model.")
            return model
    
    def _safe_tensor_conversion(self, array, dtype=None, requires_grad=False):
        """
        Safely convert NumPy array to PyTorch tensor with proper error handling.
        
        Args:
            array: NumPy array to convert
            dtype: Data type (defaults to float32)
            requires_grad: Whether to track gradients
            
        Returns:
            PyTorch tensor on the correct device
        """
        try:
            import torch
            import numpy as np
        except ImportError:
            raise ImportError("PyTorch and NumPy are required.")
        
        if dtype is None:
            dtype = torch.float32
        
        # Ensure array is actually numpy array
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        
        # Handle different numpy dtypes
        if np.issubdtype(array.dtype, np.floating):
            tensor_dtype = dtype
        elif np.issubdtype(array.dtype, np.integer):
            tensor_dtype = torch.long
        else:
            # Default to float32 for other types
            tensor_dtype = dtype
        
        try:
            return torch.tensor(array, dtype=tensor_dtype, device=self.device, requires_grad=requires_grad)
        except RuntimeError as e:
            if "CUDA" in str(e):
                # Fall back to CPU if CUDA error
                print(f"Warning: CUDA error when creating tensor: {str(e)}. Using CPU.")
                return torch.tensor(array, dtype=tensor_dtype, device='cpu', requires_grad=requires_grad)
            raise
        except ValueError as e:
            # Handle conversion errors
            print(f"Warning: Error converting array to tensor: {str(e)}. Attempting to convert to float32.")
            try:
                array = array.astype(np.float32)
                return torch.tensor(array, dtype=torch.float32, device=self.device, requires_grad=requires_grad)
            except:
                raise ValueError(f"Could not convert array to tensor: {str(e)}")
        
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the wrapped model.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to use predict. "
                             "Install it with 'pip install torch'.")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        # Convert to PyTorch tensor with error handling
        X_tensor = self._safe_tensor_conversion(X_array)
        
        # Make predictions
        try:
            with torch.no_grad():
                predictions = self.model(X_tensor)
            
            # Convert to numpy array
            predictions_np = predictions.cpu().numpy()
            
            # For classifiers, return class indices
            if self.is_classifier:
                return np.argmax(predictions_np, axis=1)
            else:
                return predictions_np
        except Exception as e:
            raise RuntimeError(f"Error making predictions with PyTorch model: {str(e)}")
    
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
        
        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("PyTorch is required to use predict_proba. "
                             "Install it with 'pip install torch'.")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        # Convert to PyTorch tensor with error handling
        X_tensor = self._safe_tensor_conversion(X_array)
        
        try:
            # Make predictions
            with torch.no_grad():
                logits = self.model(X_tensor)
                probabilities = F.softmax(logits, dim=1)
            
            # Convert to numpy array
            return probabilities.cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"Error getting probabilities with PyTorch model: {str(e)}")
    
    def get_feature_importances(self) -> Optional[np.ndarray]:
        """
        Get feature importances from the model if available.
        
        For PyTorch models, feature importances are not directly available.
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
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to use get_gradients. "
                             "Install it with 'pip install torch'.")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        # Convert to PyTorch tensor with gradients enabled
        X_tensor = self._safe_tensor_conversion(X_array, requires_grad=True)
        
        try:
            # Make predictions
            self.model.zero_grad()
            predictions = self.model(X_tensor)
            
            if self.is_classifier and y is not None:
                # For classifiers, compute gradients for the target class
                if len(predictions.shape) == 2:
                    # One-hot encode y if needed
                    if isinstance(y, np.ndarray) and len(y.shape) == 1:
                        y_tensor = torch.tensor(y, dtype=torch.long, device=self.device)
                        loss = predictions.gather(1, y_tensor.unsqueeze(1)).squeeze()
                    else:
                        y_tensor = torch.tensor(y, dtype=torch.float32, device=self.device)
                        loss = torch.sum(predictions * y_tensor, dim=1)
                else:
                    loss = predictions
            else:
                # For regressors, compute gradients for the output
                loss = predictions
            
            # Compute gradients
            if len(loss.shape) == 0:
                loss.backward()
            else:
                loss.sum().backward()
            
            # Get gradients
            if X_tensor.grad is None:
                raise ValueError("No gradients were computed. This might be due to disconnected operations.")
                
            gradients = X_tensor.grad.clone()
            
            # Convert to numpy array
            return gradients.cpu().numpy()
        except Exception as e:
            raise RuntimeError(f"Error computing gradients with PyTorch model: {str(e)}")
    
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
            import torch
        except ImportError:
            raise ImportError("PyTorch is required to use get_layer_outputs. "
                             "Install it with 'pip install torch'.")
        
        # Convert input to the appropriate format
        X_array = convert_to_numpy(X)
        
        # Convert to PyTorch tensor
        X_tensor = self._safe_tensor_conversion(X_array)
        
        try:
            # Create hooks to capture layer outputs
            outputs = {}
            handles = []
            
            def get_hook(name):
                def hook(module, input, output):
                    # Convert output to numpy array
                    if isinstance(output, torch.Tensor):
                        outputs[name] = output.detach().cpu().numpy()
                    elif isinstance(output, tuple) and isinstance(output[0], torch.Tensor):
                        # Handle cases where output is a tuple of tensors
                        outputs[name] = tuple(o.detach().cpu().numpy() if isinstance(o, torch.Tensor) else o 
                                             for o in output)
                    else:
                        outputs[name] = output
                return hook
            
            # Register hooks
            found_layer = False
            if layer_name is not None:
                # Find the layer by name
                for name, module in self.model.named_modules():
                    if name == layer_name:
                        handles.append(module.register_forward_hook(get_hook(name)))
                        found_layer = True
                        break
                
                if not found_layer:
                    available_layers = [name for name, _ in self.model.named_modules() if name]
                    raise ValueError(f"Layer '{layer_name}' not found. Available layers: {available_layers}")
            else:
                # Register hooks for all layers
                for name, module in self.model.named_modules():
                    if name:  # Skip the root module
                        handles.append(module.register_forward_hook(get_hook(name)))
            
            # Forward pass
            with torch.no_grad():
                self.model(X_tensor)
            
            # Remove hooks
            for handle in handles:
                handle.remove()
            
            return outputs
        except Exception as e:
            raise RuntimeError(f"Error getting layer outputs with PyTorch model: {str(e)}")