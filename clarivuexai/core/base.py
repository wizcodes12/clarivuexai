"""
Base classes for ClarivueXAI.

This module contains the abstract base classes that define the interfaces
for models, explainers, and data handlers in the ClarivueXAI framework.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """
    Abstract base class for all model wrappers in ClarivueXAI.
    
    This class defines the interface that all model wrappers must implement
    to be compatible with the ClarivueXAI framework.
    """
    
    def __init__(self, model: Any, model_type: str, feature_names: Optional[List[str]] = None):
        """
        Initialize a BaseModel.
        
        Args:
            model: The underlying model object
            model_type: The type of model (e.g., 'classifier', 'regressor')
            feature_names: Optional list of feature names
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names
        
        # Initialize framework attribute
        from clarivuexai.core.utils import detect_framework
        try:
            self._framework = detect_framework(model)
        except ValueError:
            self._framework = "unknown"
        
    @abstractmethod
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions using the wrapped model.
        
        Args:
            X: Input data
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get probability estimates for each class.
        
        Args:
            X: Input data
            
        Returns:
            Probability estimates
        """
        pass
    
    @property
    def framework(self) -> str:
        """
        Get the framework of the underlying model.
        
        Returns:
            Framework name (e.g., 'sklearn', 'tensorflow', 'pytorch')
        """
        return self._framework
    
    def __repr__(self) -> str:
        """String representation of the model wrapper."""
        return f"{self.__class__.__name__}(model_type={self.model_type}, framework={self.framework})"


class BaseExplainer(ABC):
    """
    Abstract base class for all explainers in ClarivueXAI.
    
    This class defines the interface that all explainers must implement
    to be compatible with the ClarivueXAI framework.
    """
    
    def __init__(self, model: BaseModel, name: str):
        """
        Initialize a BaseExplainer.
        
        Args:
            model: A ClarivueXAI model wrapper
            name: Name of the explainer
        """
        self.model = model
        self.name = name
        
    @abstractmethod
    def explain_global(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'ExplanationResult':
        """
        Generate global explanations for the model.
        
        Args:
            X: Input data
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Global explanation results
        """
        pass
    
    @abstractmethod
    def explain_local(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> 'ExplanationResult':
        """
        Generate local explanations for specific instances.
        
        Args:
            X: Input data
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Local explanation results
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the explainer."""
        return f"{self.__class__.__name__}(name={self.name})"


class ExplanationResult:
    """
    Container for explanation results.
    
    This class stores the results of an explanation and provides methods
    for visualizing and analyzing the explanations.
    """
    
    def __init__(
        self, 
        explanation_type: str,
        data: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
        explainer_name: Optional[str] = None
    ):
        """
        Initialize an ExplanationResult.
        
        Args:
            explanation_type: Type of explanation ('global' or 'local')
            data: Dictionary containing explanation data
            feature_names: Optional list of feature names
            explainer_name: Name of the explainer that generated this result
        """
        self.explanation_type = explanation_type
        self.data = data
        self.feature_names = feature_names
        self.explainer_name = explainer_name
        
    def plot(self, plot_type: Optional[str] = None, **kwargs):
        """
        Visualize the explanation results.
        
        Args:
            plot_type: Type of plot to generate
            **kwargs: Additional arguments for the visualization
            
        Returns:
            Visualization object
        """
        try:
            from clarivuexai.visualization.plots import create_plot
            return create_plot(self, plot_type=plot_type, **kwargs)
        except ImportError:
            raise ImportError(
                "Visualization modules not found. Make sure the visualization "
                "package is installed or implement your own visualization method."
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert explanation results to a dictionary.
        
        Returns:
            Dictionary representation of the explanation results
        """
        return {
            'explanation_type': self.explanation_type,
            'data': self.data,
            'feature_names': self.feature_names,
            'explainer_name': self.explainer_name
        }
    
    def __repr__(self) -> str:
        """String representation of the explanation results."""
        return f"ExplanationResult(type={self.explanation_type}, explainer={self.explainer_name})"


class BaseDataHandler(ABC):
    """
    Abstract base class for data handlers in ClarivueXAI.
    
    This class defines the interface for data handlers that preprocess
    and transform data for different modalities.
    """
    
    def __init__(self, data_type: str):
        """
        Initialize a BaseDataHandler.
        
        Args:
            data_type: Type of data (e.g., 'tabular', 'text', 'image')
        """
        self.data_type = data_type
        
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """
        Preprocess the input data.
        
        Args:
            data: Input data
            
        Returns:
            Preprocessed data
        """
        pass
    
    @abstractmethod
    def transform(self, data: Any) -> Any:
        """
        Transform the input data into a format suitable for explanation.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        pass
    
    def __repr__(self) -> str:
        """String representation of the data handler."""
        return f"{self.__class__.__name__}(data_type={self.data_type})"