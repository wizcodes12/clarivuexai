"""
SHAP explainers for ClarivueXAI.

This module provides explainers that use SHAP (SHapley Additive exPlanations)
to explain model predictions.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseExplainer, BaseModel, ExplanationResult
from clarivuexai.core.registry import register_explainer
from clarivuexai.core.utils import get_feature_names


@register_explainer('shap')
class ShapExplainer(BaseExplainer):
    """
    Explainer that uses SHAP values to explain model predictions.
    
    This explainer uses the SHAP library to compute Shapley values
    for explaining model predictions.
    """
    
    def __init__(self, model: BaseModel, background_data: Optional[Union[np.ndarray, pd.DataFrame]] = None):
        """
        Initialize a ShapExplainer.
        
        Args:
            model: A ClarivueXAI model wrapper
            background_data: Background data for SHAP explainer (required for some explainer types)
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is required to use ShapExplainer. "
                             "Install it with 'pip install shap'.")
        
        super().__init__(model, 'shap')
        self.background_data = background_data
        self._explainer = None
        
    def _create_explainer(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Create a SHAP explainer based on the model type.
        
        Args:
            X: Input data
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP is required to use ShapExplainer. "
                             "Install it with 'pip install shap'.")
        
        # Choose the appropriate SHAP explainer based on the model framework
        if self.model.framework == 'sklearn':
            if self.background_data is None:
                # Use a sample of X as background data
                if isinstance(X, pd.DataFrame):
                    self.background_data = X.sample(min(100, len(X))).values
                else:
                    indices = np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)
                    self.background_data = X[indices]
            
            # Try to use TreeExplainer for tree-based models
            try:
                self._explainer = shap.TreeExplainer(self.model.model)
            except:
                # Fall back to KernelExplainer
                self._explainer = shap.KernelExplainer(
                    self.model.predict if not self.model.is_classifier else self.model.predict_proba,
                    self.background_data
                )
        elif self.model.framework in ['tensorflow', 'pytorch']:
            if self.background_data is None:
                raise ValueError("Background data is required for TensorFlow and PyTorch models")
            
            # Use DeepExplainer for deep learning models
            self._explainer = shap.DeepExplainer(self.model.model, self.background_data)
        else:
            # Use KernelExplainer for other models
            if self.background_data is None:
                # Use a sample of X as background data
                if isinstance(X, pd.DataFrame):
                    self.background_data = X.sample(min(100, len(X))).values
                else:
                    indices = np.random.choice(X.shape[0], min(100, X.shape[0]), replace=False)
                    self.background_data = X[indices]
            
            self._explainer = shap.KernelExplainer(
                self.model.predict if not self.model.is_classifier else self.model.predict_proba,
                self.background_data
            )
    
    def explain_global(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> ExplanationResult:
        """
        Generate global explanations for the model using SHAP values.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Returns:
            Global explanation results
        """
        # Create explainer if not already created
        if self._explainer is None:
            self._create_explainer(X)
        
        # Get feature names
        feature_names = kwargs.get('feature_names', None)
        if feature_names is None:
            feature_names = self.model.feature_names
        if feature_names is None:
            feature_names = get_feature_names(X)
        
        # Compute SHAP values
        shap_values = self._explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(shap_values, list):
            # For multi-class models, average across classes
            if len(shap_values) > 1:
                shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_values = shap_values[0]
        
        # Compute global feature importance
        feature_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create explanation data
        explanation_data = {
            'shap_values': shap_values,
            'feature_importance': feature_importance,
            'feature_names': feature_names
        }
        
        return ExplanationResult(
            explanation_type='global',
            data=explanation_data,
            feature_names=feature_names,
            explainer_name=self.name
        )
    
    def explain_local(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> ExplanationResult:
        """
        Generate local explanations for specific instances using SHAP values.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Returns:
            Local explanation results
        """
        # Create explainer if not already created
        if self._explainer is None:
            self._create_explainer(X)
        
        # Get feature names
        feature_names = kwargs.get('feature_names', None)
        if feature_names is None:
            feature_names = self.model.feature_names
        if feature_names is None:
            feature_names = get_feature_names(X)
        
        # Compute SHAP values
        shap_values = self._explainer.shap_values(X)
        
        # Handle different output formats
        if isinstance(shap_values, list) and len(shap_values) > 1:
            # For multi-class models, use the class with the highest probability
            if self.model.is_classifier:
                proba = self.model.predict_proba(X)
                if len(proba.shape) == 2:
                    class_idx = np.argmax(proba, axis=1)
                    local_shap_values = np.array([shap_values[class_idx[i]][i] for i in range(len(X))])
                else:
                    local_shap_values = shap_values[0]
            else:
                local_shap_values = shap_values[0]
        else:
            local_shap_values = shap_values
        
        # Create explanation data
        explanation_data = {
            'shap_values': local_shap_values,
            'feature_names': feature_names
        }
        
        return ExplanationResult(
            explanation_type='local',
            data=explanation_data,
            feature_names=feature_names,
            explainer_name=self.name
        )