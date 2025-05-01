"""
Feature importance explainer for ClarivueXAI.

This module provides an explainer that uses feature importances
from the model to explain its behavior.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseExplainer, BaseModel, ExplanationResult
from clarivuexai.core.registry import register_explainer
from clarivuexai.core.utils import get_feature_names


@register_explainer('feature_importance')
class FeatureImportanceExplainer(BaseExplainer):
    """
    Explainer that uses feature importances from the model.
    
    This explainer extracts feature importances directly from the model
    if available, or uses permutation importance otherwise.
    """
    
    def __init__(self, model: BaseModel, n_repeats: int = 5, random_state: Optional[int] = None):
        """
        Initialize a FeatureImportanceExplainer.
        
        Args:
            model: A ClarivueXAI model wrapper
            n_repeats: Number of times to permute each feature (for permutation importance)
            random_state: Random seed for reproducibility
        """
        super().__init__(model, 'feature_importance')
        self.n_repeats = n_repeats
        self.random_state = random_state
        
    def explain_global(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> ExplanationResult:
        """
        Generate global explanations for the model using feature importances.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Returns:
            Global explanation results
        """
        # Get feature names
        feature_names = kwargs.get('feature_names', None)
        if feature_names is None:
            feature_names = self.model.feature_names
        if feature_names is None:
            feature_names = get_feature_names(X)
        
        # Try to get feature importances directly from the model
        importances = self.model.get_feature_importances()
        
        # If not available, compute permutation importance
        if importances is None:
            importances = self._compute_permutation_importance(X, **kwargs)
        
        # Normalize importances
        if importances is not None:
            importances = importances / np.sum(importances)
        
        # Create explanation data
        explanation_data = {
            'importances': importances,
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
        Generate local explanations for specific instances.
        
        Feature importance explainers don't provide local explanations,
        so this method raises a ValueError.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Raises:
            ValueError: Feature importance explainers don't provide local explanations
        """
        raise ValueError("Feature importance explainers don't provide local explanations. "
                        "Use SHAP, LIME, or another local explainer instead.")
    
    def _compute_permutation_importance(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> np.ndarray:
        """
        Compute permutation importance for the features.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Returns:
            Permutation importance scores
        """
        try:
            from sklearn.inspection import permutation_importance
        except ImportError:
            raise ImportError("scikit-learn is required to compute permutation importance. "
                             "Install it with 'pip install scikit-learn'.")
        
        # Get target variable if provided
        y = kwargs.get('y', None)
        if y is None:
            raise ValueError("Target variable 'y' is required to compute permutation importance")
        
        # Create a scikit-learn compatible model
        class SklearnCompatibleModel:
            def __init__(self, model):
                self.model = model
            
            def predict(self, X):
                return self.model.predict(X)
        
        sklearn_model = SklearnCompatibleModel(self.model)
        
        # Compute permutation importance
        result = permutation_importance(
            sklearn_model, X, y,
            n_repeats=self.n_repeats,
            random_state=self.random_state
        )
        
        return result.importances_mean