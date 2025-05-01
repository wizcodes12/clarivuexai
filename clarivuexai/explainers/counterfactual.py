"""
Counterfactual explainer for ClarivueXAI.

This module provides an explainer that generates counterfactual explanations
for model predictions.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseExplainer, BaseModel, ExplanationResult
from clarivuexai.core.registry import register_explainer
from clarivuexai.core.utils import get_feature_names


@register_explainer('counterfactual')
class CounterfactualExplainer(BaseExplainer):
    """
    Explainer that generates counterfactual explanations.
    
    This explainer generates counterfactual examples that would change
    the model's prediction to a desired outcome.
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        feature_ranges: Optional[Dict[str, tuple]] = None,
        categorical_features: Optional[List[str]] = None,
        max_iter: int = 1000,
        step_size: float = 0.1,
        random_state: Optional[int] = None
    ):
        """
        Initialize a CounterfactualExplainer.
        
        Args:
            model: A ClarivueXAI model wrapper
            feature_ranges: Dictionary mapping feature names to (min, max) tuples
            categorical_features: List of categorical feature names
            max_iter: Maximum number of iterations for optimization
            step_size: Step size for optimization
            random_state: Random seed for reproducibility
        """
        super().__init__(model, 'counterfactual')
        self.feature_ranges = feature_ranges or {}
        self.categorical_features = categorical_features or []
        self.max_iter = max_iter
        self.step_size = step_size
        self.random_state = random_state
        
        # Set random seed
        if self.random_state is not None:
            np.random.seed(self.random_state)
    
    def explain_global(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> ExplanationResult:
        """
        Generate global explanations for the model using counterfactuals.
        
        Counterfactual explainers don't provide global explanations,
        so this method raises a ValueError.
        
        Args:
            X: Input data
            **kwargs: Additional arguments
            
        Raises:
            ValueError: Counterfactual explainers don't provide global explanations
        """
        raise ValueError("Counterfactual explainers don't provide global explanations. "
                        "Use feature importance, SHAP, or another global explainer instead.")
    
    def explain_local(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        target_class: Optional[int] = None, 
        **kwargs
    ) -> ExplanationResult:
        """
        Generate local explanations for specific instances using counterfactuals.
        
        Args:
            X: Input data
            target_class: Target class for counterfactual (for classifiers)
            **kwargs: Additional arguments
            
        Returns:
            Local explanation results
        """
        # Get feature names
        feature_names = kwargs.get('feature_names', None)
        if feature_names is None:
            feature_names = self.model.feature_names
        if feature_names is None:
            feature_names = get_feature_names(X)
        
        # Convert input to numpy array
        if isinstance(X, pd.DataFrame):
            X_array = X.values
        else:
            X_array = X
        
        # Ensure we have a single instance
        if len(X_array.shape) > 1 and X_array.shape[0] > 1:
            X_array = X_array[0:1]
        
        # Get current prediction
        current_prediction = self.model.predict(X_array)[0]
        
        # For classifiers, determine target class
        if self.model.is_classifier:
            if target_class is None:
                # If no target class is specified, use a different class
                if self.model.n_classes_ is not None and self.model.n_classes_ > 2:
                    # For multi-class, use the second most likely class
                    proba = self.model.predict_proba(X_array)[0]
                    sorted_classes = np.argsort(proba)[::-1]
                    target_class = sorted_classes[1]
                else:
                    # For binary classification, use the opposite class
                    target_class = 1 - current_prediction
        
        # Generate counterfactual
        counterfactual, success = self._generate_counterfactual(
            X_array[0], target_class, feature_names, **kwargs
        )
        
        # Calculate feature importance as the difference between original and counterfactual
        feature_importance = np.abs(counterfactual - X_array[0])
        
        # Create explanation data
        explanation_data = {
            'original': X_array[0],
            'counterfactual': counterfactual,
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'success': success,
            'current_prediction': current_prediction,
            'target_class': target_class
        }
        
        return ExplanationResult(
            explanation_type='local',
            data=explanation_data,
            feature_names=feature_names,
            explainer_name=self.name
        )
    
    def _generate_counterfactual(
        self, 
        instance: np.ndarray, 
        target_class: Optional[int], 
        feature_names: List[str],
        **kwargs
    ) -> tuple:
        """
        Generate a counterfactual example for the given instance.
        
        Args:
            instance: Original instance
            target_class: Target class for counterfactual (for classifiers)
            feature_names: List of feature names
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (counterfactual, success)
        """
        # Initialize counterfactual as a copy of the original instance
        counterfactual = instance.copy()
        
        # Get feature ranges if not provided
        if not self.feature_ranges:
            for i, feature in enumerate(feature_names):
                if feature in self.categorical_features:
                    # For categorical features, use the original value
                    self.feature_ranges[feature] = (instance[i], instance[i])
                else:
                    # For numerical features, use a reasonable range
                    self.feature_ranges[feature] = (
                        instance[i] - 2 * np.abs(instance[i]),
                        instance[i] + 2 * np.abs(instance[i])
                    )
        
        # Define the objective function
        def objective(x):
            # Predict using the model
            if self.model.is_classifier:
                pred = self.model.predict_proba(x.reshape(1, -1))[0]
                if target_class is not None:
                    # Maximize probability of target class
                    return -pred[target_class]
                else:
                    # Minimize probability of current class
                    current_class = self.model.predict(instance.reshape(1, -1))[0]
                    return pred[current_class]
            else:
                # For regression, minimize the difference from the target value
                pred = self.model.predict(x.reshape(1, -1))[0]
                target_value = kwargs.get('target_value', instance.mean())
                return np.abs(pred - target_value)
        
        # Perform optimization
        best_counterfactual = counterfactual.copy()
        best_objective = objective(counterfactual)
        
        for _ in range(self.max_iter):
            # Randomly select a feature to modify
            feature_idx = np.random.randint(0, len(feature_names))
            feature = feature_names[feature_idx]
            
            # Skip categorical features for now
            if feature in self.categorical_features:
                continue
            
            # Get feature range
            feature_min, feature_max = self.feature_ranges.get(
                feature, (counterfactual[feature_idx] - 1, counterfactual[feature_idx] + 1)
            )
            
            # Generate a new value for the feature
            if np.random.random() < 0.5:
                # Increase the feature value
                new_value = min(
                    counterfactual[feature_idx] + self.step_size * (feature_max - feature_min),
                    feature_max
                )
            else:
                # Decrease the feature value
                new_value = max(
                    counterfactual[feature_idx] - self.step_size * (feature_max - feature_min),
                    feature_min
                )
            
            # Create a new counterfactual with the modified feature
            new_counterfactual = counterfactual.copy()
            new_counterfactual[feature_idx] = new_value
            
            # Evaluate the new counterfactual
            new_objective = objective(new_counterfactual)
            
            # Update if the new counterfactual is better
            if new_objective < best_objective:
                best_counterfactual = new_counterfactual.copy()
                best_objective = new_objective
                counterfactual = new_counterfactual.copy()
            
            # Check if we've found a good enough counterfactual
            if self.model.is_classifier:
                pred_class = self.model.predict(counterfactual.reshape(1, -1))[0]
                if pred_class == target_class:
                    break
            else:
                pred_value = self.model.predict(counterfactual.reshape(1, -1))[0]
                target_value = kwargs.get('target_value', instance.mean())
                if np.abs(pred_value - target_value) < kwargs.get('threshold', 0.1):
                    break
        
        # Check if the counterfactual is successful
        if self.model.is_classifier:
            success = self.model.predict(best_counterfactual.reshape(1, -1))[0] == target_class
        else:
            pred_value = self.model.predict(best_counterfactual.reshape(1, -1))[0]
            target_value = kwargs.get('target_value', instance.mean())
            success = np.abs(pred_value - target_value) < kwargs.get('threshold', 0.1)
        
        return best_counterfactual, success