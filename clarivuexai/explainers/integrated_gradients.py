"""
Integrated Gradients explainer for ClarivueXAI.

This module provides an explainer that uses Integrated Gradients
to explain model predictions.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseExplainer, BaseModel, ExplanationResult
from clarivuexai.core.registry import register_explainer
from clarivuexai.core.utils import get_feature_names


@register_explainer('integrated_gradients')
class IntegratedGradientsExplainer(BaseExplainer):
    """
    Explainer that uses Integrated Gradients to explain model predictions.
    
    This explainer computes the integrated gradients of the model output
    with respect to the input features to explain predictions.
    """
    
    def __init__(
        self, 
        model: BaseModel, 
        baseline: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        steps: int = 50
    ):
        """
        Initialize an IntegratedGradientsExplainer.
        
        Args:
            model: A ClarivueXAI model wrapper
            baseline: Baseline input for integration (defaults to zeros)
            steps: Number of steps for integration
        """
        super().__init__(model, 'integrated_gradients')
        self.baseline = baseline
        self.steps = steps
    
    def explain_global(self, X: Union[np.ndarray, pd.DataFrame], **kwargs) -> ExplanationResult:
        """
        Generate global explanations for the model using Integrated Gradients.
        
        Integrated Gradients is primarily a local explanation method, so this method
        generates local explanations for a sample of instances and aggregates them
        to create a global explanation.
        
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
        
        # Sample instances for explanation
        n_samples = kwargs.get('n_samples', min(10, len(X)))
        if isinstance(X, pd.DataFrame):
            sample_indices = np.random.choice(len(X), n_samples, replace=False)
            sample = X.iloc[sample_indices]
        else:
            sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
            sample = X[sample_indices]
        
        # Generate local explanations for each sample
        attributions = []
        for i in range(n_samples):
            if isinstance(sample, pd.DataFrame):
                instance = sample.iloc[i:i+1]
            else:
                instance = sample[i:i+1]
            
            # Get local explanation
            local_explanation = self.explain_local(instance, **kwargs)
            attributions.append(local_explanation.data['attributions'])
        
        # Aggregate local explanations to create a global explanation
        mean_attributions = np.mean(attributions, axis=0)
        
        # Create explanation data
        explanation_data = {
            'attributions': mean_attributions,
            'feature_names': feature_names
        }
        
        return ExplanationResult(
            explanation_type='global',
            data=explanation_data,
            feature_names=feature_names,
            explainer_name=self.name
        )
    
    def explain_local(
        self, 
        X: Union[np.ndarray, pd.DataFrame], 
        target_class: Optional[int] = None, 
        **kwargs
    ) -> ExplanationResult:
        """
        Generate local explanations for specific instances using Integrated Gradients.
        
        Args:
            X: Input data
            target_class: Target class for explanation (for classifiers)
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
        
        # Create baseline if not provided
        if self.baseline is None:
            self.baseline = np.zeros_like(X_array)
        
        # Compute integrated gradients
        attributions = self._compute_integrated_gradients(X_array, target_class)
        
        # Create explanation data
        explanation_data = {
            'attributions': attributions,
            'feature_names': feature_names
        }
        
        return ExplanationResult(
            explanation_type='local',
            data=explanation_data,
            feature_names=feature_names,
            explainer_name=self.name
        )
    
    def _compute_integrated_gradients(
        self, 
        instance: np.ndarray, 
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute integrated gradients for the given instance.
        
        Args:
            instance: Input instance
            target_class: Target class for explanation (for classifiers)
            
        Returns:
            Integrated gradients attributions
        """
        # Check if the model supports gradient computation
        if not hasattr(self.model, 'get_gradients'):
            raise ValueError("Model does not support gradient computation. "
                            "Use a different explainer for this model.")
        
        # Create interpolated inputs
        alphas = np.linspace(0, 1, self.steps)
        interpolated_inputs = np.array([
            self.baseline + alpha * (instance - self.baseline)
            for alpha in alphas
        ])
        
        # Reshape interpolated inputs
        interpolated_inputs = interpolated_inputs.reshape(-1, instance.shape[1])
        
        # Compute gradients for all interpolated inputs
        gradients = []
        batch_size = 10  # Process in batches to avoid memory issues
        for i in range(0, len(interpolated_inputs), batch_size):
            batch = interpolated_inputs[i:i+batch_size]
            batch_gradients = self.model.get_gradients(batch, target_class)
            gradients.append(batch_gradients)
        
        gradients = np.vstack(gradients)
        
        # Reshape gradients to match interpolated inputs
        gradients = gradients.reshape(self.steps, -1)
        
        # Compute integrated gradients
        integrated_gradients = (instance - self.baseline) * np.mean(gradients, axis=0)
        
        return integrated_gradients[0]