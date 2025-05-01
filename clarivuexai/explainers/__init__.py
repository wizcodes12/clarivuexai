"""
Explainers for ClarivueXAI.

This module contains explainers for different explanation techniques to explain model predictions.
"""

from clarivuexai.explainers.feature_importance import FeatureImportanceExplainer
from clarivuexai.explainers.shap_explainers import ShapExplainer
from clarivuexai.explainers.lime_explainers import LimeExplainer
from clarivuexai.explainers.counterfactual import CounterfactualExplainer
from clarivuexai.explainers.integrated_gradients import IntegratedGradientsExplainer

# Define the list of available explainers
__all__ = [
    'FeatureImportanceExplainer',
    'ShapExplainer',
    'LimeExplainer',
    'CounterfactualExplainer',
    'IntegratedGradientsExplainer',
]

# Dictionary mapping explainer names to their respective classes
# This can be useful for programmatic access to explainers
EXPLAINER_REGISTRY = {
    'feature_importance': FeatureImportanceExplainer,
    'shap': ShapExplainer,
    'lime': LimeExplainer,
    'counterfactual': CounterfactualExplainer,
    'integrated_gradients': IntegratedGradientsExplainer,
}

def get_explainer(name, model, **kwargs):
    """
    Get an explainer instance by name.
    
    Args:
        name: Name of the explainer
        model: A ClarivueXAI model wrapper
        **kwargs: Additional arguments for the explainer
        
    Returns:
        An instance of the requested explainer
        
    Raises:
        ValueError: If the explainer name is not recognized
    """
    if name not in EXPLAINER_REGISTRY:
        raise ValueError(f"Explainer '{name}' not found. Available explainers: {list(EXPLAINER_REGISTRY.keys())}")
    
    return EXPLAINER_REGISTRY[name](model, **kwargs)