"""
ClarivueXAI: Unified Explainable AI Framework.

ClarivueXAI provides a unified framework for explaining AI models across
different frameworks, model types, and data formats with a consistent API.
"""

__version__ = "0.1.0"

from clarivuexai.core.base import BaseModel, BaseExplainer, ExplanationResult
from clarivuexai.core.registry import registry

# Import model wrappers
from clarivuexai.models.sklearn_models import SklearnModel
from clarivuexai.models.tensorflow_models import TensorflowModel
from clarivuexai.models.pytorch_models import PytorchModel
from clarivuexai.models.custom_models import CustomModel

# Import explainers
from clarivuexai.explainers.feature_importance import FeatureImportanceExplainer
from clarivuexai.explainers.shap_explainers import ShapExplainer
from clarivuexai.explainers.lime_explainers import LimeExplainer
from clarivuexai.explainers.counterfactual import CounterfactualExplainer
from clarivuexai.explainers.integrated_gradients import IntegratedGradientsExplainer

# Import data handlers
from clarivuexai.data.tabular import TabularDataHandler
from clarivuexai.data.text import TextDataHandler
from clarivuexai.data.image import ImageDataHandler
from clarivuexai.data.timeseries import TimeSeriesDataHandler

# Import visualization tools
from clarivuexai.visualization.plots import (
    create_plot,
    plot_feature_importance,
    plot_local_explanation,
    plot_shap_local,
    plot_shap_summary,
    plot_shap_dependence,
    plot_lime_explanation,
    plot_counterfactual,
    plot_integrated_gradients
)

from clarivuexai.visualization.interactive import (
    create_interactive_plot,
    interactive_feature_importance,
    interactive_local_explanation,
    interactive_shap_local,
    interactive_shap_summary,
    interactive_counterfactual,
    interactive_integrated_gradients,
    debug_interactive_plot
)

from clarivuexai.visualization.dashboards import (
    create_dashboard,
    create_basic_dashboard,
    create_shap_dashboard,
    create_comparison_dashboard
)


# Convenience class for model creation
class Model:
    """
    Factory class for creating model wrappers.
    
    This class provides static methods for creating model wrappers
    for different frameworks.
    """
    
    @staticmethod
    def from_sklearn(model, feature_names=None):
        """Create a model wrapper for a scikit-learn model."""
        return SklearnModel(model, feature_names=feature_names)
    
    @staticmethod
    def from_tensorflow(model, feature_names=None):
        """Create a model wrapper for a TensorFlow model."""
        return TensorflowModel(model, feature_names=feature_names)
    
    @staticmethod
    def from_pytorch(model, feature_names=None):
        """Create a model wrapper for a PyTorch model."""
        return PytorchModel(model, feature_names=feature_names)
    
    @staticmethod
    def from_custom(model, predict_fn=None, feature_names=None):
        """Create a model wrapper for a custom model."""
        return CustomModel(model, predict_fn=predict_fn, feature_names=feature_names)


# Convenience class for explainer creation
class Explainer:
    """
    Factory class for creating explainers.
    
    This class provides methods for creating explainers
    for different explanation techniques.
    """
    
    def __init__(self, model):
        """
        Initialize an Explainer.
        
        Args:
            model: A ClarivueXAI model wrapper
        """
        self.model = model
    
    def feature_importance(self, **kwargs):
        """Create a feature importance explainer."""
        return FeatureImportanceExplainer(self.model, **kwargs)
    
    def shap(self, **kwargs):
        """Create a SHAP explainer."""
        return ShapExplainer(self.model, **kwargs)
    
    def lime(self, **kwargs):
        """Create a LIME explainer."""
        return LimeExplainer(self.model, **kwargs)
    
    def counterfactual(self, **kwargs):
        """Create a counterfactual explainer."""
        return CounterfactualExplainer(self.model, **kwargs)
    
    def integrated_gradients(self, **kwargs):
        """Create an integrated gradients explainer."""
        return IntegratedGradientsExplainer(self.model, **kwargs)
    
    def explain_global(self, X, method='auto', **kwargs):
        """
        Generate global explanations for the model.
        
        Args:
            X: Input data
            method: Explanation method to use
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Global explanation results
        """
        if method == 'auto':
            # Choose the best explainer based on the model type
            if hasattr(self.model.model, 'feature_importances_'):
                explainer = self.feature_importance()
            else:
                explainer = self.shap()
        elif method == 'feature_importance':
            explainer = self.feature_importance(**kwargs)
        elif method == 'shap':
            explainer = self.shap(**kwargs)
        elif method == 'lime':
            explainer = self.lime(**kwargs)
        elif method == 'counterfactual':
            explainer = self.counterfactual(**kwargs)
        elif method == 'integrated_gradients':
            explainer = self.integrated_gradients(**kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
        
        return explainer.explain_global(X, **kwargs)
    
    def explain_local(self, X, method='auto', **kwargs):
        """
        Generate local explanations for specific instances.
        
        Args:
            X: Input data
            method: Explanation method to use
            **kwargs: Additional arguments for the explainer
            
        Returns:
            Local explanation results
        """
        if method == 'auto':
            # Choose the best explainer based on the model type
            if self.model.framework == 'sklearn':
                explainer = self.lime()
            elif self.model.framework in ['tensorflow', 'pytorch']:
                explainer = self.integrated_gradients()
            else:
                explainer = self.shap()
        elif method == 'feature_importance':
            explainer = self.feature_importance(**kwargs)
        elif method == 'shap':
            explainer = self.shap(**kwargs)
        elif method == 'lime':
            explainer = self.lime(**kwargs)
        elif method == 'counterfactual':
            explainer = self.counterfactual(**kwargs)
        elif method == 'integrated_gradients':
            explainer = self.integrated_gradients(**kwargs)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
        
        return explainer.explain_local(X, **kwargs)