API Reference
This document provides a detailed API reference for ClarivueXAI, aligned with the use cases in Examples.
Top-Level API
Model Factory
class clarivuexai.Model

Factory class for creating model wrappers for different frameworks. See Examples for usage.
Methods

from_sklearn(model, feature_names=None): Wrap a scikit-learn model. Used in Random Forest Example.
from_tensorflow(model, feature_names=None): Wrap a TensorFlow model. Used in CNN Image Classification.
from_pytorch(model, feature_names=None): Wrap a PyTorch model. Used in Time Series Classification.
from_custom(model, predict_fn=None, feature_names=None): Wrap a custom model with a prediction function. Used in Text Classification.

Explainer Factory
class clarivuexai.Explainer

Factory class for creating explainers. Initialized with a ClarivueXAI model wrapper.
Methods

init(model): Initialize with a model wrapper.
**feature_importance(kwargs): Create a feature importance explainer.
**shap(kwargs): Create a SHAP explainer. Used in Random Forest Example.
**lime(kwargs): Create a LIME explainer. Used in Text Classification.
**counterfactual(kwargs): Create a counterfactual explainer.
**integrated_gradients(kwargs): Create an Integrated Gradients explainer. Used in CNN Image Classification.
**explain_global(X, method='auto', kwargs): Generate global explanations. See Wine Classification.
**explain_local(X, method='auto', kwargs): Generate local explanations for specific instances. See Gradient Boosting Example.

Core API
Base Classes
class clarivuexai.BaseModel

Base class for model wrappers.
Methods

init(model, feature_names=None): Initialize the wrapper.
predict(X): Make predictions.
predict_proba(X): Get probability predictions (if supported).
get_feature_names(): Retrieve feature names.

class clarivuexai.BaseExplainer

Base class for explainers.
Methods

init(model): Initialize with a model wrapper.
**explain_global(X, kwargs): Generate global explanations.
**explain_local(X, kwargs): Generate local explanations.

class clarivuexai.ExplanationResult

Holds explanation results.
Methods

init(values, feature_names=None, metadata=None): Initialize the result.
to_dict(): Convert to a dictionary.
**plot(kwargs): Plot the result. Used in most Examples.

Model Wrappers

clarivuexai.SklearnModel(BaseModel): For scikit-learn models.
clarivuexai.TensorflowModel(BaseModel): For TensorFlow models.
clarivuexai.PytorchModel(BaseModel): For PyTorch models.
clarivuexai.CustomModel(BaseModel): For custom models with a predict function.

Data Handlers

clarivuexai.TabularDataHandler(BaseDataHandler): For tabular data. See Random Forest Example.
clarivuexai.TextDataHandler(BaseDataHandler): For text data. See Text Classification.
clarivuexai.ImageDataHandler(BaseDataHandler): For image data. See CNN Image Classification.
clarivuexai.TimeSeriesDataHandler(BaseDataHandler): For time series data. See LSTM Forecasting.

Explainers

clarivuexai.FeatureImportanceExplainer(BaseExplainer): Uses model feature importance.
clarivuexai.ShapExplainer(BaseExplainer): Uses SHAP values.
clarivuexai.LimeExplainer(BaseExplainer): Uses LIME.
clarivuexai.CounterfactualExplainer(BaseExplainer): Generates counterfactuals.
clarivuexai.IntegratedGradientsExplainer(BaseExplainer): Uses Integrated Gradients.

Visualization
clarivuexai.create_plot(explanation, plot_type='bar', **kwargs)

Create a static plot. Used in Random Forest Example.
clarivuexai.plot_feature_importance(explanation, **kwargs)

Plot feature importance.
clarivuexai.plot_shap_summary(explanation, **kwargs)

Plot a SHAP summary. Used in Wine Classification.
clarivuexai.plot_lime_explanation(explanation, **kwargs)

Plot a LIME explanation. Used in Gradient Boosting Example.
clarivuexai.create_interactive_plot(explanation, plot_type='bar', **kwargs)

Create an interactive plot. Used in Wine Classification.
clarivuexai.create_dashboard(explanation, **kwargs)

Create a dashboard for explanations.
Registry
clarivuexai.registry

Manages model wrappers, explainers, and data handlers.
Methods

register_model(name, model_class): Register a model wrapper.
register_explainer(name, explainer_class): Register an explainer.
register_data_handler(name, data_handler_class): Register a data handler.
get_model(name): Retrieve a model wrapper.
get_explainer(name): Retrieve an explainer.
get_data_handler(name): Retrieve a data handler.

Utils
clarivuexai.detect_framework(model)

Detect the model’s framework.
clarivuexai.wrap_model(model, feature_names=None, **kwargs)

Wrap a model automatically.
clarivuexai.get_feature_names(X, feature_names=None)

Get feature names.
clarivuexai.check_model_compatibility(model, data Techno

Check model-data compatibility.
CLI
Command-line interface for ClarivueXAI. See Examples for usage.
clarivuexai --model MODEL --data DATA [--output OUTPUT] [--method METHOD] 
           [--type TYPE] [--instance INSTANCE] [--feature-names FEATURE_NAMES]
           [--framework FRAMEWORK] [--visualize]

Options:

--model: Path to model file (required).
--data: Path to data file (required).
--output: Output file (default: explanation.json).
--method: Explanation method (choices: auto, feature_importance, shap, lime, counterfactual, integrated_gradients; default:  debesauto).
--type: Explanation type (choices: global, local; default: global).
--instance: Instance index for local explanations (default: 0).
--feature-names: Feature names (optional).
--framework: Model framework (choices: sklearn, tensorflow, pytorch, custom; optional).
--visualize: Visualize the explanation (optional).

