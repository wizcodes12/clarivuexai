ClarivueXAI: Unified Explainable AI Framework

ClarivueXAI is a Python library for explainable AI, providing a unified API to explain models across frameworks (scikit-learn, TensorFlow, PyTorch, custom) and data types (tabular, text, images, time series).
Problem
Explainable AI is fragmented, with different tools for different models and data types, leading to inconsistent APIs and slower workflows for data scientists and ML engineers.
Solution
ClarivueXAI unifies XAI with:

A consistent API for all model types and explanation methods.
Support for multiple frameworks and data types.
Global and local explanations using methods like SHAP, LIME, and Integrated Gradients.
Advanced visualization tools, including interactive plots.

See Examples for detailed demonstrations.
Key Features

Unified API: Consistent interface for all models.
Multi-modal Support: Handles tabular, text, image, and time series data.
Framework Agnostic: Supports scikit-learn, TensorFlow, PyTorch, and custom models.
Global & Local Explanations: Model-level and instance-level insights.
Advanced Visualization: Static and interactive plots, as shown in Wine Classification.

Installation
# Base installation
pip install clarivuexai

# TensorFlow support
pip install clarivuexai[tensorflow]

# PyTorch support
pip install clarivuexai[torch]

# All dependencies
pip install clarivuexai[all]

Quick Example
This example, inspired by Random Forest Example, shows how to explain a scikit-learn model:
from clarivuexai import Model, Explainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Wrap with ClarivueXAI
cxai_model = Model.from_sklearn(model, feature_names=iris.feature_names)

# Create explainer
explainer = Explainer(cxai_model)

# Get explanations
global_exp = explainer.explain_global(X, method='shap')
local_exp = explainer.explain_local(X[0:1], method='shap')

# Visualize
plt.figure(figsize=(10, 6))
global_exp.plot()
plt.title("Global SHAP Feature Importance")
plt.tight_layout()
plt.savefig("global_shap.png")

Documentation
Full documentation is available at clarivuexai.readthedocs.io. Key sections:

Quickstart Guide: Basic setup.
API Reference: Detailed API docs.
Examples: Use cases for tabular, text, image, and time series data.

Contributing
Contributions are welcome! Submit a Pull Request on GitHub.
License
Licensed under the MIT License. See the LICENSE file.
