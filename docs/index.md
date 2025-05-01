ClarivueXAI: Unified Explainable AI Framework
ClarivueXAI is a Python library that provides a unified framework for explaining AI models across different frameworks, model types, and data formats with a consistent API.
Problem Statement
Explainable AI (XAI) is fragmented, with different tools for various model types (e.g., scikit-learn, TensorFlow, PyTorch) and data formats (e.g., tabular, text, images, time series). This forces data scientists and ML engineers to learn multiple libraries, leading to inconsistent explanations and slower workflows.
Solution
ClarivueXAI offers a unified framework that simplifies XAI by providing:

A consistent API for explaining models across frameworks and data types.
Support for scikit-learn, TensorFlow, PyTorch, and custom models.
Handling of tabular, text, image, and time series data.
Global and local explanation methods (e.g., SHAP, LIME, Integrated Gradients).
Advanced visualization tools, including interactive plots and dashboards.

See the Examples for practical demonstrations of these capabilities.
Key Features

Unified API: Consistent interface for all model types and explanation methods.
Multi-modal Support: Works with tabular, text, image, and time series data, as shown in Examples.
Framework Agnostic: Supports scikit-learn, TensorFlow, PyTorch, and custom models.
Global & Local Explanations: Provides model-level (global) and instance-level (local) insights.
Advanced Visualization: Includes static and interactive plots, demonstrated in Wine Classification Example.

Installation
pip install clarivuexai

For specific framework support:
# TensorFlow support
pip install clarivuexai[tensorflow]

# PyTorch support
pip install clarivuexai[torch]

# All dependencies
pip install clarivuexai[all]

Getting Started
To quickly start using ClarivueXAI, check the Quickstart Guide. For detailed use cases, explore the Examples, which cover:

Tabular data with Random Forest and SHAP (Example).
Text classification with LIME (Example).
Image classification with Integrated Gradients (Example).
Time series forecasting with LSTM (Example).

Documentation

Quickstart Guide: Basic setup and usage.
API Reference: Detailed API documentation.
Examples: Comprehensive use cases across data types and models.
GitHub Repository.

Citation
If you use ClarivueXAI in your research, please cite:
@software{clarivuexai2025,
  author = {wizcodes},
  title = {ClarivueXAI: Unified Explainable AI Framework},
  year = {2025},
  url = {https://github.com/wizcodes12/clarivuexai},
}

License
Licensed under the MIT License. See the LICENSE file for details.
