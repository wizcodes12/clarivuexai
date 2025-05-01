

<div align="center">

<img src="https://github.com/user-attachments/assets/391a5958-7c62-47e4-b513-11216389e0d7" alt="Clarivue X AI Logo" width="150"/>

<br/>

[![PyPI version](https://img.shields.io/pypi/v/clarivuexai.svg)](https://pypi.org/project/clarivuexai/)
[![Python](https://img.shields.io/pypi/pyversions/clarivuexai)](https://pypi.org/project/clarivuexai/)
[![Build Status](https://img.shields.io/github/workflow/status/clarivuexai/clarivuexai/CI)](https://github.com/clarivuexai/.github/workflows)
[![Documentation Status](https://img.shields.io/readthedocs/clarivuexai)](https://clarivuexai.readthedocs.io/)
[![License](https://img.shields.io/github/license/clarivuexai/clarivuexai)](https://github.com/wizcodes12/clarivuexai/blob/9c9cb43e4c7dc67e02ea26e02a164ae915323b43/LICENSE)
[![Downloads](https://img.shields.io/pypi/dm/clarivuexai)](https://pypi.org/project/clarivuexai/)

**A Comprehensive Python Library for Multi-Modal Machine Learning Explainability**

</div>


## 📖 Overview

ClarivueXAI is a powerful and flexible explainable AI library that enables users to understand and interpret machine learning models across multiple data modalities. Whether you're working with tabular data, text, images, or time series, ClarivueXAI provides a unified interface to generate intuitive and informative explanations for your models.

## ✨ Key Features

- **Universal Compatibility**: Works with scikit-learn, TensorFlow, PyTorch, and custom models
- **Multi-Modal Support**: Explain models for tabular, text, image, and time series data
- **Multiple Explainability Methods**: SHAP, LIME, Integrated Gradients, and more
- **Rich Visualizations**: Static and interactive plots for both global and local explanations
- **Flexible API**: Consistent interface across different models and data types
- **Production Ready**: Optimized for performance and scalability

## 🛠️ Installation

```bash
pip install clarivuexai
```

For GPU support with TensorFlow or PyTorch integration:

```bash
pip install clarivuexai[gpu]
```

## 🚀 Quick Start

### Tabular Data Example

```python
import clarivuexai as cxai
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# Load data
data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier().fit(X_train, y_train)

# Create ClarivueXAI wrapper
cxai_model = cxai.Model.from_sklearn(model, feature_names=data.feature_names)
explainer = cxai.Explainer(cxai_model)

# Generate explanations
global_exp = explainer.explain_global(X_test, method='shap')
local_exp = explainer.explain_local(X_test[0:1], method='shap')

# Visualize
global_exp.plot(title="Global Feature Importance")
local_exp.plot(title="Instance Explanation")
```

### Text Data Example

```python
import clarivuexai as cxai
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Train text classification model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', LogisticRegression())
])
pipeline.fit(texts, labels)

# Create ClarivueXAI wrapper
cxai_model = cxai.Model.from_custom(
    pipeline, 
    predict_fn=lambda x: pipeline.predict_proba(x)
)
explainer = cxai.Explainer(cxai_model)

# Generate explanation
lime_exp = explainer.explain_local(
    ["This product exceeded my expectations!"], 
    method='lime'
)
lime_exp.plot()
```

### Image Data Example

```python
import clarivuexai as cxai
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load pre-trained model
model = MobileNetV2(weights='imagenet')

# Load and preprocess image
img_path = 'example_image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Create ClarivueXAI model wrapper
def predict_fn(img_array):
    preds = model.predict(img_array)
    return preds

cxai_model = cxai.Model.from_custom(model, predict_fn=predict_fn)

# Create explainer
explainer = cxai.Explainer(cxai_model)

# Generate explanation using Integrated Gradients
ig_exp = explainer.explain_local(x, method='integrated_gradients')

# Visualize explanation
plt.figure(figsize=(10, 8))
ig_exp.plot()
plt.title("Integrated Gradients Explanation")
plt.tight_layout()
```

## 📊 Interactive Visualizations

ClarivueXAI provides powerful interactive visualizations:

```python
from clarivuexai.visualization.interactive import (
    interactive_shap_summary,
    interactive_shap_local
)

# Generate interactive plots
fig_interactive = interactive_shap_summary(global_exp)
fig_interactive.write_html("global_shap_interactive.html")
```

## 🔍 Supported Explainability Methods

| Method | Description | Best For |
|--------|-------------|----------|
| SHAP | SHapley Additive exPlanations | Feature importance, global & local explanations |
| LIME | Local Interpretable Model-agnostic Explanations | Local explanations, text & image data |
| Integrated Gradients | Attribution method for deep learning | Neural networks, image data |
| Permutation Importance | Feature importance via permutation | Quick feature assessment |
| Partial Dependence | Shows feature effect on predictions | Feature relationship analysis |

## 📚 Documentation

For comprehensive documentation, tutorials, and API reference, visit:
[https://clarivuexai.readthedocs.io/](https://clarivuexai.readthedocs.io/)

## 🗺️ Roadmap

- [ ] Additional explainability methods
- [ ] Model comparison tools
- [ ] Counterfactual explanations
- [ ] Adversarial robustness testing
- [ ] R language integration

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## 📜 Citation

If you use ClarivueXAI in your research, please cite:

```bibtex
@software{clarivuexai2025,
  author = wizcodes12,
  title = {ClarivueXAI: A Comprehensive Python Library for Multi-Modal Machine Learning Explainability},
  url = {https://github.com/wizcodes12/clarivuexai},
  version = {0.1.0},
  year = {2025},
}
```

## 📄 License

ClarivueXAI is released under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ by the WizCodes
</div>
