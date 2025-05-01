# Examples

This page contains examples demonstrating the usage of ClarivueXAI with different types of data and models.

## Table of Contents

- [Tabular Data Examples](#tabular-data-examples)
- [Text Data Examples](#text-data-examples)
- [Image Data Examples](#image-data-examples)
- [Time Series Data Examples](#time-series-data-examples)
- [Advanced Usage Examples](#advanced-usage-examples)

## Tabular Data Examples

### Random Forest Classifier with SHAP Explanation

This example shows how to explain a Random Forest model trained on the breast cancer dataset using SHAP values:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import clarivuexai as cxai
import matplotlib.pyplot as plt

# Load and prepare data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # Convert y to pandas Series for iloc compatibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a ClarivueXAI model wrapper
cxai_model = cxai.Model.from_sklearn(model, feature_names=X.columns)

# Create an explainer
explainer = cxai.Explainer(cxai_model)

# Generate global explanation using SHAP
global_exp = explainer.explain_global(X_test, method='shap')

# Generate local explanation for a single instance
local_exp = explainer.explain_local(X_test.iloc[0:1], method='shap')

# Visualize explanations
plt.figure(figsize=(12, 8))
global_exp.plot()
plt.title("Global SHAP Feature Importance")
plt.tight_layout()
plt.savefig("global_shap.png")

plt.figure(figsize=(12, 8))
local_exp.plot()
plt.title(f"Local SHAP Explanation for Instance 0 (True Class: {y_test.iloc[0]})")
plt.tight_layout()
plt.savefig("local_shap.png")
```

### Gradient Boosting Regressor with LIME Explanation

This example demonstrates how to use LIME to explain a Gradient Boosting regressor trained on the California housing dataset:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import clarivuexai as cxai
import matplotlib.pyplot as plt

# Load and prepare data
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = GradientBoostingRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a ClarivueXAI model wrapper
cxai_model = cxai.Model.from_sklearn(model, feature_names=X.columns)

# Create an explainer
explainer = cxai.Explainer(cxai_model)

# Generate local explanation using LIME
lime_exp = explainer.explain_local(X_test.iloc[0:1], method='lime')

# Visualize explanation
plt.figure(figsize=(10, 6))
lime_exp.plot()
plt.title(f"LIME Explanation for Instance 0 (True Value: {y_test[0]:.2f})")
plt.tight_layout()
plt.savefig("lime_explanation.png")
```

### Wine Classification with Interactive Visualizations

This example shows how to create interactive visualizations for a Random Forest model trained on the wine dataset:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import clarivuexai as cxai
import matplotlib.pyplot as plt
import os

# Import ClarivueXAI interactive visualization components
from clarivuexai.visualization.interactive import (
    interactive_shap_summary,
    interactive_shap_local,
    debug_interactive_plot
)

# Apply debug_interactive_plot decorator to interactive visualization functions
interactive_shap_summary = debug_interactive_plot(interactive_shap_summary)
interactive_shap_local = debug_interactive_plot(interactive_shap_local)

# Create output directory for saving visualizations
OUTPUT_DIR = "clarivuexai_wine_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and prepare data
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)  # Ensure it's a pandas Series for iloc compatibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a ClarivueXAI model wrapper
cxai_model = cxai.Model.from_sklearn(model, feature_names=X.columns)

# Create an explainer
explainer = cxai.Explainer(cxai_model)

# Generate global explanation using SHAP
global_exp = explainer.explain_global(X_test, method='shap')

# Generate local explanation for a single instance
local_exp = explainer.explain_local(X_test.iloc[0:1], method='shap')

# Visualize explanations (Static Plots)
plt.figure(figsize=(12, 8))
global_exp.plot()
plt.title("Global SHAP Feature Importance (Wine Dataset)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "global_shap_wine.png"))
plt.close()

plt.figure(figsize=(12, 8))
local_exp.plot()
plt.title(f"Local SHAP Explanation for Instance 0 (True Class: {y_test.iloc[0]})")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "local_shap_wine.png"))
plt.close()

# Interactive global SHAP summary plot
fig_shap_summary_interactive = interactive_shap_summary(global_exp)
fig_shap_summary_interactive.write_html(os.path.join(OUTPUT_DIR, "global_shap_interactive.html"))
print("Interactive global SHAP plot saved.")

# Interactive local SHAP explanation plot
fig_shap_local_interactive = interactive_shap_local(local_exp)
fig_shap_local_interactive.write_html(os.path.join(OUTPUT_DIR, "local_shap_interactive.html"))
print("Interactive local SHAP plot saved.")
```

## Text Data Examples

### Text Classification with LIME

This example demonstrates how to explain a text classification model using LIME:

```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import clarivuexai as cxai
import matplotlib.pyplot as plt

# Sample data
texts = [
    "I love this movie, it's amazing!",
    "The plot was terrible and the acting was even worse.",
    "Great performance and excellent directing.",
    "Poor script, bad acting, terrible movie overall.",
    "This is the best film I've seen all year!",
    "Waste of time and money. Don't watch it.",
    "The soundtrack was perfect for the movie.",
    "I hated every minute of this boring film."
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1: positive, 0: negative

# Create and train model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(texts, labels)

# Create a ClarivueXAI model wrapper with custom functions
def predict_fn(text_samples):
    return pipeline.predict_proba(text_samples)

cxai_model = cxai.Model.from_custom(pipeline, predict_fn=predict_fn)

# Create an explainer - initialize directly with LimeExplainer
from clarivuexai.explainers.lime_explainers import LimeExplainer
lime_explainer = LimeExplainer(cxai_model, mode='text')

# Generate local explanation for a new text
new_text = ["I found the movie quite entertaining despite some plot holes."]

# Call the explain_local method directly
local_exp = lime_explainer.explain_local(new_text)

# Visualize explanation
plt.figure(figsize=(12, 6))
local_exp.plot()
plt.title("LIME Explanation for Text Classification")
plt.tight_layout()
plt.savefig("text_lime_explanation.png")
```

### Sentiment Analysis with SHAP

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import clarivuexai as cxai

# Sample sentiment analysis data
texts = [
    "This product exceeded my expectations, highly recommend!",
    "Not worth the money, I'm very disappointed.",
    "Okay product but nothing special.",
    "Absolutely love it, best purchase ever!",
    "Terrible quality and customer service."
]
labels = [1, 0, 0.5, 1, 0]  # 1: positive, 0: negative, 0.5: neutral

# Create and train model
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=500)),
    ('clf', LogisticRegression(random_state=42))
])
pipeline.fit(texts, labels)

# Create ClarivueXAI model wrapper
cxai_model = cxai.Model.from_custom(
    pipeline,
    predict_fn=lambda x: pipeline.predict_proba(x)
)

# Create explainer
explainer = cxai.Explainer(cxai_model)

# Generate local explanation for a new text
new_text = ["The product is good, but shipping took forever."]
shap_exp = explainer.explain_local(new_text, method='shap')

# Visualize explanation
shap_exp.plot()
```

## Image Data Examples

### CNN Image Classification with Integrated Gradients

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import clarivuexai as cxai
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
plt.savefig("image_ig_explanation.png")
```

### Object Detection with SHAP

```python
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import clarivuexai as cxai
import matplotlib.pyplot as plt
from PIL import Image

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess image
img_path = 'street_scene.jpg'
img = Image.open(img_path).convert('RGB')
img_tensor = torchvision.transforms.functional.to_tensor(img).unsqueeze(0)

# Create ClarivueXAI model wrapper
def predict_fn(img_tensor):
    with torch.no_grad():
        return model(img_tensor)

cxai_model = cxai.Model.from_custom(model, predict_fn=predict_fn)

# Create explainer
explainer = cxai.Explainer(cxai_model)

# Generate explanation using SHAP
shap_exp = explainer.explain_local(img_tensor, method='shap')

# Visualize explanation
plt.figure(figsize=(12, 10))
shap_exp.plot()
plt.title("SHAP Explanation for Object Detection")
plt.tight_layout()
plt.savefig("object_detection_shap.png")
```

## Time Series Data Examples

### LSTM Forecasting with Feature Attribution

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import clarivuexai as cxai
import matplotlib.pyplot as plt

# Generate sample time series data
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', periods=1000, freq='H')
series = pd.Series(np.sin(np.arange(1000) * 0.1) + np.random.normal(0, 0.1, 1000), index=dates)

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 24  # 24 hours for prediction
X, y = create_sequences(series.values, seq_length)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(seq_length, 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

# Create ClarivueXAI model wrapper
cxai_model = cxai.Model.from_tensorflow(model)

# Create explainer
explainer = cxai.Explainer(cxai_model)

# Generate explanation for a test sample
test_sample = X_test[0:1]
time_exp = explainer.explain_local(test_sample, method='integrated_gradients')

# Visualize explanation
plt.figure(figsize=(12, 6))
time_exp.plot()
plt.title("Feature Attribution for Time Series Forecast")
plt.xlabel("Time Steps")
plt.tight_layout()
plt.savefig("time_series_explanation.png")
```

### Time Series Classification with SHAP

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import clarivuexai as cxai
import matplotlib.pyplot as plt

# Generate sample multivariate time series data
np.random.seed(42)
n_samples = 1000
n_features = 5
seq_length = 50

# Generate classification datasets (2 classes)
X = np.random.randn(n_samples, seq_length, n_features)
# Class depends on patterns in the first two features
y = (np.mean(X[:, :, 0], axis=1) > 0) & (np.std(X[:, :, 1], axis=1) < 1)
y = y.astype(int)

# Reshape to 2D for RandomForest (samples, features*seq_length)
X_reshaped = X.reshape(n_samples, seq_length * n_features)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Create feature names
feature_names = [f'feature_{i}_time_{t}' for i in range(n_features) for t in range(seq_length)]

# Train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create a ClarivueXAI model wrapper
cxai_model = cxai.Model.from_sklearn(model, feature_names=feature_names)

# Create an explainer
explainer = cxai.Explainer(cxai_model)

# Generate SHAP explanations
global_exp = explainer.explain_global(X_test, method='shap')
local_exp = explainer.explain_local(X_test[0:1], method='shap')

# Visualize explanations
plt.figure(figsize=(14, 8))
global_exp.plot()
plt.title("Global SHAP Feature Importance for Time Series Classification")
plt.tight_layout()
plt.savefig("ts_global_shap.png")

plt.figure(figsize=(14, 8))
local_exp.plot()
plt.title(f"Local SHAP Explanation for Time Series Instance (True Class: {y_test[0]})")
plt.tight_layout()
plt.savefig("ts_local_shap.png")
```
