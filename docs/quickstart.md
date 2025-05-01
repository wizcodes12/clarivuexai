Quickstart Guide
This guide provides a quick introduction to using ClarivueXAI. For detailed use cases, refer to the Examples.
Installation
Install the base package:
pip install clarivuexai

For additional dependencies:
# TensorFlow support
pip install clarivuexai[tensorflow]

# PyTorch support
pip install clarivuexai[torch]

# All dependencies
pip install clarivuexai[all]

Basic Usage
Below are examples demonstrating how to explain models using ClarivueXAI. These are simplified versions of the detailed examples in Examples.
1. Explaining a scikit-learn Model (Tabular Data)
This example uses a Random Forest Classifier on the Iris dataset, similar to the Random Forest Example.
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import clarivuexai as cxai
import matplotlib.pyplot as plt

# Load data
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Create ClarivueXAI model wrapper
cxai_model = cxai.Model.from_sklearn(model, feature_names=feature_names)

# Create explainer
explainer = cxai.Explainer(cxai_model)

# Generate explanations
global_exp = explainer.explain_global(X, method='shap')
local_exp = explainer.explain_local(X[0:1], method='shap')

# Visualize
plt.figure(figsize=(10, 6))
global_exp.plot()
plt.title("Global SHAP Feature Importance")
plt.tight_layout()
plt.savefig("global_shap_iris.png")

plt.figure(figsize=(10, 6))
local_exp.plot()
plt.title("Local SHAP Explanation for Instance 0")
plt.tight_layout()
plt.savefig("local_shap_iris.png")

2. Explaining a TensorFlow Model (Tabular Data)
This example uses a neural network, similar to the TensorFlow Example but adapted for tabular data.
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import clarivuexai as cxai
import matplotlib.pyplot as plt

# Generate data
X = np.random.rand(1000, 20)
y = (X[:, 0] > 0.5).astype(int)
feature_names = [f"feature_{i}" for i in range(20)]

# Create and train model
model = Sequential([
    Dense(10, activation='relu', input_shape=(20,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(X, y, epochs=5, verbose=0)

# Create ClarivueXAI model wrapper
cxai_model = cxai.Model.from_tensorflow(model, feature_names=feature_names)

# Create explainer
explainer = cxai.Explainer(cxai_model)

# Generate and visualize explanation
ig_exp = explainer.explain_local(X[0:1], method='integrated_gradients')
plt.figure(figsize=(10, 6))
ig_exp.plot()
plt.title("Integrated Gradients Explanation")
plt.tight_layout()
plt.savefig("ig_tensorflow.png")

3. Explaining a PyTorch Model (Text Data)
This example uses a simple neural network for text classification, inspired by the Text Classification Example.
import torch
import torch.nn as nn
import numpy as np
import clarivuexai as cxai
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer

# Sample text data
texts = ["Great product!", "Terrible experience.", "Very good service."]
labels = [1, 0, 1]
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(texts).toarray().astype(np.float32)
y = np.array(labels, dtype=np.float32)

# Define PyTorch model
class TextNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 5)
        self.fc2 = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

model = TextNN(input_size=X.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train model
for _ in range(5):
    optimizer.zero_grad()
    inputs = torch.from_numpy(X)
    targets = torch.from_numpy(y).view(-1, 1)
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

# Create ClarivueXAI model wrapper
cxai_model = cxai.Model.from_pytorch(model, feature_names=vectorizer.get_feature_names_out())

# Create explainer
explainer = cxai.Explainer(cxai_model)

# Generate and visualize explanation
shap_exp = explainer.explain_local(X[0:1], method='shap')
plt.figure(figsize=(10, 6))
shap_exp.plot()
plt.title("SHAP Explanation for Text Classification")
plt.tight_layout()
plt.savefig("shap_text.png")

4. Using the Command-Line Interface
Run ClarivueXAI from the command line, as shown in the Examples:
# Global SHAP explanation
clarivuexai --model model.pkl --data data.csv --output explanation.json --method shap --type global

# Local LIME explanation for instance 5
clarivuexai --model model.pkl --data data.csv --output explanation.json --method lime --type local --instance 5

# Visualize SHAP explanation
clarivuexai --model model.pkl --data data.csv --output explanation.json --method shap --visualize

Next Steps

Explore the API Reference for detailed class and method documentation.
Dive into Examples for advanced use cases, including image and time series data.
Visit the GitHub Repository for source code and issues.

