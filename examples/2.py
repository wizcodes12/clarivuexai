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

# Fix: Accessing numpy array directly instead of using .iloc
plt.title(f"LIME Explanation for Instance 0 (True Value: {y_test[0]:.2f})")
plt.tight_layout()
plt.savefig("lime_explanation.png")