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

