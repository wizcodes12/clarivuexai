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