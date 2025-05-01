import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from clarivuexai.models.sklearn_models import SklearnModel
from clarivuexai.explainers import FeatureImportanceExplainer

def get_dummy_data():
    """
    Creates a small dummy dataset for testing.
    """
    X = pd.DataFrame({
        'age': [25, 45, 35, 50],
        'income': [50000, 80000, 60000, 90000]
    })
    y = [0, 1, 0, 1]
    return X, y

def get_wrapped_model(X, y):
    """
    Trains a scikit-learn model and wraps it using ClarivueXAI's SklearnModel.
    """
    model = LogisticRegression()
    model.fit(X, y)
    return SklearnModel(model, model_type='classifier', feature_names=X.columns.tolist())

def test_feature_importance_explainer():
    """
    Tests the FeatureImportanceExplainer with dummy data and prints the result.
    """
    print("\nRunning test_feature_importance_explainer...")

    X, y = get_dummy_data()
    model = get_wrapped_model(X, y)
    explainer = FeatureImportanceExplainer(model)

    explanation = explainer.explain_global(X, y=y)

    # Print outputs clearly
    print("Explanation type:", explanation.explanation_type)
    print("Explainer used:", explanation.explainer_name)
    print("Feature Names:", explanation.feature_names)
    print("Feature Importances:")

    for name, imp in zip(explanation.feature_names, explanation.data['importances']):
        print(f"  {name}: {imp:.4f}")

    # Simple assert to check it runs
    assert explanation.explanation_type == 'global'
    assert 'importances' in explanation.data
