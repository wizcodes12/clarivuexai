"""
Command-line interface for ClarivueXAI.

This module provides a command-line interface for using ClarivueXAI.
"""

import argparse
import importlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai import Explainer, Model
from clarivuexai.core.utils import detect_framework


def load_model(model_path: str) -> Any:
    """
    Load a model from a file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded model
    """
    if model_path.endswith('.pkl') or model_path.endswith('.joblib'):
        try:
            import joblib
            return joblib.load(model_path)
        except ImportError:
            try:
                import pickle
                with open(model_path, 'rb') as f:
                    return pickle.load(f)
            except ImportError:
                raise ImportError("Either joblib or pickle is required to load the model")
    elif model_path.endswith('.h5'):
        try:
            import tensorflow as tf
            return tf.keras.models.load_model(model_path)
        except ImportError:
            raise ImportError("TensorFlow is required to load .h5 models")
    elif model_path.endswith('.pt') or model_path.endswith('.pth'):
        try:
            import torch
            return torch.load(model_path)
        except ImportError:
            raise ImportError("PyTorch is required to load .pt/.pth models")
    else:
        raise ValueError(f"Unsupported model file format: {model_path}")


def load_data(data_path: str) -> Union[np.ndarray, pd.DataFrame]:
    """
    Load data from a file.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        Loaded data
    """
    if data_path.endswith('.csv'):
        return pd.read_csv(data_path)
    elif data_path.endswith('.json'):
        return pd.read_json(data_path)
    elif data_path.endswith('.npy'):
        return np.load(data_path)
    elif data_path.endswith('.pkl') or data_path.endswith('.joblib'):
        try:
            import joblib
            return joblib.load(data_path)
        except ImportError:
            try:
                import pickle
                with open(data_path, 'rb') as f:
                    return pickle.load(f)
            except ImportError:
                raise ImportError("Either joblib or pickle is required to load the data")
    else:
        raise ValueError(f"Unsupported data file format: {data_path}")


def save_explanation(explanation: Dict[str, Any], output_path: str) -> None:
    """
    Save explanation results to a file.
    
    Args:
        explanation: Explanation results
        output_path: Path to save the results
    """
    # Convert numpy arrays to lists
    for key, value in explanation.items():
        if isinstance(value, np.ndarray):
            explanation[key] = value.tolist()
    
    # Save to file
    with open(output_path, 'w') as f:
        json.dump(explanation, f, indent=2)


def main():
    """Run the ClarivueXAI CLI."""
    parser = argparse.ArgumentParser(description='ClarivueXAI: Unified Explainable AI Framework')
    
    # Add arguments
    parser.add_argument('--model', required=True, help='Path to the model file')
    parser.add_argument('--data', required=True, help='Path to the data file')
    parser.add_argument('--output', default='explanation.json', help='Path to save the explanation results')
    parser.add_argument('--method', default='auto', choices=['auto', 'feature_importance', 'shap', 'lime', 'counterfactual', 'integrated_gradients'], help='Explanation method to use')
    parser.add_argument('--type', default='global', choices=['global', 'local'], help='Type of explanation to generate')
    parser.add_argument('--instance', type=int, default=0, help='Index of the instance to explain (for local explanations)')
    parser.add_argument('--feature-names', nargs='+', help='Names of the features')
    parser.add_argument('--framework', choices=['sklearn', 'tensorflow', 'pytorch', 'custom'], help='Framework of the model')
    parser.add_argument('--visualize', action='store_true', help='Visualize the explanation')
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Load model and data
        print(f"Loading model from {args.model}...")
        model = load_model(args.model)
        
        print(f"Loading data from {args.data}...")
        data = load_data(args.data)
        
        # Detect framework if not specified
        framework = args.framework
        if framework is None:
            try:
                framework = detect_framework(model)
                print(f"Detected framework: {framework}")
            except ValueError:
                print("Could not detect framework, using 'custom'")
                framework = 'custom'
        
        # Create ClarivueXAI model wrapper
        print("Creating model wrapper...")
        if framework == 'sklearn':
            cxai_model = Model.from_sklearn(model, feature_names=args.feature_names)
        elif framework == 'tensorflow':
            cxai_model = Model.from_tensorflow(model, feature_names=args.feature_names)
        elif framework == 'pytorch':
            cxai_model = Model.from_pytorch(model, feature_names=args.feature_names)
        else:
            cxai_model = Model.from_custom(model, feature_names=args.feature_names)
        
        # Create explainer
        print("Creating explainer...")
        explainer = Explainer(cxai_model)
        
        # Generate explanation
        print(f"Generating {args.type} explanation using {args.method} method...")
        if args.type == 'global':
            explanation = explainer.explain_global(data, method=args.method)
        else:
            if isinstance(data, pd.DataFrame):
                instance = data.iloc[[args.instance]]
            else:
                instance = data[args.instance:args.instance+1]
            
            explanation = explainer.explain_local(instance, method=args.method)
        
        # Save explanation
        print(f"Saving explanation to {args.output}...")
        save_explanation(explanation.to_dict(), args.output)
        
        # Visualize explanation if requested
        if args.visualize:
            print("Visualizing explanation...")
            fig = explanation.plot()
            
            # Save visualization
            viz_path = os.path.splitext(args.output)[0] + '.png'
            fig.savefig(viz_path)
            print(f"Visualization saved to {viz_path}")
        
        print("Done!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()