"""
Text data handler for ClarivueXAI.

This module provides a data handler for text data.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from clarivuexai.core.base import BaseDataHandler
from clarivuexai.core.registry import register_data_handler


@register_data_handler('text')
class TextDataHandler(BaseDataHandler):
    """
    Data handler for text data.
    
    This class provides methods for preprocessing and transforming
    text data for explanation.
    """
    
    def __init__(
        self, 
        max_features: int = 10000,
        max_length: Optional[int] = None,
        tokenizer: Optional[Any] = None,
        vectorizer: Optional[Any] = None
    ):
        """
        Initialize a TextDataHandler.
        
        Args:
            max_features: Maximum number of features (words) to consider
            max_length: Maximum length of text sequences
            tokenizer: Custom tokenizer to use
            vectorizer: Custom vectorizer to use
        """
        super().__init__('text')
        self.max_features = max_features
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.vectorizer = vectorizer
        
        # Initialize internal state
        self._fitted = False
        self._vocabulary = None
        
    def preprocess(self, data: Union[str, List[str], np.ndarray, pd.DataFrame, pd.Series]) -> List[str]:
        """
        Preprocess the input text data.
        
        Args:
            data: Input text data
            
        Returns:
            Preprocessed text data as a list of strings
        """
        # Convert to list of strings
        if isinstance(data, str):
            texts = [data]
        elif isinstance(data, list):
            texts = data
        elif isinstance(data, np.ndarray):
            texts = data.flatten().tolist()
        elif isinstance(data, pd.DataFrame):
            if data.shape[1] == 1:
                texts = data.iloc[:, 0].tolist()
            else:
                raise ValueError("DataFrame must have exactly one column for text data")
        elif isinstance(data, pd.Series):
            texts = data.tolist()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Basic preprocessing
        texts = [str(text).lower() for text in texts]
        
        # Remove special characters and extra whitespace
        import re
        texts = [re.sub(r'[^\w\s]', '', text) for text in texts]
        texts = [re.sub(r'\s+', ' ', text).strip() for text in texts]
        
        return texts
    
    def transform(self, data: Union[str, List[str], np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
        """
        Transform the input text data into a format suitable for explanation.
        
        Args:
            data: Input text data
            
        Returns:
            Transformed text data
        """
        # Preprocess the data
        texts = self.preprocess(data)
        
        # Use custom vectorizer if provided
        if self.vectorizer is not None:
            if not self._fitted:
                return self.vectorizer.fit_transform(texts)
            else:
                return self.vectorizer.transform(texts)
        
        # Use custom tokenizer if provided
        if self.tokenizer is not None:
            if not self._fitted:
                self._fitted = True
                return self.tokenizer.fit_on_texts(texts).texts_to_sequences(texts)
            else:
                return self.tokenizer.texts_to_sequences(texts)
        
        # Default: use scikit-learn's CountVectorizer
        try:
            from sklearn.feature_extraction.text import CountVectorizer
            if not hasattr(self, '_vectorizer'):
                self._vectorizer = CountVectorizer(max_features=self.max_features)
                self._fitted = True
                return self._vectorizer.fit_transform(texts).toarray()
            else:
                return self._vectorizer.transform(texts).toarray()
        except ImportError:
            # If scikit-learn is not available, use a simple bag-of-words approach
            if not self._fitted:
                # Build vocabulary
                words = set()
                for text in texts:
                    words.update(text.split())
                self._vocabulary = {word: i for i, word in enumerate(sorted(words)[:self.max_features])}
                self._fitted = True
            
            # Transform texts to bag-of-words
            result = np.zeros((len(texts), len(self._vocabulary)))
            for i, text in enumerate(texts):
                for word in text.split():
                    if word in self._vocabulary:
                        result[i, self._vocabulary[word]] += 1
            
            return result
    
    def get_feature_names(self) -> List[str]:
        """
        Get feature names (words) from the vectorizer.
        
        Returns:
            List of feature names
        """
        if self.vectorizer is not None:
            if hasattr(self.vectorizer, 'get_feature_names_out'):
                return self.vectorizer.get_feature_names_out()
            elif hasattr(self.vectorizer, 'get_feature_names'):
                return self.vectorizer.get_feature_names()
        
        if hasattr(self, '_vectorizer'):
            if hasattr(self._vectorizer, 'get_feature_names_out'):
                return self._vectorizer.get_feature_names_out()
            elif hasattr(self._vectorizer, 'get_feature_names'):
                return self._vectorizer.get_feature_names()
        
        if self._vocabulary is not None:
            return sorted(self._vocabulary.keys(), key=lambda x: self._vocabulary[x])
        
        return []
    
    def inverse_transform(self, data: np.ndarray) -> List[str]:
        """
        Transform data back to text format.
        
        Args:
            data: Transformed data
            
        Returns:
            List of reconstructed texts
        """
        if self.vectorizer is not None and hasattr(self.vectorizer, 'inverse_transform'):
            return self.vectorizer.inverse_transform(data)
        
        if hasattr(self, '_vectorizer') and hasattr(self._vectorizer, 'inverse_transform'):
            return self._vectorizer.inverse_transform(data)
        
        if self._vocabulary is not None:
            # Invert vocabulary mapping
            inv_vocabulary = {v: k for k, v in self._vocabulary.items()}
            
            # Reconstruct texts
            texts = []
            for row in data:
                words = []
                for i, count in enumerate(row):
                    if i in inv_vocabulary and count > 0:
                        words.extend([inv_vocabulary[i]] * int(count))
                texts.append(' '.join(words))
            
            return texts
        
        return []