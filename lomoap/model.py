"""
Model module for loading and running models locally
"""
import pickle
import json
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np


class LocalModel:
    """
    A class for loading and running machine learning models locally.
    
    Supports sklearn models and custom models that implement predict/predict_proba methods.
    """
    
    def __init__(self, model: Optional[Any] = None):
        """
        Initialize LocalModel with an optional pre-loaded model.
        
        Args:
            model: A pre-loaded model object (optional)
        """
        self.model = model
        self.metadata = {}
    
    def load(self, path: Union[str, Path]) -> "LocalModel":
        """
        Load a model from a file.
        
        Args:
            path: Path to the model file (.pkl for pickle, .json for metadata)
            
        Returns:
            self for method chaining
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        if path.suffix == ".pkl":
            with open(path, 'rb') as f:
                self.model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Try to load metadata if exists
        metadata_path = path.with_suffix('.json')
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        
        return self
    
    def save(self, path: Union[str, Path], metadata: Optional[dict] = None) -> None:
        """
        Save the model to a file.
        
        Args:
            path: Path where to save the model
            metadata: Optional metadata dictionary to save alongside the model
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save metadata if provided
        if metadata is not None:
            self.metadata = metadata
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Run prediction on input data.
        
        Args:
            X: Input features as numpy array
            
        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("No model loaded. Use load() or provide model in constructor")
        
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model does not have a predict method")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Run probability prediction on input data (for classification models).
        
        Args:
            X: Input features as numpy array
            
        Returns:
            Class probabilities as numpy array
        """
        if self.model is None:
            raise ValueError("No model loaded. Use load() or provide model in constructor")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not have a predict_proba method")
        
        return self.model.predict_proba(X)
