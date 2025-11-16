"""
Trainer module for training models locally
"""
from typing import Any, Dict, Optional, Callable
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score


class ModelTrainer:
    """
    A class for training machine learning models locally.
    
    Provides methods for training, evaluation, and model management.
    """
    
    def __init__(self, model: Any):
        """
        Initialize ModelTrainer with a model.
        
        Args:
            model: A model object that implements fit() method (e.g., sklearn models)
        """
        self.model = model
        self.training_history = {}
        self.best_model = None
        self.best_score = None
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2,
        random_state: int = 42,
        **fit_params
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X: Training features
            y: Training labels/targets
            validation_split: Fraction of data to use for validation (0-1)
            random_state: Random seed for reproducibility
            **fit_params: Additional parameters to pass to model.fit()
            
        Returns:
            Dictionary containing training metrics
        """
        if not hasattr(self.model, 'fit'):
            raise ValueError("Model does not have a fit method")
        
        # Split data if validation is requested
        if validation_split > 0:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=random_state
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        
        # Train the model
        self.model.fit(X_train, y_train, **fit_params)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train)
        metrics = {
            'train_samples': len(X_train),
        }
        
        # Add appropriate metrics based on problem type
        if self._is_classification():
            metrics['train_accuracy'] = accuracy_score(y_train, train_pred)
        else:
            metrics['train_mse'] = mean_squared_error(y_train, train_pred)
            metrics['train_r2'] = r2_score(y_train, train_pred)
        
        # Evaluate on validation set if available
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            metrics['val_samples'] = len(X_val)
            
            if self._is_classification():
                val_accuracy = accuracy_score(y_val, val_pred)
                metrics['val_accuracy'] = val_accuracy
                # Track best model based on validation accuracy
                if self.best_score is None or val_accuracy > self.best_score:
                    self.best_score = val_accuracy
                    self.best_model = self.model
            else:
                val_mse = mean_squared_error(y_val, val_pred)
                val_r2 = r2_score(y_val, val_pred)
                metrics['val_mse'] = val_mse
                metrics['val_r2'] = val_r2
                # Track best model based on validation R2
                if self.best_score is None or val_r2 > self.best_score:
                    self.best_score = val_r2
                    self.best_model = self.model
        
        self.training_history = metrics
        return metrics
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test features
            y: Test labels/targets
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        predictions = self.model.predict(X)
        metrics = {}
        
        if self._is_classification():
            metrics['accuracy'] = accuracy_score(y, predictions)
        else:
            metrics['mse'] = mean_squared_error(y, predictions)
            metrics['r2'] = r2_score(y, predictions)
        
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Perform cross-validation on the model.
        
        Args:
            X: Features
            y: Labels/targets
            cv: Number of cross-validation folds
            scoring: Optional custom scoring function
            
        Returns:
            Dictionary containing cross-validation results
        """
        from sklearn.model_selection import cross_val_score
        
        if scoring is None:
            scoring = 'accuracy' if self._is_classification() else 'r2'
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring=scoring)
        
        return {
            'scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'cv_folds': cv
        }
    
    def get_model(self) -> Any:
        """
        Get the trained model.
        
        Returns:
            The trained model object
        """
        return self.model
    
    def get_best_model(self) -> Optional[Any]:
        """
        Get the best model based on validation performance.
        
        Returns:
            The best model object, or None if no validation was performed
        """
        return self.best_model
    
    def _is_classification(self) -> bool:
        """
        Check if the model is a classification model.
        
        Returns:
            True if classification, False otherwise
        """
        # Check common classification model attributes
        return hasattr(self.model, 'predict_proba') or \
               hasattr(self.model, 'classes_') or \
               'Classifier' in self.model.__class__.__name__
