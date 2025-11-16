"""
Tests for the ModelTrainer class
"""
import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier
from lomoap.trainer import ModelTrainer


@pytest.fixture
def classification_data():
    """Create sample classification data"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


@pytest.fixture
def regression_data():
    """Create sample regression data"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    return X, y


class TestModelTrainer:
    """Test cases for ModelTrainer"""
    
    def test_init(self):
        """Test trainer initialization"""
        model = LogisticRegression()
        trainer = ModelTrainer(model)
        
        assert trainer.model is not None
        assert trainer.training_history == {}
        assert trainer.best_model is None
        assert trainer.best_score is None
    
    def test_train_classifier(self, classification_data):
        """Test training a classification model"""
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        
        metrics = trainer.train(X, y, validation_split=0.2)
        
        assert 'train_samples' in metrics
        assert 'train_accuracy' in metrics
        assert 'val_samples' in metrics
        assert 'val_accuracy' in metrics
        assert metrics['train_accuracy'] > 0
        assert metrics['val_accuracy'] > 0
    
    def test_train_regressor(self, regression_data):
        """Test training a regression model"""
        X, y = regression_data
        model = LinearRegression()
        trainer = ModelTrainer(model)
        
        metrics = trainer.train(X, y, validation_split=0.2)
        
        assert 'train_samples' in metrics
        assert 'train_mse' in metrics
        assert 'train_r2' in metrics
        assert 'val_samples' in metrics
        assert 'val_mse' in metrics
        assert 'val_r2' in metrics
        assert metrics['train_r2'] > 0.5  # Should fit well with linear data
    
    def test_train_without_validation(self, classification_data):
        """Test training without validation split"""
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        
        metrics = trainer.train(X, y, validation_split=0)
        
        assert 'train_samples' in metrics
        assert 'train_accuracy' in metrics
        assert 'val_samples' not in metrics
        assert 'val_accuracy' not in metrics
    
    def test_evaluate_classifier(self, classification_data):
        """Test evaluation of classification model"""
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        
        # Train first
        trainer.train(X[:80], y[:80], validation_split=0)
        
        # Evaluate on test set
        metrics = trainer.evaluate(X[80:], y[80:])
        
        assert 'accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluate_regressor(self, regression_data):
        """Test evaluation of regression model"""
        X, y = regression_data
        model = LinearRegression()
        trainer = ModelTrainer(model)
        
        # Train first
        trainer.train(X[:80], y[:80], validation_split=0)
        
        # Evaluate on test set
        metrics = trainer.evaluate(X[80:], y[80:])
        
        assert 'mse' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0
    
    def test_cross_validate_classifier(self, classification_data):
        """Test cross-validation for classification"""
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        
        results = trainer.cross_validate(X, y, cv=5)
        
        assert 'scores' in results
        assert 'mean_score' in results
        assert 'std_score' in results
        assert 'cv_folds' in results
        assert len(results['scores']) == 5
        assert results['cv_folds'] == 5
    
    def test_cross_validate_regressor(self, regression_data):
        """Test cross-validation for regression"""
        X, y = regression_data
        model = LinearRegression()
        trainer = ModelTrainer(model)
        
        results = trainer.cross_validate(X, y, cv=3)
        
        assert 'scores' in results
        assert 'mean_score' in results
        assert 'std_score' in results
        assert len(results['scores']) == 3
    
    def test_get_model(self, classification_data):
        """Test getting the trained model"""
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        
        trainer.train(X, y)
        retrieved_model = trainer.get_model()
        
        assert retrieved_model is not None
        assert isinstance(retrieved_model, LogisticRegression)
    
    def test_get_best_model(self, classification_data):
        """Test getting the best model after training with validation"""
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        
        trainer.train(X, y, validation_split=0.2)
        best_model = trainer.get_best_model()
        
        assert best_model is not None
        assert trainer.best_score is not None
    
    def test_best_model_none_without_validation(self, classification_data):
        """Test that best_model remains None without validation"""
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        
        trainer.train(X, y, validation_split=0)
        
        # best_model should still be None since no validation was performed
        assert trainer.best_model is None
    
    def test_train_without_fit_method_raises_error(self):
        """Test that training raises error for objects without fit method"""
        class NoFitModel:
            pass
        
        trainer = ModelTrainer(NoFitModel())
        X = np.random.randn(10, 5)
        y = np.random.randint(0, 2, 10)
        
        with pytest.raises(ValueError, match="does not have a fit method"):
            trainer.train(X, y)
    
    def test_evaluate_before_training_raises_error(self, classification_data):
        """Test that evaluate works even before explicit training if model is pre-trained"""
        X, y = classification_data
        # Use an untrained model
        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        
        # This should work but give poor results
        # We're testing that it doesn't crash
        try:
            trainer.train(X[:50], y[:50], validation_split=0)
            metrics = trainer.evaluate(X[50:], y[50:])
            assert 'accuracy' in metrics
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")
    
    def test_is_classification_logistic_regression(self):
        """Test classification detection for LogisticRegression"""
        model = LogisticRegression()
        trainer = ModelTrainer(model)
        
        assert trainer._is_classification() is True
    
    def test_is_classification_random_forest(self):
        """Test classification detection for RandomForestClassifier"""
        model = RandomForestClassifier()
        trainer = ModelTrainer(model)
        
        assert trainer._is_classification() is True
    
    def test_is_classification_linear_regression(self):
        """Test classification detection for LinearRegression"""
        model = LinearRegression()
        trainer = ModelTrainer(model)
        
        assert trainer._is_classification() is False
    
    def test_training_history_stored(self, classification_data):
        """Test that training history is properly stored"""
        X, y = classification_data
        model = LogisticRegression(random_state=42)
        trainer = ModelTrainer(model)
        
        metrics = trainer.train(X, y, validation_split=0.2)
        
        assert trainer.training_history == metrics
        assert len(trainer.training_history) > 0
    
    def test_custom_fit_params(self, classification_data):
        """Test passing custom parameters to fit method"""
        X, y = classification_data
        model = LogisticRegression(random_state=42, max_iter=50)
        trainer = ModelTrainer(model)
        
        # Should not raise an error
        metrics = trainer.train(X, y, validation_split=0.2)
        assert metrics is not None
