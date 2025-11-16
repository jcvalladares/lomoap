"""
Tests for the LocalModel class
"""
import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from sklearn.linear_model import LogisticRegression, LinearRegression
from lomoap.model import LocalModel


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y_classification = np.random.randint(0, 2, 100)
    y_regression = np.random.randn(100)
    return X, y_classification, y_regression


@pytest.fixture
def trained_classifier(sample_data):
    """Create a trained classification model"""
    X, y, _ = sample_data
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model


@pytest.fixture
def trained_regressor(sample_data):
    """Create a trained regression model"""
    X, _, y = sample_data
    model = LinearRegression()
    model.fit(X, y)
    return model


class TestLocalModel:
    """Test cases for LocalModel"""
    
    def test_init_empty(self):
        """Test initialization without model"""
        model = LocalModel()
        assert model.model is None
        assert model.metadata == {}
    
    def test_init_with_model(self, trained_classifier):
        """Test initialization with pre-loaded model"""
        model = LocalModel(trained_classifier)
        assert model.model is not None
        assert isinstance(model.model, LogisticRegression)
    
    def test_save_and_load_classifier(self, trained_classifier, temp_dir):
        """Test saving and loading a classification model"""
        model = LocalModel(trained_classifier)
        save_path = Path(temp_dir) / "classifier.pkl"
        
        # Save model
        metadata = {"type": "classifier", "version": "1.0"}
        model.save(save_path, metadata=metadata)
        
        assert save_path.exists()
        assert save_path.with_suffix('.json').exists()
        
        # Load model
        loaded_model = LocalModel()
        loaded_model.load(save_path)
        
        assert loaded_model.model is not None
        assert loaded_model.metadata == metadata
    
    def test_save_and_load_regressor(self, trained_regressor, temp_dir):
        """Test saving and loading a regression model"""
        model = LocalModel(trained_regressor)
        save_path = Path(temp_dir) / "regressor.pkl"
        
        # Save model
        model.save(save_path)
        
        assert save_path.exists()
        
        # Load model
        loaded_model = LocalModel()
        loaded_model.load(save_path)
        
        assert loaded_model.model is not None
    
    def test_predict_classifier(self, trained_classifier, sample_data):
        """Test prediction with classification model"""
        X, _, _ = sample_data
        model = LocalModel(trained_classifier)
        
        predictions = model.predict(X[:10])
        
        assert predictions is not None
        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)
    
    def test_predict_regressor(self, trained_regressor, sample_data):
        """Test prediction with regression model"""
        X, _, _ = sample_data
        model = LocalModel(trained_regressor)
        
        predictions = model.predict(X[:10])
        
        assert predictions is not None
        assert len(predictions) == 10
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_predict_proba(self, trained_classifier, sample_data):
        """Test probability prediction with classification model"""
        X, _, _ = sample_data
        model = LocalModel(trained_classifier)
        
        probabilities = model.predict_proba(X[:10])
        
        assert probabilities is not None
        assert probabilities.shape == (10, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0)
    
    def test_predict_without_model_raises_error(self):
        """Test that predict raises error when no model is loaded"""
        model = LocalModel()
        X = np.random.randn(10, 5)
        
        with pytest.raises(ValueError, match="No model loaded"):
            model.predict(X)
    
    def test_save_without_model_raises_error(self, temp_dir):
        """Test that save raises error when no model is loaded"""
        model = LocalModel()
        save_path = Path(temp_dir) / "model.pkl"
        
        with pytest.raises(ValueError, match="No model to save"):
            model.save(save_path)
    
    def test_load_nonexistent_file_raises_error(self):
        """Test that load raises error for nonexistent file"""
        model = LocalModel()
        
        with pytest.raises(FileNotFoundError):
            model.load("/nonexistent/path/model.pkl")
    
    def test_load_unsupported_format_raises_error(self, temp_dir):
        """Test that load raises error for unsupported file format"""
        model = LocalModel()
        bad_path = Path(temp_dir) / "model.txt"
        bad_path.touch()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            model.load(bad_path)
    
    def test_predict_proba_without_method_raises_error(self, trained_regressor, sample_data):
        """Test that predict_proba raises error when model doesn't support it"""
        X, _, _ = sample_data
        model = LocalModel(trained_regressor)
        
        with pytest.raises(ValueError, match="does not have a predict_proba method"):
            model.predict_proba(X)
    
    def test_save_creates_parent_directories(self, trained_classifier, temp_dir):
        """Test that save creates parent directories if they don't exist"""
        model = LocalModel(trained_classifier)
        save_path = Path(temp_dir) / "subdir" / "model.pkl"
        
        model.save(save_path)
        
        assert save_path.exists()
    
    def test_predictions_match_after_save_load(self, trained_classifier, sample_data, temp_dir):
        """Test that predictions are consistent after save/load"""
        X, _, _ = sample_data
        X_test = X[:10]
        
        # Original predictions
        original_model = LocalModel(trained_classifier)
        original_preds = original_model.predict(X_test)
        
        # Save and load
        save_path = Path(temp_dir) / "model.pkl"
        original_model.save(save_path)
        
        loaded_model = LocalModel()
        loaded_model.load(save_path)
        loaded_preds = loaded_model.predict(X_test)
        
        # Compare
        np.testing.assert_array_equal(original_preds, loaded_preds)
