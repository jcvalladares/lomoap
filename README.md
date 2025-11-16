# lomoap - Local Model Application

A Python application designed to run machine learning models locally and perform training. lomoap provides a simple, intuitive interface for loading, training, and deploying ML models without requiring cloud infrastructure.

## Features

- 🚀 **Local Model Execution**: Run ML models locally without external dependencies
- 🎓 **Built-in Training**: Train models with automatic validation and cross-validation
- 💾 **Model Persistence**: Save and load trained models with metadata
- 📊 **Multiple Model Types**: Support for classification and regression models
- 🔧 **CLI Interface**: Command-line tools for easy model management
- 🧪 **Well-Tested**: Comprehensive test suite included

## Installation

### From source

```bash
git clone https://github.com/jcvalladares/lomoap.git
cd lomoap
pip install -e .
```

### For development

```bash
pip install -e ".[dev]"
```

## Quick Start

### Training a Model

```python
from lomoap import LocalModel, ModelTrainer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Generate data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# Create and train model
model = LogisticRegression()
trainer = ModelTrainer(model)
metrics = trainer.train(X, y, validation_split=0.2)

print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")

# Save the trained model
local_model = LocalModel(trainer.get_model())
local_model.save("my_model.pkl", metadata={"accuracy": metrics['val_accuracy']})
```

### Using a Trained Model

```python
from lomoap import LocalModel
import numpy as np

# Load model
model = LocalModel()
model.load("my_model.pkl")

# Make predictions
X_new = np.random.randn(10, 20)
predictions = model.predict(X_new)
probabilities = model.predict_proba(X_new)
```

## CLI Usage

lomoap provides a command-line interface for common tasks:

### Run Predictions

```bash
# Basic prediction
lomoap predict my_model.pkl input_data.npy

# Save predictions to file
lomoap predict my_model.pkl input_data.npy --output predictions.csv

# Get probability predictions (for classification)
lomoap predict my_model.pkl input_data.npy --proba
```

### Display Model Info

```bash
lomoap info my_model.pkl
```

### Version Information

```bash
lomoap version
```

## API Reference

### LocalModel

The `LocalModel` class handles model loading, saving, and inference.

**Methods:**
- `load(path)`: Load a model from file
- `save(path, metadata=None)`: Save model to file with optional metadata
- `predict(X)`: Make predictions on input data
- `predict_proba(X)`: Get probability predictions (classification only)

### ModelTrainer

The `ModelTrainer` class handles model training and evaluation.

**Methods:**
- `train(X, y, validation_split=0.2, **fit_params)`: Train the model
- `evaluate(X, y)`: Evaluate model on test data
- `cross_validate(X, y, cv=5)`: Perform cross-validation
- `get_model()`: Get the trained model
- `get_best_model()`: Get the best model from validation

## Examples

See the `examples/` directory for complete examples:

- `classification_example.py`: Classification model training and usage
- `regression_example.py`: Regression model training and comparison

Run examples:

```bash
python examples/classification_example.py
python examples/regression_example.py
```

## Development

### Running Tests

```bash
pytest
```

### Running Tests with Coverage

```bash
pytest --cov=lomoap --cov-report=html
```

## Supported Model Types

lomoap works with any scikit-learn compatible model that implements:
- `fit(X, y)` for training
- `predict(X)` for predictions
- `predict_proba(X)` for probability predictions (classification)

Tested with:
- Linear models (LogisticRegression, LinearRegression)
- Tree-based models (RandomForest, DecisionTree)
- Ensemble models (GradientBoosting, AdaBoost)
- And many more!

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.