"""
Example: Training and using a classification model with lomoap
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from lomoap import LocalModel, ModelTrainer


def main():
    print("=== lomoap Classification Example ===\n")
    
    # Generate sample data
    print("1. Generating sample classification data...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    print(f"   Generated {len(X)} samples with {X.shape[1]} features\n")
    
    # Create and train model
    print("2. Training a logistic regression model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    trainer = ModelTrainer(model)
    
    # Train with validation split
    metrics = trainer.train(X, y, validation_split=0.2)
    print(f"   Training samples: {metrics['train_samples']}")
    print(f"   Training accuracy: {metrics['train_accuracy']:.4f}")
    print(f"   Validation samples: {metrics['val_samples']}")
    print(f"   Validation accuracy: {metrics['val_accuracy']:.4f}\n")
    
    # Perform cross-validation
    print("3. Running 5-fold cross-validation...")
    cv_results = trainer.cross_validate(X, y, cv=5)
    print(f"   Mean CV score: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})\n")
    
    # Save the model
    print("4. Saving the trained model...")
    local_model = LocalModel(trainer.get_model())
    metadata = {
        "model_type": "LogisticRegression",
        "n_features": X.shape[1],
        "n_samples": len(X),
        "val_accuracy": metrics['val_accuracy']
    }
    local_model.save("classifier_model.pkl", metadata=metadata)
    print("   Model saved to: classifier_model.pkl\n")
    
    # Load and use the model
    print("5. Loading the model and making predictions...")
    loaded_model = LocalModel()
    loaded_model.load("classifier_model.pkl")
    
    # Make predictions on new data
    X_new = np.random.randn(5, 20)
    predictions = loaded_model.predict(X_new)
    probabilities = loaded_model.predict_proba(X_new)
    
    print("   Sample predictions:")
    for i, (pred, proba) in enumerate(zip(predictions, probabilities)):
        print(f"   Sample {i+1}: Class {pred} (confidence: {proba[pred]:.4f})")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
