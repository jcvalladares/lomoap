"""
Example: Training and using a regression model with lomoap
"""
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lomoap import LocalModel, ModelTrainer


def main():
    print("=== lomoap Regression Example ===\n")
    
    # Generate sample data
    print("1. Generating sample regression data...")
    X, y = make_regression(
        n_samples=1000,
        n_features=10,
        n_informative=8,
        noise=10.0,
        random_state=42
    )
    print(f"   Generated {len(X)} samples with {X.shape[1]} features\n")
    
    # Train Linear Regression
    print("2. Training a Linear Regression model...")
    lr_model = LinearRegression()
    lr_trainer = ModelTrainer(lr_model)
    
    lr_metrics = lr_trainer.train(X, y, validation_split=0.2)
    print(f"   Training samples: {lr_metrics['train_samples']}")
    print(f"   Training R²: {lr_metrics['train_r2']:.4f}")
    print(f"   Training MSE: {lr_metrics['train_mse']:.4f}")
    print(f"   Validation R²: {lr_metrics['val_r2']:.4f}")
    print(f"   Validation MSE: {lr_metrics['val_mse']:.4f}\n")
    
    # Train Random Forest Regressor
    print("3. Training a Random Forest Regressor...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_trainer = ModelTrainer(rf_model)
    
    rf_metrics = rf_trainer.train(X, y, validation_split=0.2)
    print(f"   Training R²: {rf_metrics['train_r2']:.4f}")
    print(f"   Validation R²: {rf_metrics['val_r2']:.4f}\n")
    
    # Compare models
    print("4. Model comparison:")
    print(f"   Linear Regression - Val R²: {lr_metrics['val_r2']:.4f}")
    print(f"   Random Forest     - Val R²: {rf_metrics['val_r2']:.4f}\n")
    
    # Save the best model
    if rf_metrics['val_r2'] > lr_metrics['val_r2']:
        best_model = rf_trainer.get_model()
        best_name = "RandomForest"
        best_score = rf_metrics['val_r2']
    else:
        best_model = lr_trainer.get_model()
        best_name = "LinearRegression"
        best_score = lr_metrics['val_r2']
    
    print(f"5. Saving the best model ({best_name})...")
    local_model = LocalModel(best_model)
    metadata = {
        "model_type": best_name,
        "n_features": X.shape[1],
        "n_samples": len(X),
        "val_r2": best_score
    }
    local_model.save("regressor_model.pkl", metadata=metadata)
    print(f"   Model saved to: regressor_model.pkl\n")
    
    # Load and use the model
    print("6. Loading the model and making predictions...")
    loaded_model = LocalModel()
    loaded_model.load("regressor_model.pkl")
    
    # Make predictions on new data
    X_new = np.random.randn(5, 10)
    predictions = loaded_model.predict(X_new)
    
    print("   Sample predictions:")
    for i, pred in enumerate(predictions):
        print(f"   Sample {i+1}: {pred:.2f}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()
