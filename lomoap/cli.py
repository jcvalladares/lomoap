"""
Command-line interface for lomoap
"""
import argparse
import sys
import json
from pathlib import Path
import numpy as np
from .model import LocalModel


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="lomoap - Local Model Application for running and training ML models"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run prediction with a model')
    predict_parser.add_argument('model_path', help='Path to the model file')
    predict_parser.add_argument('data_path', help='Path to input data (npy or csv file)')
    predict_parser.add_argument('--output', '-o', help='Output file for predictions')
    predict_parser.add_argument('--proba', action='store_true', 
                               help='Use predict_proba for classification models')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display model information')
    info_parser.add_argument('model_path', help='Path to the model file')
    
    # Version command
    subparsers.add_parser('version', help='Display version information')
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        run_predict(args)
    elif args.command == 'info':
        run_info(args)
    elif args.command == 'version':
        from . import __version__
        print(f"lomoap version {__version__}")
    else:
        parser.print_help()
        sys.exit(1)


def run_predict(args):
    """Run prediction command"""
    try:
        # Load model
        model = LocalModel()
        model.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
        
        # Load data
        data_path = Path(args.data_path)
        if data_path.suffix == '.npy':
            X = np.load(data_path)
        elif data_path.suffix == '.csv':
            import csv
            with open(data_path, 'r') as f:
                reader = csv.reader(f)
                X = np.array([list(map(float, row)) for row in reader])
        else:
            print(f"Error: Unsupported data format: {data_path.suffix}")
            sys.exit(1)
        
        print(f"Loaded data with shape {X.shape}")
        
        # Run prediction
        if args.proba:
            predictions = model.predict_proba(X)
            print("Predicted probabilities:")
        else:
            predictions = model.predict(X)
            print("Predictions:")
        
        # Output results
        if args.output:
            output_path = Path(args.output)
            if output_path.suffix == '.npy':
                np.save(output_path, predictions)
            elif output_path.suffix == '.csv':
                np.savetxt(output_path, predictions, delimiter=',')
            else:
                np.save(output_path, predictions)
            print(f"Saved predictions to {args.output}")
        else:
            print(predictions)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def run_info(args):
    """Display model information"""
    try:
        model = LocalModel()
        model.load(args.model_path)
        
        print(f"Model: {args.model_path}")
        print(f"Type: {type(model.model).__name__}")
        
        if model.metadata:
            print("\nMetadata:")
            print(json.dumps(model.metadata, indent=2))
        
        # Try to get model attributes
        if hasattr(model.model, 'n_features_in_'):
            print(f"Input features: {model.model.n_features_in_}")
        
        if hasattr(model.model, 'classes_'):
            print(f"Classes: {model.model.classes_}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
