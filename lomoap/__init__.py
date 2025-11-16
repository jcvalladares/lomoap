"""
lomoap - Local Model Application
Run and train ML models locally
"""

__version__ = "0.1.0"

from .model import LocalModel
from .trainer import ModelTrainer

__all__ = ["LocalModel", "ModelTrainer"]
