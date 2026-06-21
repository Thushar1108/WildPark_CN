"""
src/__init__.py
---------------
Makes `src` a Python package and exposes the public pipeline API.
"""

from src.data_loader import load_data
from src.preprocessing import preprocess
from src.feature_engineering import engineer_features

__all__ = ["load_data", "preprocess", "engineer_features"]
