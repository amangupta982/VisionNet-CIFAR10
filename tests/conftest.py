"""
tests/conftest.py
Shared pytest configuration and fixtures.
"""
import os
import warnings

# Suppress TF and NumPy version warnings during testing
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
