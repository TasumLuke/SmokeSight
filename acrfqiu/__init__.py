"""
ACRF-QIU: Adaptive Causal Random Forest with Quantum-Inspired 
Uncertainty Quantification

Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Luke Rimmo Lego, Denver Jn. Baptiste"
__email__ = "djnbaptiste@stevens.edu"

from .estimator import ACRFQIUClassifier
from .causal_discovery import CausalDiscovery, CausalGraph
from .quantum_encoder import QuantumEncoder
from .causal_forest import CausalRandomForest
from .conformal import ConformalPredictor
from .utils import evaluate_model, load_model, save_model

__all__ = [
    "ACRFQIUClassifier",
    "CausalDiscovery",
    "CausalGraph",
    "QuantumEncoder",
    "CausalRandomForest",
    "ConformalPredictor",
    "evaluate_model",
    "load_model",
    "save_model",
]
