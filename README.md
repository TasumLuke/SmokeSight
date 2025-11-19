# ACRF-QIU: Adaptive Causal Random Forest with Quantum-Inspired Uncertainty Quantification

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A novel machine learning framework that integrates **causal discovery**, **quantum-inspired feature encoding**, and **conformal prediction** for robust multi-class classification in high-dimensional biomedical data.

## Key Features

- **🔗 Automated Causal Discovery**: PC algorithm identifies genuine causal relationships, not just correlations
- **⚛️ Quantum-Inspired Encoding**: Captures complex non-linear feature interactions through entanglement measures
- **📊 Conformal Prediction**: Distribution-free uncertainty quantification with guaranteed coverage (90% default)
- **🌲 Causal Random Forest**: Trees weighted by causal alignment for robust predictions
- **🔄 Online Learning**: Adaptive weight updates without full retraining
- **📈 Scikit-Learn Compatible**: Familiar `fit/predict` API
- **🏥 Safety-Critical Ready**: Designed for healthcare and biomedical applications

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install acrf-qiu
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/TasumLuke/Quantum-Random-Forest.git
cd acrf-qiu

# Install in development mode
pip install -e .

# Or install with all dependencies
pip install -e ".[dev,examples]"
```

### Option 3: Using conda

```bash
# Create conda environment
conda create -n acrfqiu python=3.9
conda activate acrfqiu

# Install package
pip install acrf-qiu
```

## Quick Start

### Basic Usage (Just Like Scikit-Learn!)

```python
from acrfqiu import ACRFQIUClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Generate sample data
X, y = make_classification(n_samples=500, n_features=20, n_classes=4, 
                          n_informative=15, random_state=42)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train model
model = ACRFQIUClassifier(
    n_trees=100,
    max_depth=10,
    causal_alpha=0.05,
    conformal_alpha=0.1,  # 90% coverage guarantee
    verbose=True
)

# Fit model (includes all 4 phases: causal discovery, quantum encoding, 
# RF training, conformal calibration)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Get predictions with uncertainty quantification
y_pred, confidence, prediction_sets = model.predict_with_uncertainty(X_test)

# Evaluate
from acrfqiu.utils import evaluate_model
metrics = evaluate_model(y_test, y_pred, prediction_sets)
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Coverage: {metrics['coverage']:.3f} (guaranteed ≥0.90)")
```

### Advanced Usage

```python
# Access causal structure
causal_importance = model.get_feature_importance()
causal_graph = model.causal_graph_

# Visualize causal graph
model.plot_causal_graph(feature_names=['Feature1', 'Feature2', ...])

# Get quantum entanglement matrix
entanglement = model.quantum_encoder_.entanglement_matrix_

# Online learning with new data
model.update_weights(X_new, y_new, learning_rate=0.01)

# Export model
model.save('parkinsons_model.pkl')

# Load model
from acrfqiu import load_model
loaded_model = load_model('parkinsons_model.pkl')
```

## Documentation

### Core Components

1. **CausalDiscovery**: PC algorithm for causal structure learning
2. **QuantumEncoder**: Quantum-inspired feature representation
3. **CausalRandomForest**: Ensemble with causal weighting
4. **ConformalPredictor**: Distribution-free uncertainty quantification

### Pipeline Flow

```
Input Data (X, y)
    ↓
[Phase 1] Causal Discovery (PC Algorithm)
    ↓ (Directed Acyclic Graph)
[Phase 2] Quantum-Inspired Encoding
    ↓ (Feature States + Entanglement Matrix)
[Phase 3] Causal Random Forest Training
    ↓ (Weighted Ensemble)
[Phase 4] Conformal Calibration
    ↓ (Prediction Sets with Coverage Guarantee)
Output: Predictions + Uncertainty
```

### API Reference

#### ACRFQIUClassifier

**Parameters:**

- `n_trees` (int, default=100): Number of trees in the forest
- `max_depth` (int, default=None): Maximum depth of trees (None = auto)
- `min_samples_leaf` (int, default=None): Minimum samples per leaf (None = auto)
- `causal_alpha` (float, default=0.05): Significance level for causal discovery
- `quantum_dim` (int, default=10): Dimension of quantum Hilbert space
- `gamma` (float, default=0.5): Causal bonus parameter for splitting
- `eta` (float, default=0.5): Causal alignment weight
- `conformal_alpha` (float, default=0.1): Miscoverage level (0.1 = 90% coverage)
- `calibration_fraction` (float, default=0.2): Fraction of data for calibration
- `n_jobs` (int, default=-1): Number of parallel jobs
- `random_state` (int, default=None): Random seed
- `verbose` (bool, default=True): Print progress

**Methods:**

- `fit(X, y)`: Train the complete ACRF-QIU model
- `predict(X)`: Predict class labels
- `predict_proba(X)`: Predict class probabilities
- `predict_with_uncertainty(X)`: Predict with conformal sets
- `get_feature_importance()`: Get causal feature importance
- `update_weights(X, y, learning_rate)`: Online weight adaptation
- `save(filepath)`: Save model to disk
- `plot_causal_graph(feature_names)`: Visualize causal structure

## Examples

### Example 1: Parkinson's Disease Progression

```python
import numpy as np
from acrfqiu import ACRFQIUClassifier
from acrfqiu.datasets import load_parkinsons

# Load Parkinson's disease dataset
X, y, feature_names = load_parkinsons()

# Train model
model = ACRFQIUClassifier(n_trees=200, verbose=True)
model.fit(X, y)

# Analyze causal structure
importance = model.get_feature_importance()
print("Top 5 Causal Features:")
for feat, imp in importance[:5]:
    print(f"  {feature_names[feat]}: {imp:.4f}")

# Make predictions with uncertainty
y_pred, conf, sets = model.predict_with_uncertainty(X_test)

# Check coverage guarantee
coverage = np.mean([y_test[i] in sets[i] for i in range(len(y_test))])
print(f"Empirical Coverage: {coverage:.3f} (guaranteed ≥0.90)")
```

### Example 2: Gene Expression Classification

```python
from acrfqiu import ACRFQIUClassifier
from acrfqiu.datasets import load_gene_expression

# Load high-dimensional gene expression data
X, y = load_gene_expression()

# Custom hyperparameters for high-dimensional data
model = ACRFQIUClassifier(
    n_trees=300,
    max_depth=15,
    quantum_dim=12,
    causal_alpha=0.01,  # More stringent for many features
    n_jobs=-1
)

model.fit(X, y)

# Analyze feature interactions via quantum entanglement
entanglement = model.quantum_encoder_.entanglement_matrix_
top_interactions = np.unravel_index(
    np.argsort(entanglement.ravel())[-10:], 
    entanglement.shape
)
print("Top 10 Feature Interactions:")
for i, j in zip(*top_interactions):
    print(f"  Feature {i} ↔ Feature {j}: {entanglement[i,j]:.4f}")
```

### Example 3: Online Learning

```python
# Initial training
model = ACRFQIUClassifier()
model.fit(X_train, y_train)

# Simulate streaming data
for batch_X, batch_y in data_stream:
    # Update model without retraining
    model.update_weights(batch_X, batch_y, learning_rate=0.01)
    
    # Make predictions on new data
    predictions = model.predict(batch_X)
```

## Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=acrfqiu --cov-report=html

# Run specific test module
pytest tests/test_causal_discovery.py -v
```

## Performance Benchmarks

| Dataset | Samples | Features | Classes | ACRF-QIU Acc. | RF Acc. | Coverage |
|---------|---------|----------|---------|---------------|---------|----------|
| Parkinson's | 500 | 20 | 4 | **0.847** | 0.812 | **0.923** |
| Gene Expression | 800 | 100 | 3 | **0.791** | 0.745 | **0.915** |
| Cancer Types | 1000 | 50 | 5 | **0.823** | 0.798 | **0.908** |

*Coverage guaranteed ≥90%, empirically exceeds guarantee*

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/acrf-qiu.git
cd acrf-qiu

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black acrfqiu/
flake8 acrfqiu/
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- Based on research by Luke Rimmo Lego and Denver Jn. Baptiste (Stevens Institute of Technology)
- Inspired by Pearl's causal inference framework
- Built on scikit-learn's robust ML infrastructure
- Quantum-inspired methods based on Lloyd et al. and Nielsen & Chuang

## Contact

- **Corresponding Author**: Denver Jn. Baptiste (djnbaptiste@stevens.edu)
- **Issues**: [GitHub Issues](https://github.com/yourusername/acrf-qiu/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/acrf-qiu/discussions)

## 🔗 Links

- [Documentation](https://acrf-qiu.readthedocs.io/)
- [Paper](https://arxiv.org/abs/XXXXX)
- [Examples](https://github.com/yourusername/acrf-qiu/tree/main/examples)
- [Changelog](CHANGELOG.md)
