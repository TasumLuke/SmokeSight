"""
ACRF-QIU Main Estimator

Scikit-learn compatible estimator implementing the complete ACRF-QIU pipeline.
"""

import numpy as np
import warnings
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split

from .causal_discovery import CausalDiscovery
from .quantum_encoder import QuantumEncoder
from .causal_forest import CausalRandomForest
from .conformal import ConformalPredictor
from .utils import adaptive_hyperparameters


class ACRFQIUClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptive Causal Random Forest with Quantum-Inspired Uncertainty Quantification
    
    Implements the complete ACRF-QIU pipeline:
    1. Causal Discovery (PC Algorithm)
    2. Quantum-Inspired Feature Encoding
    3. Causal Random Forest Training
    4. Conformal Prediction Calibration
    
    Parameters
    ----------
    n_trees : int, default=100
        Number of trees in the random forest ensemble.
        If None, uses adaptive formula based on data dimensions.
        
    max_depth : int, default=None
        Maximum depth of decision trees.
        If None, uses adaptive formula.
        
    min_samples_leaf : int, default=None
        Minimum number of samples required at a leaf node.
        If None, uses adaptive formula.
        
    causal_alpha : float, default=0.05
        Significance level for conditional independence tests in causal discovery.
        Lower values require stronger evidence for edges.
        
    quantum_dim : int, default=10
        Dimension of quantum Hilbert space for feature encoding.
        Higher values capture more distributional detail but increase computation.
        
    gamma : float, default=0.5
        Causal bonus parameter for split criterion (Equation 15).
        Controls influence of causal importance on feature selection.
        
    eta : float, default=0.5
        Causal alignment weight parameter (Equation 19).
        Controls tree weighting based on causal structure.
        
    conformal_alpha : float, default=0.1
        Miscoverage level for conformal prediction (1 - coverage).
        Default 0.1 provides 90% coverage guarantee.
        
    calibration_fraction : float, default=0.2
        Fraction of training data used for conformal calibration.
        
    max_conditioning_size : int, default=3
        Maximum size of conditioning sets in PC algorithm.
        Limits computational complexity of causal discovery.
        
    n_jobs : int, default=-1
        Number of parallel jobs for tree training.
        -1 uses all available processors.
        
    random_state : int, RandomState instance or None, default=None
        Controls random number generation for reproducibility.
        
    verbose : bool, default=True
        Whether to print progress messages during training.
        
    Attributes
    ----------
    causal_graph_ : CausalGraph
        Learned causal graph structure with adjacency matrix and edge weights.
        
    quantum_encoder_ : QuantumEncoder
        Fitted quantum encoder with amplitude representations and entanglement matrix.
        
    forest_ : CausalRandomForest
        Trained random forest ensemble with causal weights.
        
    conformal_ : ConformalPredictor
        Calibrated conformal predictor for uncertainty quantification.
        
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
        
    n_features_in_ : int
        Number of features seen during fit.
        
    feature_importances_ : ndarray of shape (n_features,)
        Causal feature importance scores.
        
    Examples
    --------
    >>> from acrfqiu import ACRFQIUClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=500, n_features=20, n_classes=4)
    >>> clf = ACRFQIUClassifier(n_trees=100, random_state=42)
    >>> clf.fit(X, y)
    >>> y_pred = clf.predict(X)
    >>> y_pred, conf, sets = clf.predict_with_uncertainty(X)
    
    References
    ----------
    Lego, L.R. & Baptiste, D.J. (2024). ACRF-QIU: Adaptive Causal Random Forest
    with Quantum-Inspired Uncertainty Quantification for Multi-Class Prediction
    in High-Dimensional Data.
    """
    
    def __init__(
        self,
        n_trees=100,
        max_depth=None,
        min_samples_leaf=None,
        causal_alpha=0.05,
        quantum_dim=10,
        gamma=0.5,
        eta=0.5,
        conformal_alpha=0.1,
        calibration_fraction=0.2,
        max_conditioning_size=3,
        n_jobs=-1,
        random_state=None,
        verbose=True
    ):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.causal_alpha = causal_alpha
        self.quantum_dim = quantum_dim
        self.gamma = gamma
        self.eta = eta
        self.conformal_alpha = conformal_alpha
        self.calibration_fraction = calibration_fraction
        self.max_conditioning_size = max_conditioning_size
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        
    def fit(self, X, y):
        """
        Fit the ACRF-QIU model.
        
        Executes the complete 4-phase pipeline:
        1. Causal discovery to learn graph structure
        2. Quantum encoding of features
        3. Causal random forest training
        4. Conformal predictor calibration
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training feature matrix.
            
        y : array-like of shape (n_samples,)
            Target class labels.
            
        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate input
        X, y = check_X_y(X, y, dtype=np.float64, force_all_finite=True)
        self.classes_ = unique_labels(y)
        self.n_features_in_ = X.shape[1]
        
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        if self.verbose:
            self._print_header()
            print(f"Dataset: {n_samples} samples × {n_features} features × {n_classes} classes")
        
        # Adaptive hyperparameters if not specified
        if self.n_trees is None or self.max_depth is None or self.min_samples_leaf is None:
            n_trees_auto, max_depth_auto, min_leaf_auto = adaptive_hyperparameters(
                n_samples, n_features, n_classes
            )
            if self.n_trees is None:
                self.n_trees = n_trees_auto
            if self.max_depth is None:
                self.max_depth = max_depth_auto
            if self.min_samples_leaf is None:
                self.min_samples_leaf = min_leaf_auto
                
            if self.verbose:
                print(f"\nAdaptive hyperparameters:")
                print(f"  n_trees: {self.n_trees}")
                print(f"  max_depth: {self.max_depth}")
                print(f"  min_samples_leaf: {self.min_samples_leaf}")
        
        # Split data for training and calibration
        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y,
            test_size=self.calibration_fraction,
            random_state=self.random_state,
            stratify=y
        )
        
        # ========================================
        # PHASE 1: CAUSAL DISCOVERY
        # ========================================
        if self.verbose:
            print("\n" + "="*70)
            print("PHASE 1: Causal Discovery (PC Algorithm)")
            print("="*70)
        
        self.causal_discovery_ = CausalDiscovery(
            alpha=self.causal_alpha,
            max_conditioning_size=self.max_conditioning_size,
            verbose=self.verbose
        )
        self.causal_graph_ = self.causal_discovery_.fit(X_train, y_train)
        
        # ========================================
        # PHASE 2: QUANTUM ENCODING
        # ========================================
        if self.verbose:
            print("\n" + "="*70)
            print("PHASE 2: Quantum-Inspired Feature Encoding")
            print("="*70)
        
        self.quantum_encoder_ = QuantumEncoder(
            n_dims=self.quantum_dim,
            verbose=self.verbose
        )
        self.quantum_encoder_.fit(X_train)
        
        # ========================================
        # PHASE 3: CAUSAL RANDOM FOREST
        # ========================================
        if self.verbose:
            print("\n" + "="*70)
            print("PHASE 3: Causal Random Forest Training")
            print("="*70)
        
        self.forest_ = CausalRandomForest(
            n_trees=self.n_trees,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            gamma=self.gamma,
            eta=self.eta,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )
        self.forest_.fit(X_train, y_train, self.causal_graph_)
        
        # ========================================
        # PHASE 4: CONFORMAL CALIBRATION
        # ========================================
        if self.verbose:
            print("\n" + "="*70)
            print("PHASE 4: Conformal Prediction Calibration")
            print("="*70)
        
        self.conformal_ = ConformalPredictor(
            model=self.forest_,
            alpha=self.conformal_alpha,
            verbose=self.verbose
        )
        self.conformal_.calibrate(X_cal, y_cal)
        
        # Store feature importances
        self.feature_importances_ = self.causal_graph_.causal_importance
        
        if self.verbose:
            self._print_footer()
        
        return self
    
    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
            
        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64
