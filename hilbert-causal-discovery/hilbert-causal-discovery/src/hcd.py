"""
Hilbert Causal Discovery (HCD)
==============================

A causal discovery algorithm combining distance correlation for skeleton 
learning with MDL-based direction identification.

Author: Shiqian Zhu (souhu2013@gmail.com)
License: MIT
Version: 1.0.0

Usage:
    from hcd import HilbertCausalDiscovery
    
    model = HilbertCausalDiscovery(dcor_threshold=0.1)
    model.fit(data)  # data: (n_samples, n_variables)
    
    adj_matrix = model.get_adjacency_matrix()  # adj[i,j]=1 means i->j
    edges = model.get_edges()  # List of (i, j) tuples

Reference:
    Zhu, S. (2024). Hilbert Causal Discovery: A Distance Correlation Approach
    with MDL-based Direction Identification.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from typing import Tuple, Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')


class HilbertCausalDiscovery:
    """
    Hilbert Causal Discovery Algorithm
    
    Combines distance correlation for skeleton learning with 
    MDL (Minimum Description Length) for direction identification.
    
    Parameters
    ----------
    dcor_threshold : float, default=0.1
        Threshold for distance correlation. Edges with dCor below 
        this threshold are considered independent.
        
    ci_threshold_ratio : float, default=0.5
        Ratio for conditional independence threshold (relative to dcor_threshold).
        Used in Stage 2 to remove spurious edges.
        
    max_degree : int, default=4
        Maximum polynomial degree for MDL fitting.
        
    n_subsample : int, default=500
        Number of samples for subsampling in distance correlation computation.
        Used to speed up computation for large datasets.
    
    Attributes
    ----------
    adj_matrix_ : ndarray of shape (n_vars, n_vars)
        Adjacency matrix after fitting. adj_matrix_[i,j] = 1 means i -> j.
        
    skeleton_ : ndarray of shape (n_vars, n_vars)
        Undirected skeleton before orientation.
        
    dcor_matrix_ : ndarray of shape (n_vars, n_vars)
        Pairwise distance correlation matrix.
        
    action_ : dict
        Dictionary containing action (loss) components.
    
    Examples
    --------
    >>> import numpy as np
    >>> from hcd import HilbertCausalDiscovery
    >>> 
    >>> # Generate causal data: X -> Y
    >>> np.random.seed(42)
    >>> X = np.random.randn(500)
    >>> Y = 0.8 * X**2 + np.random.randn(500) * 0.3
    >>> data = np.column_stack([X, Y])
    >>> 
    >>> # Discover causal structure
    >>> model = HilbertCausalDiscovery()
    >>> model.fit(data)
    >>> print(model.get_edges())  # [(0, 1)] meaning X -> Y
    
    Notes
    -----
    The algorithm proceeds in three stages:
    
    1. **Skeleton Learning**: Use distance correlation to identify edges.
       dCor(X_i, X_j) > threshold implies edge exists.
       
    2. **Conditional Independence Testing**: Remove spurious edges using
       partial distance correlation. If dCor(X_i, X_j | X_k) < threshold,
       remove edge i-j.
       
    3. **Direction Identification**: Use MDL (BIC) to determine edge
       orientation. If MDL(i->j) < MDL(j->i), orient as i -> j.
    
    Limitations
    -----------
    - Assumes no hidden confounders (causal sufficiency)
    - May show bias on linear Gaussian data
    - Performance degrades for >5 variables
    - Not suitable for discrete/categorical causes
    """
    
    def __init__(self, 
                 dcor_threshold: float = 0.1,
                 ci_threshold_ratio: float = 0.5,
                 max_degree: int = 4,
                 n_subsample: int = 500):
        self.dcor_threshold = dcor_threshold
        self.ci_threshold_ratio = ci_threshold_ratio
        self.max_degree = max_degree
        self.n_subsample = n_subsample
        
        # Results storage
        self.adj_matrix_ = None
        self.skeleton_ = None
        self.dcor_matrix_ = None
        self.action_ = None
    
    def _distance_correlation(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Compute distance correlation between X and Y.
        
        Distance correlation measures both linear and nonlinear dependence.
        dCor(X,Y) = 0 if and only if X and Y are independent.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_features)
        Y : array-like of shape (n_samples,) or (n_samples, n_features)
        
        Returns
        -------
        dcor : float
            Distance correlation, between 0 and 1.
        """
        # Subsample for efficiency
        n = len(X)
        if n > self.n_subsample:
            idx = np.random.choice(n, self.n_subsample, replace=False)
            X, Y = X[idx], Y[idx]
            n = self.n_subsample
        
        # Ensure 2D
        X = X.reshape(-1, 1) if len(X.shape) == 1 else X
        Y = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y
        
        # Compute distance matrices
        a = squareform(pdist(X))
        b = squareform(pdist(Y))
        
        # Double centering (key step for RKHS embedding)
        A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
        B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
        
        # Distance covariance and variance
        dcov2 = (A * B).sum() / (n * n)
        dvarX = (A * A).sum() / (n * n)
        dvarY = (B * B).sum() / (n * n)
        
        # Distance correlation
        if dvarX * dvarY > 0:
            dcor = np.sqrt(max(0, dcov2)) / np.sqrt(np.sqrt(dvarX) * np.sqrt(dvarY))
        else:
            dcor = 0.0
        
        return dcor
    
    def _partial_distance_correlation(self, X: np.ndarray, Y: np.ndarray, 
                                       Z: np.ndarray) -> float:
        """
        Compute conditional distance correlation dCor(X, Y | Z).
        
        Uses residual-based approach: regress X and Y on Z, then compute
        distance correlation of residuals.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
        Y : array-like of shape (n_samples,)
        Z : array-like of shape (n_samples,) or (n_samples, n_features)
        
        Returns
        -------
        pdcor : float
            Partial distance correlation.
        """
        if Z is None:
            return self._distance_correlation(X, Y)
        
        if len(Z.shape) == 1:
            Z = Z.reshape(-1, 1)
        
        try:
            # Nonlinear features to capture conditional dependencies
            Z_features = np.column_stack([Z, Z**2])
            Z_aug = np.column_stack([np.ones(len(Z)), Z_features])
            
            # Least squares regression
            coef_X = np.linalg.lstsq(Z_aug, X, rcond=None)[0]
            res_X = X - Z_aug @ coef_X
            
            coef_Y = np.linalg.lstsq(Z_aug, Y, rcond=None)[0]
            res_Y = Y - Z_aug @ coef_Y
            
            return self._distance_correlation(res_X, res_Y)
        except:
            return self._distance_correlation(X, Y)
    
    def _compute_mdl(self, cause: np.ndarray, effect: np.ndarray) -> Tuple[float, int]:
        """
        Compute MDL (Minimum Description Length) for causal direction.
        
        Uses BIC as MDL proxy:
            BIC = n * log(MSE) + k * log(n)
        
        Parameters
        ----------
        cause : array-like of shape (n_samples,)
            Hypothesized cause variable.
        effect : array-like of shape (n_samples,)
            Hypothesized effect variable.
            
        Returns
        -------
        bic : float
            BIC score (lower is better).
        degree : int
            Best polynomial degree.
        """
        n = len(cause)
        best_bic = np.inf
        best_degree = 1
        
        for degree in range(1, self.max_degree + 1):
            try:
                coef = np.polyfit(cause, effect, degree)
                pred = np.polyval(coef, cause)
                mse = np.mean((effect - pred)**2) + 1e-10
                bic = n * np.log(mse) + (degree + 1) * np.log(n)
                
                if bic < best_bic:
                    best_bic = bic
                    best_degree = degree
            except:
                continue
        
        return best_bic, best_degree
    
    def _learn_skeleton(self, data: np.ndarray) -> np.ndarray:
        """
        Learn causal skeleton using distance correlation.
        
        Stage 1: Marginal independence test
        Stage 2: Conditional independence test
        """
        n_vars = data.shape[1]
        
        # Stage 1: Compute distance correlation matrix
        self.dcor_matrix_ = np.zeros((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                dcor = self._distance_correlation(data[:, i], data[:, j])
                self.dcor_matrix_[i, j] = self.dcor_matrix_[j, i] = dcor
        
        # Initial skeleton
        skeleton = (self.dcor_matrix_ > self.dcor_threshold).astype(float)
        np.fill_diagonal(skeleton, 0)
        
        # Stage 2: Conditional independence testing
        ci_threshold = self.dcor_threshold * self.ci_threshold_ratio
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if skeleton[i, j] > 0:
                    # Try to d-separate using other variables
                    for k in range(n_vars):
                        if k != i and k != j:
                            pdcor = self._partial_distance_correlation(
                                data[:, i], data[:, j], data[:, k])
                            
                            if pdcor < ci_threshold:
                                skeleton[i, j] = skeleton[j, i] = 0
                                break
        
        return skeleton
    
    def _orient_edges(self, data: np.ndarray, skeleton: np.ndarray) -> np.ndarray:
        """
        Orient edges using MDL-based direction identification.
        
        For each undirected edge i-j, compare MDL(i->j) vs MDL(j->i).
        Orient as i->j if MDL(i->j) < MDL(j->i).
        """
        n_vars = data.shape[1]
        adj = np.zeros((n_vars, n_vars))
        
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if skeleton[i, j] > 0:
                    mdl_ij, _ = self._compute_mdl(data[:, i], data[:, j])
                    mdl_ji, _ = self._compute_mdl(data[:, j], data[:, i])
                    
                    if mdl_ij < mdl_ji:
                        adj[i, j] = 1  # i -> j
                    else:
                        adj[j, i] = 1  # j -> i
        
        return adj
    
    def _compute_action(self, data: np.ndarray, adj: np.ndarray) -> Dict:
        """
        Compute Hilbert causal action (loss components).
        """
        n_vars = data.shape[1]
        
        # L_CI: Orthogonality violation for non-edges
        L_CI = 0
        n_non_edges = 0
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj[i, j] < 0.5 and adj[j, i] < 0.5:
                    L_CI += self.dcor_matrix_[i, j] ** 2
                    n_non_edges += 1
        L_CI = L_CI / max(n_non_edges, 1)
        
        # L_geom: Mechanism complexity
        L_geom = 0
        n_edges = 0
        for i in range(n_vars):
            for j in range(n_vars):
                if adj[i, j] > 0.5:
                    mdl, _ = self._compute_mdl(data[:, i], data[:, j])
                    L_geom += mdl
                    n_edges += 1
        L_geom = L_geom / max(n_edges, 1)
        
        # L_mdl: Sparsity
        L_mdl = n_edges / (n_vars * (n_vars - 1))
        
        return {
            'L_CI': L_CI,
            'L_geom': L_geom,
            'L_mdl': L_mdl,
            'n_edges': n_edges
        }
    
    def fit(self, data: np.ndarray) -> 'HilbertCausalDiscovery':
        """
        Fit the model to discover causal structure.
        
        Parameters
        ----------
        data : ndarray of shape (n_samples, n_vars)
            Observational data matrix.
            
        Returns
        -------
        self : HilbertCausalDiscovery
            Fitted model.
        """
        # Learn skeleton
        self.skeleton_ = self._learn_skeleton(data)
        
        # Orient edges
        self.adj_matrix_ = self._orient_edges(data, self.skeleton_)
        
        # Compute action
        self.action_ = self._compute_action(data, self.adj_matrix_)
        
        return self
    
    def get_adjacency_matrix(self) -> np.ndarray:
        """
        Get the adjacency matrix of the discovered DAG.
        
        Returns
        -------
        adj : ndarray of shape (n_vars, n_vars)
            Adjacency matrix where adj[i,j]=1 means i->j.
        """
        return self.adj_matrix_
    
    def get_edges(self) -> List[Tuple[int, int]]:
        """
        Get list of directed edges.
        
        Returns
        -------
        edges : list of tuples
            List of (i, j) tuples representing edges i -> j.
        """
        edges = []
        n_vars = self.adj_matrix_.shape[0]
        for i in range(n_vars):
            for j in range(n_vars):
                if self.adj_matrix_[i, j] > 0.5:
                    edges.append((i, j))
        return edges
    
    def predict_direction(self, X: np.ndarray, Y: np.ndarray) -> str:
        """
        Predict causal direction between two variables.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            First variable.
        Y : array-like of shape (n_samples,)
            Second variable.
            
        Returns
        -------
        direction : str
            One of 'X->Y', 'Y->X', or 'independent'.
        """
        dcor = self._distance_correlation(X, Y)
        
        if dcor < self.dcor_threshold:
            return 'independent'
        
        mdl_xy, _ = self._compute_mdl(X, Y)
        mdl_yx, _ = self._compute_mdl(Y, X)
        
        if mdl_xy < mdl_yx:
            return 'X->Y'
        else:
            return 'Y->X'


# =============================================================================
# Utility Functions
# =============================================================================

def evaluate_dag(pred: np.ndarray, true: np.ndarray) -> Dict:
    """
    Evaluate predicted DAG against ground truth.
    
    Parameters
    ----------
    pred : ndarray
        Predicted adjacency matrix.
    true : ndarray
        True adjacency matrix.
        
    Returns
    -------
    metrics : dict
        Dictionary containing F1, precision, recall, accuracy, and SHD.
    """
    pred_bin = (pred > 0.5).astype(int)
    
    tp = np.sum((pred_bin == 1) & (true == 1))
    fp = np.sum((pred_bin == 1) & (true == 0))
    fn = np.sum((pred_bin == 0) & (true == 1))
    tn = np.sum((pred_bin == 0) & (true == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    # Structural Hamming Distance
    shd = np.sum(pred_bin != true)
    
    return {
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy,
        'shd': shd,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }


def discover_causal_structure(data: np.ndarray, 
                               dcor_threshold: float = 0.1) -> np.ndarray:
    """
    Convenience function to discover causal structure.
    
    Parameters
    ----------
    data : ndarray of shape (n_samples, n_vars)
        Observational data.
    dcor_threshold : float
        Distance correlation threshold.
        
    Returns
    -------
    adj : ndarray
        Adjacency matrix where adj[i,j]=1 means i->j.
    """
    model = HilbertCausalDiscovery(dcor_threshold=dcor_threshold)
    model.fit(data)
    return model.get_adjacency_matrix()


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Simple demonstration
    np.random.seed(42)
    n = 500
    
    # Generate X -> Y data
    X = np.random.randn(n)
    Y = 0.8 * X**2 + np.random.randn(n) * 0.3
    
    data = np.column_stack([X, Y])
    
    model = HilbertCausalDiscovery()
    model.fit(data)
    
    print("Hilbert Causal Discovery")
    print("=" * 40)
    print(f"Edges discovered: {model.get_edges()}")
    print(f"Direction: {'X->Y' if model.adj_matrix_[0,1] > 0 else 'Y->X'}")
    print(f"Distance correlation: {model.dcor_matrix_[0,1]:.3f}")
