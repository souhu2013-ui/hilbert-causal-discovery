"""
HCD Example: Basic Usage
========================

This example demonstrates how to use HCD for causal discovery.
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hcd import HilbertCausalDiscovery

def main():
    print("=" * 60)
    print("  Hilbert Causal Discovery - Basic Example")
    print("=" * 60)
    
    # Example 1: Simple X -> Y
    print("\n[Example 1] Simple causal relationship: X -> Y")
    print("-" * 40)
    
    np.random.seed(42)
    n = 500
    X = np.random.randn(n)
    Y = 0.8 * X**2 + np.random.randn(n) * 0.3  # Y = f(X) + noise
    
    data = np.column_stack([X, Y])
    
    model = HilbertCausalDiscovery(dcor_threshold=0.1)
    model.fit(data)
    
    print(f"True structure: X -> Y")
    print(f"Discovered edges: {model.get_edges()}")
    print(f"Distance correlation: {model.dcor_matrix_[0,1]:.3f}")
    
    # Example 2: Reverse direction Y -> X
    print("\n[Example 2] Reverse direction: Y -> X")
    print("-" * 40)
    
    np.random.seed(42)
    Y = np.random.randn(n)
    X = np.tanh(2 * Y) + np.random.randn(n) * 0.3  # X = f(Y) + noise
    
    data = np.column_stack([X, Y])
    model.fit(data)
    
    print(f"True structure: Y -> X")
    print(f"Discovered edges: {model.get_edges()}")
    direction = "Y -> X" if model.adj_matrix_[1, 0] > 0.5 else "X -> Y"
    print(f"HCD says: {direction}")
    
    # Example 3: Chain structure A -> B -> C
    print("\n[Example 3] Chain structure: A -> B -> C")
    print("-" * 40)
    
    np.random.seed(42)
    A = np.random.randn(n)
    B = 0.8 * A**2 + np.random.randn(n) * 0.3
    C = 0.6 * B + np.random.randn(n) * 0.3
    
    data = np.column_stack([A, B, C])
    model.fit(data)
    
    print(f"True structure: A -> B -> C")
    print(f"Discovered edges: {model.get_edges()}")
    
    # Example 4: Independence
    print("\n[Example 4] Independent variables")
    print("-" * 40)
    
    np.random.seed(42)
    X = np.random.randn(n)
    Y = np.random.randn(n)  # Independent
    
    data = np.column_stack([X, Y])
    model.fit(data)
    
    print(f"True: X and Y are independent")
    print(f"Discovered edges: {model.get_edges()}")
    print(f"Distance correlation: {model.dcor_matrix_[0,1]:.3f}")
    
    print("\n" + "=" * 60)
    print("  Examples completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
