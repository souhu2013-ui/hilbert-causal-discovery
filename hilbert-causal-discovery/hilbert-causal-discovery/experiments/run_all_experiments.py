"""
HCD Comprehensive Experiments
=============================

Run all experiments and generate figures for the paper.

Usage:
    python run_all_experiments.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.hcd import HilbertCausalDiscovery, evaluate_dag

np.random.seed(42)

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def experiment_direction_accuracy():
    """Test direction identification accuracy for various functions."""
    print_header("Experiment 1: Direction Identification Accuracy")
    
    functions = {
        'Linear': lambda x: 0.8 * x,
        'Quadratic': lambda x: 0.8 * x**2,
        'Cubic': lambda x: 0.5 * x**3,
        'Tanh': lambda x: np.tanh(2 * x),
        'Sine': lambda x: np.sin(2 * x),
        'Exp': lambda x: np.exp(0.5 * x) - 1,
    }
    
    results = {}
    
    for name, func in functions.items():
        xy_correct = 0
        yx_correct = 0
        n_trials = 50
        
        for trial in range(n_trials):
            np.random.seed(trial)
            n = 500
            
            # Test X -> Y
            X = np.random.randn(n)
            Y = func(X) + np.random.randn(n) * 0.3
            model = HilbertCausalDiscovery(dcor_threshold=0.1)
            model.fit(np.column_stack([X, Y]))
            if model.adj_matrix_[0, 1] > 0.5:
                xy_correct += 1
            
            # Test Y -> X
            Y2 = np.random.randn(n)
            X2 = func(Y2) + np.random.randn(n) * 0.3
            model.fit(np.column_stack([X2, Y2]))
            if model.adj_matrix_[1, 0] > 0.5:
                yx_correct += 1
        
        results[name] = {
            'X->Y': xy_correct / n_trials * 100,
            'Y->X': yx_correct / n_trials * 100
        }
        print(f"  {name:12s}: X→Y = {results[name]['X->Y']:.0f}%, Y→X = {results[name]['Y->X']:.0f}%")
    
    return results


def experiment_noise_sensitivity():
    """Test sensitivity to noise levels."""
    print_header("Experiment 2: Noise Sensitivity")
    
    noise_levels = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]
    results = {'noise': [], 'f1': [], 'direction_acc': []}
    
    for noise in noise_levels:
        f1_list = []
        correct_dir = 0
        n_trials = 30
        
        for trial in range(n_trials):
            np.random.seed(trial)
            n = 500
            X = np.random.randn(n)
            Y = 0.8 * X**2 + np.random.randn(n) * noise
            
            model = HilbertCausalDiscovery(dcor_threshold=0.1)
            model.fit(np.column_stack([X, Y]))
            
            # F1 score
            true_adj = np.array([[0, 1], [0, 0]])
            metrics = evaluate_dag(model.adj_matrix_, true_adj)
            f1_list.append(metrics['f1'])
            
            if model.adj_matrix_[0, 1] > 0.5:
                correct_dir += 1
        
        results['noise'].append(noise)
        results['f1'].append(np.mean(f1_list))
        results['direction_acc'].append(correct_dir / n_trials * 100)
        
        print(f"  σ = {noise:.1f}: F1 = {np.mean(f1_list):.3f}, Dir.Acc = {correct_dir/n_trials*100:.0f}%")
    
    return results


def experiment_sample_size():
    """Test effect of sample size."""
    print_header("Experiment 3: Sample Size Effect")
    
    sample_sizes = [50, 100, 200, 500, 1000, 2000, 5000]
    results = {'n': [], 'f1_mean': [], 'f1_std': []}
    
    for n_samples in sample_sizes:
        f1_list = []
        
        for trial in range(30):
            np.random.seed(trial)
            X = np.random.randn(n_samples)
            Y = 0.8 * X**2 + np.random.randn(n_samples) * 0.3
            
            model = HilbertCausalDiscovery(dcor_threshold=0.1)
            model.fit(np.column_stack([X, Y]))
            
            true_adj = np.array([[0, 1], [0, 0]])
            metrics = evaluate_dag(model.adj_matrix_, true_adj)
            f1_list.append(metrics['f1'])
        
        results['n'].append(n_samples)
        results['f1_mean'].append(np.mean(f1_list))
        results['f1_std'].append(np.std(f1_list))
        
        print(f"  n = {n_samples:5d}: F1 = {np.mean(f1_list):.3f} ± {np.std(f1_list):.3f}")
    
    return results


def experiment_limitations():
    """Test known limitations."""
    print_header("Experiment 4: Limitation Analysis")
    
    # Linear Gaussian
    print("\n  Linear Gaussian (theoretically unidentifiable):")
    xy_count = 0
    for trial in range(100):
        np.random.seed(trial)
        n = 500
        X = np.random.randn(n)
        Y = 0.8 * X + np.random.randn(n) * 0.5
        
        model = HilbertCausalDiscovery(dcor_threshold=0.1)
        model.fit(np.column_stack([X, Y]))
        
        if model.adj_matrix_[0, 1] > 0.5:
            xy_count += 1
    
    print(f"    X→Y: {xy_count}/100, Y→X: {100-xy_count}/100")
    print(f"    ⚠️ Shows 100% bias despite theoretical unidentifiability!")
    
    # Confounding
    print("\n  Hidden Confounding:")
    np.random.seed(42)
    n = 500
    U = np.random.randn(n)
    X = 0.8 * U + np.random.randn(n) * 0.3
    Y = 0.6 * U**2 + np.random.randn(n) * 0.3
    
    model = HilbertCausalDiscovery(dcor_threshold=0.1)
    model.fit(np.column_stack([X, Y]))
    
    print(f"    True: No direct X-Y edge (common cause U)")
    print(f"    HCD:  {model.get_edges()}")
    print(f"    ⚠️ Cannot detect confounding!")


def experiment_baseline_comparison():
    """Compare with baselines."""
    print_header("Experiment 5: Baseline Comparison")
    
    structures = {
        'simple': (
            lambda n: (np.column_stack([
                X := np.random.randn(n),
                0.8 * X**2 + np.random.randn(n) * 0.3
            ]), np.array([[0,1],[0,0]]))
        ),
        'chain': (
            lambda n: (np.column_stack([
                A := np.random.randn(n),
                B := 0.8 * A**2 + np.random.randn(n) * 0.3,
                0.6 * B + np.random.randn(n) * 0.3
            ]), np.array([[0,1,0],[0,0,1],[0,0,0]]))
        ),
    }
    
    print(f"\n  {'Structure':<12} {'HCD':>8} {'Random':>8}")
    print("  " + "-" * 30)
    
    for name, gen_func in structures.items():
        np.random.seed(42)
        data, true_adj = gen_func(500)
        
        # HCD
        model = HilbertCausalDiscovery(dcor_threshold=0.1)
        model.fit(data)
        hcd_metrics = evaluate_dag(model.adj_matrix_, true_adj)
        
        # Random baseline
        random_f1s = []
        for _ in range(100):
            pred = np.random.randint(0, 2, true_adj.shape)
            np.fill_diagonal(pred, 0)
            random_f1s.append(evaluate_dag(pred, true_adj)['f1'])
        
        print(f"  {name:<12} {hcd_metrics['f1']:>8.3f} {np.mean(random_f1s):>8.3f}")


def main():
    """Run all experiments."""
    print("\n" + "=" * 70)
    print("  HILBERT CAUSAL DISCOVERY - COMPREHENSIVE EXPERIMENTS")
    print("=" * 70)
    
    experiment_direction_accuracy()
    experiment_noise_sensitivity()
    experiment_sample_size()
    experiment_limitations()
    experiment_baseline_comparison()
    
    print("\n" + "=" * 70)
    print("  ALL EXPERIMENTS COMPLETED")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
