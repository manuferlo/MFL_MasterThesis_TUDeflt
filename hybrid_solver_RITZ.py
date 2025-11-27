"""
Schur Complement CG Analysis with Ritz Value Tracking.

This script performs a detailed analysis of the hybrid solver's convergence.
It solves the multi-element problem using Conjugate Gradient (CG) on the 
Schur complement system and tracks the evolution of Ritz values
to explain the convergence behaviour.

Author: Manuel Fernandez Lopez
Master Thesis - TU Delft
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import eigh_tridiagonal
import time
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mticker

# Custom Modules
import bf_polynomials
import multigrid_utilities as mg_utils
from hybrid_solver import (
    build_local_element, 
    build_connectivity_matrix, 
    build_rhs_local_vectorized, 
    setup_p_multigrid_hierarchy,
    apply_A_inverse,
    apply_schur_op,
    forcing_f
)

# --- PLOTTING STYLE ---
try:
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = ['Computer Modern Roman']
except:
    print("Warning: LaTeX not found. Using standard fonts.")

# ==============================================================================
# 1. CG SOLVER (WITH RITZ ANALYSIS)
# ==============================================================================

def cg_with_ritz_analysis(S_op, b, tol=1e-10, maxiter=50):
    """
    Instrumented Conjugate Gradient solver that computes and stores 
    Ritz values (approximate eigenvalues of S) at every iteration.
    
    Args:
        S_op (LinearOperator): The system matrix/operator S.
        b (np.ndarray): RHS vector.
        tol (float): Relative tolerance.
        maxiter (int): Maximum iterations.
        
    Returns:
        tuple: (solution_x, ritz_history, residual_history)
    """
    x = np.zeros_like(b)
    r = b - S_op.matvec(x)
    p = r.copy()
    
    rs_old = np.dot(r, r)
    initial_res_norm = np.sqrt(rs_old)
    if initial_res_norm == 0: initial_res_norm = 1.0

    alphas, betas = [], []
    ritz_history = []
    residual_norms_relative = [1.0]

    print("--- Starting CG with Ritz Analysis ---")
    
    for k in range(maxiter):
        Sp = S_op.matvec(p)
        denom = np.dot(p, Sp)
        
        if denom <= 0:
            print("Warning: Indefinite or singular operator encountered.")
            break
            
        alpha = rs_old / denom
        x += alpha * p
        r -= alpha * Sp
        
        rs_new = np.dot(r, r)
        
        # --- Ritz Value Calculation (Lanczos Tridiagonal Matrix T) ---
        alphas.append(alpha)
        beta = rs_new / rs_old
        betas.append(beta) # Beta_k is needed for next step T_{k+1}
        
        # Construct Tridiagonal Matrix T_k
        # Diagonal: 1/alpha_i + beta_{i-1}/alpha_{i-1}
        # Off-diagonal: sqrt(beta_i) / alpha_i
        diag = np.zeros(k + 1)
        off_diag = np.zeros(k)
        
        for i in range(k + 1):
            if i == 0: 
                diag[i] = 1.0 / alphas[i]
            else: 
                diag[i] = (1.0 / alphas[i]) + (betas[i-1] / alphas[i-1])
                
        for i in range(k):
            off_diag[i] = np.sqrt(betas[i]) / alphas[i]
            
        # Compute eigenvalues of T_k (Ritz values of S)
        if k > 0: 
            ritz_values = eigh_tridiagonal(diag, off_diag, eigvals_only=True)
            ritz_history.append(ritz_values)
        
        # --- Convergence Check ---
        current_res_norm = np.sqrt(rs_new) / initial_res_norm
        residual_norms_relative.append(current_res_norm)
        
        print(f"   CG Iter {k+1:03d}: Rel Res = {current_res_norm:.4e}")

        if current_res_norm < tol:
            print(f"-> Converged in {k+1} iterations.")
            break
        
        p = r + beta * p
        rs_old = rs_new
        
    return x, ritz_history, residual_norms_relative

# ==============================================================================
# 2. PLOTTING FUNCTION
# ==============================================================================

def plot_ritz_convergence(ritz_history, residuals, p, Kx, Ky, output_dir):
    """
    Plots Ritz value evolution and Residual convergence side-by-side.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)
    fig.suptitle(rf'\textbf{{CG Analysis: Ritz Values \& Convergence}} ($p={p}$, Grid ${Kx}\times{Ky}$)', fontsize=14)

    # --- Plot 1: Ritz Values ---
    for k, ritz_vals in enumerate(ritz_history):
        # Iteration index k+1 (since k starts at 0)
        iter_idx = np.full_like(ritz_vals, k + 1)
        ax1.plot(iter_idx, ritz_vals, 
                 ls='None', marker='_', color='darkred', markersize=5, alpha=0.8)
                 
    ax1.set_title(r'\textbf{Ritz Value Evolution}', fontsize=12)
    ax1.set_xlabel('Iteration ($k$)')
    ax1.set_ylabel(r'Eigenvalues $\theta_k$')
    ax1.grid(True, ls='--', alpha=0.5)
    
    # --- Plot 2: Residuals ---
    iters = np.arange(len(residuals))
    ax2.semilogy(iters, residuals, 
                 ls='-', marker='o', markersize=4, color='navy', mfc='white')
                 
    ax2.set_title(r'\textbf{Residual Convergence}', fontsize=12)
    ax2.set_xlabel('Iteration ($k$)')
    ax2.set_ylabel(r'$\|r_k\| / \|b\|$')
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    
    plt.tight_layout()
    fname = os.path.join(output_dir, f"cg_ritz_p{p}_{Kx}x{Ky}.png")
    plt.savefig(fname)
    print(f"Plot saved to: {fname}")
    plt.close()

# ==============================================================================
# 3. MAIN
# ==============================================================================

if __name__ == "__main__":
    
    # --- Configuration ---
    P_ORDER = 8
    Kx, Ky = 4, 4  # Smaller grid for detailed spectral analysis
    QUAD_DEG = 32
    
    # p-Multigrid Setup
    P_LEVELS = [4, 8]
    SOLVER_PARAMS = {
        'cycle_type': 'V', 'tol': 1e-14, 'max_iter': 20, 
        'pre_steps': 8, 'post_steps': 8
    }
    
    PLOTS_DIR = 'hybrid_analysis_plots'
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --- 1. Hierarchy & Assembly ---
    print(f"--- Setting up Analysis for p={P_ORDER}, Grid={Kx}x{Ky} ---")
    
    p_mg_setup = setup_p_multigrid_hierarchy(P_LEVELS, QUAD_DEG)
    
    # Update smoothing parameters
    for lvl in p_mg_setup.values():
        lvl['pre_steps'] = SOLVER_PARAMS['pre_steps']
        lvl['post_steps'] = SOLVER_PARAMS['post_steps']

    # Dimensions
    dofs_u = 2 * P_ORDER * (P_ORDER + 1)
    dofs_p = P_ORDER * P_ORDER
    loc_size = dofs_u + dofs_p
    total_dofs = Kx * Ky * loc_size
    
    # Build RHS (Vectorized)
    print("-> Assembling Global RHS...")
    X_LIM, Y_LIM = (-2.0, 0.0), (0.0, 2.0)
    elem_w = (X_LIM[1] - X_LIM[0]) / Kx
    elem_h = (Y_LIM[1] - Y_LIM[0]) / Ky
    
    F_global = np.zeros(total_dofs)
    for kx in range(Kx):
        for ky in range(Ky):
            k = kx * Ky + ky
            x0 = X_LIM[0] + kx * elem_w
            y0 = Y_LIM[0] + ky * elem_h
            
            F_loc = build_rhs_local_vectorized(P_ORDER, QUAD_DEG, forcing_f, 
                                               (x0, x0+elem_w), (y0, y0+elem_h))
            offset = k * loc_size
            F_global[offset + dofs_u : offset + loc_size] = F_loc

    # Connectivity
    C_sparse = build_connectivity_matrix(P_ORDER, Kx, Ky)
    num_lambda = C_sparse.shape[0]

    # --- 2. Solve with Ritz Analysis ---
    print(f"-> Starting CG Solver (Size {num_lambda})...")
    
    # Compute Effective RHS for Schur: b = C * A^-1 * F
    A_inv_F = apply_A_inverse(F_global, None, loc_size, p_mg_setup, SOLVER_PARAMS)
    b_schur = C_sparse @ A_inv_F
    
    # Define Linear Operator
    S_op = LinearOperator(
        (num_lambda, num_lambda),
        matvec=lambda v: apply_schur_op(v, C_sparse, apply_A_inverse, None, 
                                        loc_size, p_mg_setup, SOLVER_PARAMS)
    )
    
    # Run CG
    lambda_sol, ritz_vals, res_history = cg_with_ritz_analysis(
        S_op, b_schur, tol=1e-13, maxiter=200
    )
    
    # --- 3. Plotting ---
    plot_ritz_convergence(ritz_vals, res_history, P_ORDER, Kx, Ky, PLOTS_DIR)