"""
Offline Pre-computation for p-Multigrid Solver.

This script performs the initialization phase for the high-order p-Multigrid solver.
It generates, assembles, and caches the necessary algebraic structures for a 
sequence of polynomial degrees (p-levels).

Key operations performed and cached:
1. System Assembly: Construction of Mass (M) and Incidence (E) matrices for MSEM.
2. Smoother Setup: Required for the Additive Schwarz Smoother (local subdomain solves).
3. RHS Integration: High-order quadrature integration of the dual 
   forcing function vector.

The output is a set of binary '.npy' files in the 'mg_level_data' directory. 

Author: Manuel Fernandez Lopez
Master Thesis - TU Delft
"""


import numpy as np
import os
import bf_polynomials
import multigrid_utilities as mg_utils

# ---------------------------------------------------------
# Configuration Constants
# ---------------------------------------------------------
# Define the physical domain here to ensure consistency across levels
X_RANGE = (-2.0, 0.0)
Y_RANGE = (0.0, 2.0)

def forcing_f(x, y):
    """
    The analytical forcing function f(x,y).
    Must match the problem defined in the thesis (Poisson equation).
    """
    return -2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)

def generate_and_save_level_data(N, Nq, output_dir="mg_level_data"):
    """
    Computes and saves system matrices, smoother data, and RHS for a specific polynomial order N.
    
    This function handles the computationally expensive setup phase of the p-Multigrid method.
    Data is saved as a .npy dictionary for fast loading during runtime.

    Args:
        N (int): Polynomial degree for this level.
        Nq (int): Quadrature degree for integration.
        output_dir (str): Directory to save the .npy files.
    """
    filename = os.path.join(output_dir, f"level_data_N{N:02d}.npy")
    
    print(f"--- Pre-computing data for Level N={N} ---")
    
    # 1. Build System Matrices (Mass, Stiffness, Divergence)
    print(f"    -> Assembling system matrices (MSEM)...")
    blk = mg_utils.build_mimetic_block(N, Nq, N)
    
    # Extract matrices: A=Mass(1-form), B=Div(2-form)
    A_k, B_k = blk['M1'], blk['E21']
    n_k, m_k = B_k.shape[1], B_k.shape[0]
    
    # Assemble Global Saddle Point System K
    K_k = np.block([[A_k, B_k.T], [B_k, np.zeros((m_k, m_k))]])
    
    # 2. Setup Smoother (Additive Schwarz)
    print(f"    -> Configuring Additive Schwarz Smoother...")
    smoother_data = mg_utils.setup_schwarz_smoother(A_k, B_k)
    
    # 3. Compute Right-Hand Side (RHS)
    print(f"    -> Integrating Dual RHS vector...")
    nodes, _ = bf_polynomials.lobatto_quad(N)
    
    F_vec = mg_utils.build_rhs_dual(
        N, Nq, forcing_f, nodes, 
        x_range=X_RANGE, y_range=Y_RANGE
    )
    
    # Assemble global RHS [0, g]
    f_vec = np.zeros((n_k, 1))
    g_vec = F_vec.reshape(m_k, 1)
    rhs_k = np.vstack([f_vec, g_vec])
    
    # 4. Package Data
    data_for_level = {
        'N': N,
        'A': A_k,
        'B': B_k,
        'K': K_k,
        'n': n_k,
        'm': m_k,
        'smoother_data': smoother_data,
        'rhs': rhs_k,
        # Default smoothing steps (can be overridden at runtime)
        'pre_steps': 4,  
        'post_steps': 4
    }
    
    # 5. Save to Disk
    np.save(filename, data_for_level)
    print(f"PASS: Data for N={N} saved successfully to '{filename}'")

if __name__ == '__main__':
    # ---------------------------------------------------------
    # Execution Setup
    # ---------------------------------------------------------
    
    # List of polynomial degrees to pre-compute.
    # Include all levels required for the Multigrid hierarchy.
    # e.g., for a 3-level MG with p=1,2,4, include all three.
    LEVELS_TO_COMPUTE = [7] 
    QUADRATURE_DEGREE = 64  
    
    OUTPUT_DIR = "mg_level_data"
    
    # Create directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Target directory: {os.path.abspath(OUTPUT_DIR)}")

    # Run Batch
    print("Starting Batch Pre-computation...")
    for n_val in LEVELS_TO_COMPUTE:
        try:
            generate_and_save_level_data(n_val, QUADRATURE_DEGREE, OUTPUT_DIR)
        except Exception as e:
            print(f"ERROR: Failed to compute level N={n_val}")
            print(f"Reason: {e}")
        
    print("\n--- Pre-computation Batch Completed ---")