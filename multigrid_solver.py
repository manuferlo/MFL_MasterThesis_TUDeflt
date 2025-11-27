"""
p-Multigrid Solver.

This script implements the main multigrid cycle (V-Cycle or W-Cycle) for the 
Mimetic Spectral Element Method for only one element. It assembles the hierarchy of operators 
(Prolongation, Restriction, Coarse Grid Operators) and solves the system 
using the Additive Schwarz Smoother.


Author: Manuel Fernandez Lopez
Master Thesis - TU Delft
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import os
import sys

# Custom Modules
import multigrid_utilities as mg_utils
import bf_polynomials

# ===============================================================
# VISUALIZATION UTILITIES
# ===============================================================

def reconstruct_solution(u_coeffs, p_coeffs, blk):
    """
    Reconstructs the solution fields on a fine plotting grid.
    """
    num_q_plot = blk['psi1_plot'].shape[1]
    plot_deg = int(np.sqrt(num_q_plot))
    plot_nodes, _ = bf_polynomials.gauss_quad(plot_deg)
    xi, eta = np.meshgrid(plot_nodes, plot_nodes, indexing='ij')

    a, b = blk['x_range']; c, d = blk['y_range']
    x = 0.5 * (b - a) * (xi + 1) + a
    y = 0.5 * (d - c) * (eta + 1) + c

    # Reconstruction
    phi_vec = p_coeffs.T @ blk['psi0_dual_plot']
    ux_vec  = u_coeffs.T @ blk['psi1_plot'][:,:,0]
    uy_vec  = u_coeffs.T @ blk['psi1_plot'][:,:,1]

    phi_h = phi_vec.reshape(plot_deg, plot_deg)
    ux_h  = ux_vec.reshape(plot_deg, plot_deg)
    uy_h  = uy_vec.reshape(plot_deg, plot_deg)

    return x, y, phi_h, ux_h, uy_h

def triple_plot(x, y, exact, approx, title):
    """
    Generates a 3-panel comparison plot: Exact | Numerical | Error.
    """
    err = approx - exact
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    
    vmin = min(exact.min(), approx.min())
    vmax = max(exact.max(), approx.max())
    
    # Plot 1: Exact
    h1 = axes[0].pcolormesh(x, y, exact, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0].set_title('Exact Solution')
    fig.colorbar(h1, ax=axes[0])
    
    # Plot 2: Numerical
    h2 = axes[1].pcolormesh(x, y, approx, shading='gouraud', cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1].set_title('Multigrid Solution')
    fig.colorbar(h2, ax=axes[1])
    
    # Plot 3: Error
    err_max = np.max(np.abs(err))
    h3 = axes[2].pcolormesh(x, y, err, shading='gouraud', cmap='coolwarm', vmin=-err_max, vmax=err_max)
    axes[2].set_title('Error')
    fig.colorbar(h3, ax=axes[2])

    for ax in axes:
        ax.set_aspect('equal')
    plt.show()

# Analytical solutions for validation
def exact_phi(x, y): return np.sin(np.pi*x) * np.sin(np.pi*y)

# ===============================================================
# MULTIGRID CORE LOGIC
# ===============================================================

def w_cycle_recursive(level, u, p, f, g, level_data, gamma=2):
    """
    Recursive V-Cycle (gamma=1) or W-Cycle (gamma=2).
    """
    # Base Case: Exact Solve on Coarsest Grid
    if level == 0:
        K_coarse = level_data[level]['K']
        rhs_coarse = np.vstack([f, g])
        # Using a direct solver for the coarsest level
        e_coarse = la.solve(K_coarse, rhs_coarse)
        n_coarse = level_data[level]['n']
        return e_coarse[:n_coarse].reshape(n_coarse, 1), e_coarse[n_coarse:].reshape(-1, 1)

    # --- Recursive Step ---
    A, B = level_data[level]['A'], level_data[level]['B']
    smoother_data = level_data[level]['smoother_data']
    pre_steps = level_data[level]['pre_steps']
    post_steps = level_data[level]['post_steps']

    # 1. Pre-Smoothing
    u_smooth, p_smooth = mg_utils.apply_schwarz_smoother(
        smoother_data, pre_steps, u, p, f, g, A, B)

    # 2. Residual Calculation & Restriction
    res_u = f - (A @ u_smooth + B.T @ p_smooth)
    res_p = g - (B @ u_smooth)
    
    R_u, R_p = level_data[level]['R_u'], level_data[level]['R_p']
    res_u_coarse = R_u @ res_u
    res_p_coarse = R_p @ res_p
    
    # 3. Coarse Grid Correction (Recursion)
    n_coarse = level_data[level-1]['n']
    m_coarse = level_data[level-1]['m']
    e_u_coarse = np.zeros((n_coarse, 1))
    e_p_coarse = np.zeros((m_coarse, 1))

    for _ in range(gamma):
        e_u_coarse, e_p_coarse = w_cycle_recursive(
            level - 1, e_u_coarse, e_p_coarse, 
            res_u_coarse, res_p_coarse, level_data, gamma)

    # 4. Prolongation & Correction
    P_u, P_p = level_data[level]['P_u'], level_data[level]['P_p']
    u_corrected = u_smooth + P_u @ e_u_coarse
    p_corrected = p_smooth + P_p @ e_p_coarse

    # 5. Post-Smoothing
    u_final, p_final = mg_utils.apply_schwarz_smoother(
        smoother_data, post_steps, u_corrected, p_corrected, f, g, A, B)
        
    return u_final, p_final

def multigrid_solver(level_data, f_fine, g_fine, cycle_type='W', tol=1e-8, max_iter=50, logger=None):
    """
    Main driver loop for the Multigrid Solver.
    """
    gamma = 2 if cycle_type.upper() == 'W' else 1
    finest_level = len(level_data) - 1
    
    K_fine = level_data[finest_level]['K']
    rhs_fine = np.vstack([f_fine, g_fine])
    n_fine = level_data[finest_level]['n']
    m_fine = level_data[finest_level]['m']
    
    # Initial Guess (Zero)
    u_sol = np.zeros((n_fine, 1))
    p_sol = np.zeros((m_fine, 1))

    initial_res_norm = la.norm(rhs_fine)
    if initial_res_norm == 0: initial_res_norm = 1.0
    
    residual_history = [1.0]

    log_print = logger.log if logger else print
    log_print(f"--- Starting Multigrid Solver ({cycle_type}-Cycle) ---")

    for i in range(max_iter):
        u_sol, p_sol = w_cycle_recursive(
            finest_level, u_sol, p_sol, f_fine, g_fine, level_data, gamma)
        
        current_res = rhs_fine - K_fine @ np.vstack([u_sol, p_sol])
        res_norm = la.norm(current_res)
        rel_res = res_norm / initial_res_norm
        
        log_print(f"Iter {i+1:02d}: Relative Residual = {rel_res:.4e}")
        residual_history.append(rel_res)
        
        if rel_res < tol:
            log_print(f"\nConvergence achieved in {i+1} iterations.")
            break
    else:
        log_print("\nSolver did NOT converge within maximum iterations.")
        
    
    # Calculate Asymptotic Convergence Rate
    # We use the last 'k' iterations.
    steps_to_measure = 4  
    
    # Check if we have enough history
    if len(residual_history) > steps_to_measure:
        res_end = residual_history[-1]
        res_start = residual_history[-(steps_to_measure + 1)]
        denom = steps_to_measure
    else:
        # Fallback: Use full history if convergence was super fast
        res_end = residual_history[-1]
        res_start = residual_history[0]
        denom = len(residual_history) - 1

    if denom > 0:
        conv_rate = (res_end / res_start)**(1.0 / denom)
    else:
        conv_rate = 0.0
        
    return u_sol, p_sol, conv_rate

# ===============================================================
# UTILITIES FOR LOGGING
# ===============================================================
class Logger:
    """Simple logger to write to both console and file simultaneously."""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w')

    def log(self, message):
        print(message)
        self.log_file.write(message + '\n')

    def close(self):
        self.log_file.close()

# ===============================================================
# MAIN EXECUTION
# ===============================================================

if __name__ == '__main__':

    # --- 1. CONFIGURATION ---
    K_LEVELS = [4, 8, 12]  # Polynomial degrees hierarchy
    NUM_PRE_SMOOTH = 8
    NUM_POST_SMOOTH = 8
    EPS_SAFETY = 1e-8
    CYCLE_TYPE = 'W'
    TOLERANCE = 1e-12
    
    # Directories
    INPUT_DIR = "mg_level_data"
    OUTPUT_DIR = "multigrid_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 2. HIERARCHY ASSEMBLY ---
    print("--- Assembling Multigrid Hierarchy ---")
    level_data = {}
    num_levels = len(K_LEVELS)
    
    # A. Load Pre-computed Data & Generate Smoothers
    for i in range(num_levels):
        N = K_LEVELS[i]
        print(f"-> Loading Level {i} (p={N})...")
        
        data_file = os.path.join(INPUT_DIR, f"level_data_N{N:02d}.npy")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file missing: {data_file}. Run precompute script first.")
            
        level_data[i] = np.load(data_file, allow_pickle=True).item()
        level_data[i]['pre_steps'] = NUM_PRE_SMOOTH
        level_data[i]['post_steps'] = NUM_POST_SMOOTH
        
        print(f"   Generating Additive Schwarz Smoother...")
        level_data[i]['smoother_data'] = mg_utils.setup_schwarz_smoother(
            level_data[i]['A'], 
            level_data[i]['B'], 
            eps=EPS_SAFETY
        )

    # B. Compute Transfer Operators (P, R)
    print("\n--- Calculating Transfer Operators & Coarse Matrices (Galerkin) ---")
    
    for i in range(1, num_levels):
        Nc = K_LEVELS[i-1]
        Nf = K_LEVELS[i]
        print(f"-> Transfer p={Nc} <-> p={Nf}")
        
        # 1D Operators
        I0_1D = mg_utils.prolongate0_1D(Nc, Nf)
        I1_1D = mg_utils.prolongate1_1D(Nc, Nf)
        
        # 1D Mass Matrices for L2 Restriction
        nodes_f, _ = bf_polynomials.lobatto_quad(Nf)
        nodes_q, wq = bf_polynomials.gauss_quad(60) # High quadrature for exactness
        
        h_f = bf_polynomials.lagrange_basis(nodes_f, nodes_q)
        e_f = bf_polynomials.edge_basis(nodes_f, nodes_q)
        sqrt_w = np.sqrt(wq)
        
        M0_1D_f = (h_f * sqrt_w) @ (h_f * sqrt_w).T
        M1_1D_f = (e_f * sqrt_w) @ (e_f * sqrt_w).T
        
        nodes_c, _ = bf_polynomials.lobatto_quad(Nc)
        h_c = bf_polynomials.lagrange_basis(nodes_c, nodes_q)
        e_c = bf_polynomials.edge_basis(nodes_c, nodes_q)
        
        M0_1D_c = (h_c * sqrt_w) @ (h_c * sqrt_w).T
        M1_1D_c = (e_c * sqrt_w) @ (e_c * sqrt_w).T
        
        R0_1D = mg_utils.restrict_L2(I0_1D, M0_1D_f, M0_1D_c)
        R1_1D = mg_utils.restrict_L2(I1_1D, M1_1D_f, M1_1D_c)
        
        # 2D Operators
        P0, P1, P2, R0, R1, R2 = mg_utils.build_2D_ops(I0_1D, I1_1D, R0_1D, R1_1D)
        
        # Assemble P_u (1-forms) and P_p (2-forms / Dual 0-forms)
        P_u = P1
        P_p = P2 
        
        R_u = R1
        R_p = R2

        # Block Diagonal Operators for System [u, p]
        P_total = la.block_diag(P_u, P_p)
        R_total = la.block_diag(R_u, R_p)
        
        # Store for use in Cycle
        level_data[i]['P'] = P_total
        level_data[i]['R'] = R_total
        level_data[i]['P_u'] = P_u; level_data[i]['P_p'] = P_p
        level_data[i]['R_u'] = R_u; level_data[i]['R_p'] = R_p

    # C. Galerkin Projection (K_coarse = R * K_fine * P)
    for i in range(num_levels - 1, 0, -1):
        print(f"-> Assembling K_coarse for Level {i-1} (p={K_LEVELS[i-1]})...")
        K_fine = level_data[i]['K']
        R = level_data[i]['R']
        P = level_data[i]['P']
        
        # Explicit Galerkin Assembly
        level_data[i-1]['K'] = R @ K_fine @ P
        
    print("Hierarchy Assembly Complete.")

    # --- 3. RUN SOLVER ---
    log_filename = f"{CYCLE_TYPE.lower()}_cycle_p{K_LEVELS[-1]}_sm{NUM_PRE_SMOOTH}.txt"
    log_path = os.path.join(OUTPUT_DIR, log_filename)
    logger = Logger(log_path)
    
    try:
        logger.log("=== Multigrid Solver Configuration ===")
        logger.log(f"Cycle:       {CYCLE_TYPE}")
        logger.log(f"Levels (p):  {K_LEVELS}")
        logger.log(f"Tolerance:   {TOLERANCE:.2e}")
        logger.log(f"Smoothing:   {NUM_PRE_SMOOTH} pre / {NUM_POST_SMOOTH} post")
        logger.log("======================================\n")
        
        # Prepare RHS
        finest_idx = num_levels - 1
        rhs_fine = -level_data[finest_idx]['rhs']
        n_fine = level_data[finest_idx]['n']
        
        f_vec = rhs_fine[:n_fine]
        g_vec = rhs_fine[n_fine:]
        
        u_final, p_final, rate = multigrid_solver(
            level_data, f_vec, g_vec, 
            cycle_type=CYCLE_TYPE, 
            tol=TOLERANCE, 
            logger=logger
        )
        
        logger.log("\n------------------------------------------")
        logger.log(f"Average Convergence Rate: {rate:.4f}")
        logger.log("------------------------------------------")
        
    finally:
        logger.close()
        
    print(f"\nExecution finished. Log saved to: {log_path}")