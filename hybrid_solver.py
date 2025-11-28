"""
Hybrid Schur-Complement Solver with p-Multigrid.

This script solves the multi-element mixed Poisson problem using a hybrid strategy:
1. Outer Solver: Conjugate Gradient (CG) on the Schur Complement system for interface multipliers (lambda).
2. Inner Solver: p-Multigrid ( to apply the local inverse operator (K_elem^-1) efficiently.

Author: Manuel Fernandez Lopez
Master Thesis - TU Delft
"""

import numpy as np
import scipy.linalg as la
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import LinearOperator, cg
import os
import matplotlib.pyplot as plt

# Custom Modules
import bf_polynomials
import multigrid_utilities as mg_utils

# ==============================================================================
# 1. ELEMENT & CONNECTIVITY BUILDERS
# ==============================================================================

def build_local_element(p, quad_degree, x_range, y_range):
    """
    Constructs local system matrices (M1, E21) for a single element.
    Uses efficient NumPy broadcasting instead of loops where possible.
    """
    N = p
    # Nodes & Quadrature
    nodes, _ = bf_polynomials.lobatto_quad(N)
    q_nodes, q_weights = bf_polynomials.gauss_quad(quad_degree)
    q_weights_2D = np.outer(q_weights, q_weights).flatten()
    num_q = quad_degree**2
    
    # Basis Evaluation
    h_x = bf_polynomials.lagrange_basis(nodes, q_nodes)
    h_y = bf_polynomials.lagrange_basis(nodes, q_nodes)
    e_x = bf_polynomials.edge_basis(nodes, q_nodes)
    e_y = bf_polynomials.edge_basis(nodes, q_nodes)

    # Geometry
    a, b = x_range; c, d = y_range
    dx, dy = (b - a) / 2.0, (d - c) / 2.0
    detJ = dx * dy
    ax_metric, ay_metric = dy / dx, dx / dy

    num_dofs_1 = 2 * N * (N + 1)
    num_dofs_2 = N * N

    # --- Build Psi1 (Edge Basis) ---
    psi1 = np.zeros((num_dofs_1, num_q, 2))
    count = 0
    # Vertical Edges (Component Y: h(x) * e(y))
    for i in range(N + 1):
        for j in range(N):
            psi1_y = np.outer(h_x[i], e_y[j]).flatten()
            psi1[count, :, 0] = psi1_y
            count += 1
            
    # Horizontal Edges (Component X: e(x) * h(y))
    for i in range(N):
        for j in range(N + 1):
            psi1_x = np.outer(e_x[i], h_y[j]).flatten()
            psi1[count, :, 1] = psi1_x
            count += 1

    # --- Build M1 (Mass Matrix 1-form) ---

    M1_k = np.zeros((num_dofs_1, num_dofs_1))
    
    # Vectorized Mass Matrix Construction

    # Component Y (index 0 in psi1 storage) 
    P0 = psi1[:, :, 0]
    M_y = (P0 * q_weights_2D) @ P0.T * ay_metric
    
    # Component X (index 1 in psi1 storage)
    P1 = psi1[:, :, 1]
    M_x = (P1 * q_weights_2D) @ P1.T * ax_metric
    
    M1_k = M_x + M_y

    # --- Build E21 (Incidence Matrix) ---
    E21_k = np.zeros((num_dofs_2, num_dofs_1), dtype=int)
    for i in range(N):
        for j in range(N):
            cid = i * N + j
            # Indices of surrounding edges
            b = N * (N + 1) + i * (N + 1) + j
            r = (i + 1) * N + j
            t = N * (N + 1) + i * (N + 1) + (j + 1)
            l = i * N + j
            
            E21_k[cid, b] = -1
            E21_k[cid, r] = +1
            E21_k[cid, t] = +1
            E21_k[cid, l] = -1
            
    return {'M1_k': M1_k, 'E21_k': E21_k}

def get_boundary_indices(p):
    """Returns local indices of DOFs on the 4 boundaries (Top, Bottom, Left, Right)."""
    # Vertical edges (N+1 cols of N edges)
    left_indices  = [i for i in range(p)] # Col 0
    right_indices = [p * p + i for i in range(p)] # Col N
    
    # Horizontal edges (N cols of N+1 edges)
    offset = p * (p + 1)
    bottom_indices = [offset + j * (p + 1) for j in range(p)] # Row 0
    top_indices    = [offset + j * (p + 1) + p for j in range(p)] # Row N
    
    return top_indices, bottom_indices, left_indices, right_indices

def build_connectivity_matrix(p, Kx, Ky):
    """
    Constructs the global connectivity matrix N (C_sparse) that enforces continuity
    of normal fluxes across element interfaces.
    """
    top_ids, bot_ids, left_ids, right_ids = get_boundary_indices(p)
    dofs_per_edge = p
    
    dofs_u = 2 * p * (p + 1)
    dofs_p = p * p
    local_size = dofs_u + dofs_p
    total_dofs = Kx * Ky * local_size
    
    num_lambda = (Kx - 1) * Ky * dofs_per_edge + Kx * (Ky - 1) * dofs_per_edge
    
    # Use sparse matrix triplet format (row, col, data) for efficiency
    rows, cols, data = [], [], []
    current_lambda = 0
    
    print("--- Building Connectivity Matrix N ---")
    
    for kx in range(Kx):
        for ky in range(Ky):
            k_elem = kx * Ky + ky
            
            # 1. Right Neighbor (Vertical Interface)
            if kx < Kx - 1:
                k_right = (kx + 1) * Ky + ky
                
                offset_L = k_elem * local_size
                offset_R = k_right * local_size
                
                for i in range(dofs_per_edge):
                    rows.extend([current_lambda, current_lambda])
                    cols.extend([offset_L + right_ids[i], offset_R + left_ids[i]])
                    data.extend([1, -1]) # Flux matching
                    current_lambda += 1
            
            # 2. Top Neighbor (Horizontal Interface)
            if ky < Ky - 1:
                k_top = kx * Ky + (ky + 1)
                
                offset_B = k_elem * local_size
                offset_T = k_top * local_size
                
                for i in range(dofs_per_edge):
                    rows.extend([current_lambda, current_lambda])
                    cols.extend([offset_B + top_ids[i], offset_T + bot_ids[i]])
                    data.extend([1, -1])
                    current_lambda += 1
                    
    # Build sparse matrix
    N_matrix = csc_matrix((data, (rows, cols)), shape=(num_lambda, total_dofs))
    return N_matrix

# ==============================================================================
# 2. RHS ASSEMBLY (VECTORIZED)
# ==============================================================================

def build_rhs_local_vectorized(p, quad_degree, forcing_func, x_range, y_range):
    """
    Computes local RHS vector F_k using full vectorization over quadrature points.
    """
    N = p
    nodes_q, weights_q = bf_polynomials.gauss_quad(quad_degree)
    nodes_gll, _ = bf_polynomials.lobatto_quad(N)
    
    # Map GLL nodes to physical space
    x_min, x_max = x_range
    y_min, y_max = y_range
    x_phys = 0.5 * (x_max - x_min) * nodes_gll + 0.5 * (x_max + x_min)
    y_phys = 0.5 * (y_max - y_min) * nodes_gll + 0.5 * (y_max + y_min)
    
    # Cell boundaries
    x_starts, x_ends = x_phys[:-1], x_phys[1:]
    y_starts, y_ends = y_phys[:-1], y_phys[1:]
    
    # Jacobians & Centers
    Jx = (x_ends - x_starts) / 2.0
    Jy = (y_ends - y_starts) / 2.0
    Xc = (x_ends + x_starts) / 2.0
    Yc = (y_ends + y_starts) / 2.0
    
    # X_quad: (N_quad, N_cells_x)
    X_q = nodes_q[:, None] * Jx[None, :] + Xc[None, :]
    Y_q = nodes_q[:, None] * Jy[None, :] + Yc[None, :]
    
    # Full 2D Grid for evaluation: (Nqy, Nqx, Ny, Nx)
    X_grid = X_q[None, :, None, :] 
    Y_grid = Y_q[:, None, :, None]
    
    F_vals = forcing_func(X_grid, Y_grid)
    
    # Integration Weights
    W_grid = weights_q[:, None] * weights_q[None, :]
    
    # Integrate: Sum(F * W) * J
    Integrals = np.sum(F_vals * W_grid[:, :, None, None], axis=(0, 1))
    
    # Apply Jacobians
    J_cells = Jx[None, :] * Jy[:, None]
    RHS_matrix = Integrals * J_cells
    
    return RHS_matrix.flatten(order='F') # Flatten column-major

# ==============================================================================
# 3. SCHUR & INVERSE
# ==============================================================================

def apply_A_inverse(b_vec, K_dummy, local_size, p_mg_setup, solver_params):
    """
    Applies the global inverse A^-1 * b using the p-Multigrid solver element-wise.
    """
    x_vec = np.zeros_like(b_vec)
    num_elements = len(b_vec) // local_size
    
    for k in range(num_elements):
        offset = k * local_size
        b_loc = b_vec[offset : offset + local_size]
        
        # Use the Black-Box p-Multigrid Solver
        u_k, p_k, _ = mg_utils.p_multigrid_solver(b_loc, p_mg_setup, solver_params)
        
        x_loc = np.vstack([u_k, p_k]).flatten()
        x_vec[offset : offset + local_size] = x_loc
        
    return x_vec

def apply_schur_op(v, C_sparse, A_inv_func, K, local_size, mg_setup, params):
    """
    Applies the Schur operator S * v = (C * A^-1 * C^T) * v.
    """
    # 1. Back-projection: w1 = C^T * v
    w1 = C_sparse.T @ v
    
    # 2. Local Inverses: w2 = A^-1 * w1
    w2 = A_inv_func(w1, K, local_size, mg_setup, params)
    
    # 3. Projection: res = C * w2
    return C_sparse @ w2

# ==============================================================================
# 4. MAIN SIMULATION SCRIPT
# ==============================================================================

def forcing_f(x, y): 
    return -2 * np.pi**2 * np.sin(np.pi*x) * np.sin(np.pi*y)

def exact_sol_funcs():
    phi = lambda x, y: np.sin(np.pi*x) * np.sin(np.pi*y)
    u   = lambda x, y: (np.pi*np.cos(np.pi*x)*np.sin(np.pi*y), 
                        np.pi*np.sin(np.pi*x)*np.cos(np.pi*y))
    return phi, u

# ==============================================================================
# 5. MULTIGRID HIERARCHY SETUP
# ==============================================================================
def setup_p_multigrid_hierarchy(p_levels, quad_degree):
    """
    Pre-computes all operators and matrices for the p-multigrid hierarchy.
    This function is computationally expensive and runs ONCE.
    
    Args:
        p_levels (list): List of polynomial degrees, e.g., [2, 4, 8].
        quad_degree (int): Quadrature degree for integration.
        
    Returns:
        dict: The 'level_data' dictionary with all pre-computed info.
    """
    print("--- Configuring p-Multigrid Hierarchy ---")
    level_data = {}
    num_levels = len(p_levels)

    # --- Part A: Build data for each level (matrices and smoother) ---
    for i, p in enumerate(p_levels):
        print(f"-> Level {i} (p={p}): Building local matrices and smoother...")
        local_mats = build_local_element(p, quad_degree, (-1, 1), (-1, 1)) 
        A = local_mats['M1_k']
        B = local_mats['E21_k']
        
        # Store matrices and dimensions
        n, m = B.shape[1], B.shape[0]
        K = np.block([[A, B.T], [B, np.zeros((m, m))]])
        level_data[i] = {'A': A, 'B': B, 'K': K, 'n': n, 'm': m}
        
        # Safety factor for Schwarz smoother inequalities
        EPS_SAFETY_FACTOR = 1e-8
        # Pre-compute smoother data for this level
        level_data[i]['smoother_data'] = mg_utils.setup_schwarz_smoother(A, B, eps=EPS_SAFETY_FACTOR)

    # --- Part B: Build Transfer Operators between levels ---
    for i in range(1, num_levels):
        p_coarse = p_levels[i-1]
        p_fine = p_levels[i]
        print(f"-> Transfer setup: p={p_coarse} <-> p={p_fine}...")

        # 1. Prolongation Operators 1D
        I0_1D = mg_utils.prolongate0_1D(p_coarse, p_fine)
        I1_1D = mg_utils.prolongate1_1D(p_coarse, p_fine)

        # 2. Mass Matrices 1D (for L2 restriction)
        nodes_f, _ = bf_polynomials.lobatto_quad(p_fine)
        nodes_q, wq = bf_polynomials.gauss_quad(quad_degree)
        h_f = bf_polynomials.lagrange_basis(nodes_f, nodes_q)
        e_f = bf_polynomials.edge_basis(nodes_f, nodes_q)
        sqrt_w = np.sqrt(wq)
        
        # Weighted mass matrices (B @ W @ B.T)
        M0_1D_f = (h_f * sqrt_w) @ (h_f * sqrt_w).T
        M1_1D_f = (e_f * sqrt_w) @ (e_f * sqrt_w).T
        
        nodes_c, _ = bf_polynomials.lobatto_quad(p_coarse)
        h_c = bf_polynomials.lagrange_basis(nodes_c, nodes_q)
        e_c = bf_polynomials.edge_basis(nodes_c, nodes_q)
        M0_1D_c = (h_c * sqrt_w) @ (h_c * sqrt_w).T
        M1_1D_c = (e_c * sqrt_w) @ (e_c * sqrt_w).T

        # 3. Restriction Operators 1D
        R0_1D = mg_utils.restrict_L2(I0_1D, M0_1D_f, M0_1D_c)
        R1_1D = mg_utils.restrict_L2(I1_1D, M1_1D_f, M1_1D_c)

        # 4. Build and Store 2D Operators for the Fine Level
        _, P_u, P_p, _, R_u, R_p = mg_utils.build_2D_ops(I0_1D, I1_1D, R0_1D, R1_1D)
        
        level_data[i]['P_u'] = P_u
        level_data[i]['P_p'] = P_p
        level_data[i]['R_u'] = R_u
        level_data[i]['R_p'] = R_p
        
        # 5. Galerkin Projection for Coarse Grid Matrix
        # K_coarse = R * K_fine * P
        # Construct Block Diagonal Operators for the system [u, p]
        P_total = la.block_diag(P_u, P_p)
        R_total = la.block_diag(R_u, R_p)
        
        K_fine = level_data[i]['K']
        K_coarse_galerkin = R_total @ K_fine @ P_total
        
        level_data[i-1]['K'] = K_coarse_galerkin
        
    print("p-Multigrid Hierarchy Configured Successfully.")
    return level_data

if __name__ == '__main__':
    
    # --- CONFIGURATION ---
    P_ORDER = 6
    Kx, Ky = 4, 4
    QUAD_DEG = 64
    PLOT_DEG = 64
    
    CG_TOL = 1e-10
    
    # Inner Solver (p-Multigrid) Configuration
    P_LEVELS = [2,4,6]
    SOLVER_PARAMS = {
        'cycle_type': 'V', 
        'tol': 1e-12, 
        'max_iter': 40,
        'pre_steps': 8, 'post_steps': 8
    }
    
    # Domain
    X_LIM, Y_LIM = (-2.0, 0.0), (0.0, 2.0)
    
    # --- 1. SETUP HIERARCHY ---
    print(f"--- Hybrid Solver Setup (p={P_ORDER}, Grid={Kx}x{Ky}) ---")

    p_mg_setup = setup_p_multigrid_hierarchy(P_LEVELS, QUAD_DEG)
    
    # Update smoothing steps
    for lvl in p_mg_setup.values():
        lvl['pre_steps'] = SOLVER_PARAMS['pre_steps']
        lvl['post_steps'] = SOLVER_PARAMS['post_steps']

    # --- 2. ASSEMBLY ---
    print("-> Assembling Global System...")
    num_elements = Kx * Ky
    
    # Local Dimensions
    dofs_u = 2 * P_ORDER * (P_ORDER + 1)
    dofs_p = P_ORDER * P_ORDER
    loc_size = dofs_u + dofs_p
    total_dofs = num_elements * loc_size
    
    # Build RHS Vector
    elem_w = (X_LIM[1] - X_LIM[0]) / Kx
    elem_h = (Y_LIM[1] - Y_LIM[0]) / Ky
    
    F_global = np.zeros(total_dofs)
    
    for kx in range(Kx):
        for ky in range(Ky):
            k = kx * Ky + ky
            x0 = X_LIM[0] + kx * elem_w
            y0 = Y_LIM[0] + ky * elem_h
            
            # Vectorized Local RHS
            F_loc = build_rhs_local_vectorized(P_ORDER, QUAD_DEG, forcing_f, 
                                               (x0, x0+elem_w), (y0, y0+elem_h))
            
            # Place in global vector (Pressure part only, Velocity RHS is 0)
            offset = k * loc_size
            F_global[offset + dofs_u : offset + loc_size] = F_loc

    # Build Connectivity Matrix
    C_sparse = build_connectivity_matrix(P_ORDER, Kx, Ky)
    num_multipliers = C_sparse.shape[0]
    
    # --- 3. ITERATIVE SOLUTION (SCHUR-CG) ---
    print(f"-> Starting Iterative Solver (Schur-CG)...")

    
    # A) RHS for Schur System: b_eff = C * A^-1 * F
    A_inv_F = apply_A_inverse(F_global, None, loc_size, p_mg_setup, SOLVER_PARAMS)
    b_schur = C_sparse @ A_inv_F
    
    b_norm = np.linalg.norm(b_schur)
    if b_norm < 1e-15: b_norm = 1.0
    
    # B) Linear Operator
    S_operator = LinearOperator(
        (num_multipliers, num_multipliers),
        matvec = lambda v: apply_schur_op(v, C_sparse, apply_A_inverse, None, 
                                          loc_size, p_mg_setup, SOLVER_PARAMS)
    )
    
    # C) Conjugate Gradient
    residuals = []
    def callback(xk):
        res = b_schur - S_operator.matvec(xk)
        rel = np.linalg.norm(res) / b_norm
        residuals.append(rel)
        print(f"   CG Iter {len(residuals):03d}: Rel Res = {rel:.4e}")
        
    lambda_sol, exit_code = cg(S_operator, b_schur, tol=CG_TOL, maxiter=500, callback=callback)
    
    if exit_code == 0:
        print(f"-> CG Converged in {len(residuals)} iterations.")
    else:
        print(f"-> CG Failed to converge (Code {exit_code}).")
        
    # D) Recover Primal Solution u
    # u = A^-1 * (F - C^T * lambda)
    rhs_recovery = F_global - C_sparse.T @ lambda_sol
    u_sol_global = apply_A_inverse(rhs_recovery, None, loc_size, p_mg_setup, SOLVER_PARAMS)
    

    # --- 4. POST-PROCESSING & ERROR ANALYSIS ---
    print("Saving Results...")
    
    # Save Results Log
    fname = f"hybrid_p{P_ORDER}_K{num_elements}.txt"
    with open(fname, 'w') as f:
        f.write("--- Hybrid Solver Results ---\n")
        f.write(f"Grid: {Kx}x{Ky} (p={P_ORDER})\n")
        f.write(f"Inner Solver: p-MG {P_LEVELS}, Cycle {SOLVER_PARAMS['cycle_type']}\n")
        
        f.write(f"\n[Problem Dimensions]\n")
        f.write(f"N_dof_elemental = {loc_size}\n")
        f.write(f"N_dof_total_elements = {total_dofs}\n")
        f.write(f"N_dof_multipliers = {num_multipliers}\n")
        f.write(f"N_dof_total = {total_dofs + num_multipliers}\n")
        
        f.write(f"\n[Tolerances]\n")
        f.write(f"CG (Outer) Tolerance = {CG_TOL:.1e}\n")
        f.write(f"p-MG (Inner) Tolerance = {SOLVER_PARAMS['tol']:.1e}\n")
        
        f.write(f"Outer Solver: CG Converged in {len(residuals)} iters\n")
        
        f.write("\n[Residual History]\n")
        for i, r in enumerate(residuals):
            f.write(f"{i+1} {r:.6e}\n")
            
    print(f"Done. Results saved to {fname}.")