"""
Multigrid Utilities.

This module provides the core functionality for the Mimetic Spectral Element Method (MSEM)
solver, including:
1. Assembly of high-order mimetic matrices (Mass, Incidence) using GLL nodes.
2. Construction of p-Multigrid transfer operators (Prolongation, Restriction).
3. Setup and application of the Additive Schwarz Smoother for saddle-point systems.
4. Multigrid cycle implementations.

Author: Manuel Fernandez Lopez
Master Thesis - TU Delft
"""

import numpy as np
import scipy.linalg as la
import bf_polynomials  # Ensure this module is available in your path

# =============================================================================
# 1. TRANSFER OPERATORS (PROLONGATION / RESTRICTION)
# =============================================================================

def prolongate0_1D(Nc: int, Nf: int) -> np.ndarray:
    """
    Constructs the 1D Prolongation operator for 0-forms (Nodal).
    Interpolates from Coarse (Nc) to Fine (Nf) GLL nodes using Lagrange basis.
    """
    nodes_c, _ = bf_polynomials.lobatto_quad(Nc)
    nodes_f, _ = bf_polynomials.lobatto_quad(Nf)
    
    # Evaluate coarse Lagrange basis at fine nodes
    Lc_eval_on_f = bf_polynomials.lagrange_basis(nodes_c, nodes_f)
    
    # The prolongation matrix is the transpose of the evaluation matrix
    return Lc_eval_on_f.T

def prolongate1_1D(Nc: int, Nf: int) -> np.ndarray:
    """
    Constructs the 1D Prolongation operator for 1-forms (Edge).
    Based on the exact integration of edge functions.
    """
    xc, _ = bf_polynomials.lobatto_quad(Nc)
    xf, _ = bf_polynomials.lobatto_quad(Nf)
    
    # Evaluate coarse basis at fine nodes
    H = bf_polynomials.lagrange_basis(xc, xf).T

    P1 = np.zeros((Nf, Nc))
    for i in range(1, Nf + 1):
        # Difference of coarse basis values at fine edge endpoints
        diff = H[i-1, :] - H[i, :]
        P1[i-1, :] = np.cumsum(diff[:-1])
        
    return P1

def restrict_L2(P: np.ndarray, M_f: np.ndarray, M_c: np.ndarray) -> np.ndarray:
    """
    Computes the Restriction operator as the L2-adjoint of Prolongation.
    Solves: M_c * R = P.T * M_f
    """
    # Use solve with assumption of SPD matrix M_c for stability
    return la.solve(M_c, P.T @ M_f, assume_a='pos')

def build_2D_ops(P0_1D, P1_1D, R0_1D, R1_1D):
    """
    Assembles 2D transfer operators using Kronecker products.
    
    Returns:
        tuple: (P0, P1, P2, R0, R1, R2)
        - P0/R0: 0-forms (Nodal)
        - P1/R1: 1-forms (Edges, block [Ux, Uy])
        - P2/R2: 2-forms (Cells)
    """
    kron = np.kron

    # 0-forms (Nodal x Nodal)
    P0 = kron(P0_1D, P0_1D)
    R0 = kron(R0_1D, R0_1D)

    # 2-forms (Edge x Edge)
    P2 = kron(P1_1D, P1_1D)
    R2 = kron(R1_1D, R1_1D)

    # 1-forms (Vector valued)
    # P1_x: Ux (Nodal in X, Edge in Y)
    P1_x = kron(P0_1D, P1_1D)
    # P1_y: Uy (Edge in X, Nodal in Y)
    P1_y = kron(P1_1D, P0_1D)
    
    P1 = np.block([
        [P1_x, np.zeros((P1_x.shape[0], P1_y.shape[1]))],
        [np.zeros((P1_y.shape[0], P1_x.shape[1])), P1_y]
    ])

    R1_x = kron(R0_1D, R1_1D)
    R1_y = kron(R1_1D, R0_1D)
    R1 = np.block([
        [R1_x, np.zeros((R1_x.shape[0], R1_y.shape[1]))],
        [np.zeros((R1_y.shape[0], R1_x.shape[1])), R1_y]
    ])
    
    return P0, P1, P2, R0, R1, R2

# =============================================================================
# 2. MIMETIC SYSTEM ASSEMBLY (VECTORIZED)
# =============================================================================

def build_mimetic_block(N, quad_degree, plot_degree, x_range=(-1., 1.), y_range=(-1., 1.)):
    """
    Assembles the Mass (M) and Incidence (E) matrices for a single spectral element.
    Uses efficient vectorization (Kronecker products) to avoid slow Python loops.
    """
    
    # 1. Geometry and Nodes
    nodes, _ = bf_polynomials.lobatto_quad(N)
    
    # Quadrature
    q_nodes, q_weights = bf_polynomials.gauss_quad(quad_degree)
    q_weights_2D = np.outer(q_weights, q_weights).flatten()
    num_q = quad_degree**2

    # Geometric Factors (Affine mapping)
    a, b = x_range; c, d = y_range
    dx, dy = (b-a)/2.0, (d-c)/2.0
    detJ = dx * dy
    ax, ay = dy/dx, dx/dy  # Metric terms for 1-forms

    # DoF Counts
    num_dofs_0 = (N + 1)**2
    num_dofs_1 = 2 * N * (N + 1)
    num_dofs_2 = N**2

    # 2. Basis Functions Evaluation (1D)
    h_x = bf_polynomials.lagrange_basis(nodes, q_nodes)
    e_x = bf_polynomials.edge_basis(nodes, q_nodes)

    # 3. 2D Basis Construction (Vectorized via Kronecker)
    # psi0: Nodal basis
    psi0 = np.kron(h_x, h_x).reshape(num_dofs_0, num_q)

    # psi1: Edge basis (Vector valued)
    # Component X: 
    psi1_x_comp = np.kron(h_x, e_x).reshape(N * (N + 1), num_q)
    # Component Y:
    psi1_y_comp = np.kron(e_x, h_x).reshape(N * (N + 1), num_q)

    psi1 = np.zeros((num_dofs_1, num_q, 2))
    psi1[:N * (N + 1), :, 0] = psi1_x_comp
    psi1[N * (N + 1):, :, 1] = psi1_y_comp

    # psi2: Cell basis (Discontinuous)
    psi2 = np.kron(e_x, e_x).reshape(num_dofs_2, num_q)

    # 4. Mass Matrices Assembly
    # M0 (0-forms)
    M0 = psi0 @ np.diag(q_weights_2D) @ psi0.T
    M0 = detJ * M0

    # M1 (1-forms) - Weighted by metric terms
    psi1_x = psi1[:, :, 0]
    psi1_y = psi1[:, :, 1]
    M1_x = psi1_x @ np.diag(q_weights_2D) @ psi1_x.T
    M1_y = psi1_y @ np.diag(q_weights_2D) @ psi1_y.T
    M1 = ax * M1_x + ay * M1_y # Sum of components

    # M2 (2-forms)
    M2 = psi2 @ np.diag(q_weights_2D) @ psi2.T
    M2 = (1.0/detJ) * M2
    
    # 5. Incidence Matrices (Topology)
    num_nodes = (N + 1)**2
    num_edges = 2 * N * (N + 1)
    num_cells = N * N

    # E10: Gradient (Nodes -> Edges)
    E10 = np.zeros((num_edges, num_nodes), dtype=int)
    edge_id = 0
    # Vertical edges
    for i in range(N + 1):
        for j in range(1, N + 1):
            s = i * (N + 1) + (j - 1); e = i * (N + 1) + j
            E10[edge_id, s] = +1; E10[edge_id, e] = -1 # Orientation convention
            edge_id += 1
    # Horizontal edges
    for i in range(1, N + 1):
        for j in range(N + 1):
            s = (i - 1) * (N + 1) + j; e = i * (N + 1) + j
            E10[edge_id, s] = +1; E10[edge_id, e] = -1
            edge_id += 1

    # E21: Divergence (Edges -> Cells)
    E21 = np.zeros((num_cells, num_edges), dtype=int)
    for i in range(N):
        for j in range(N):
            cid = i * N + j
            b = N * (N + 1) + i * (N + 1) + j
            r = (i + 1) * N + j
            t = N * (N + 1) + i * (N + 1) + (j + 1)
            l = i * N + j
            # Divergence stencil
            E21[cid, b] = +1
            E21[cid, r] = +1
            E21[cid, t] = -1
            E21[cid, l] = -1

    # 6. Plotting Data (Optional, for reconstruction)
    # Pre-computes basis on a fine plotting grid
    plot_nodes, _ = bf_polynomials.gauss_quad(plot_degree)
    num_q_plot = plot_degree**2
    h_x_p = bf_polynomials.lagrange_basis(nodes, plot_nodes)
    e_x_p = bf_polynomials.edge_basis(nodes, plot_nodes)
    
    # psi2 plot
    psi2_plot = np.kron(e_x_p, e_x_p).reshape(num_dofs_2, num_q_plot)
    M2_inv = la.inv(M2)
    psi0_dual_plot = M2_inv @ psi2_plot

    # psi0 plot
    psi0_plot = np.kron(h_x_p, h_x_p).reshape(num_dofs_0, num_q_plot)
    
    # psi1 plot
    psi1_plot = np.zeros((num_dofs_1, num_q_plot, 2))
    p1x_p = np.kron(h_x_p, e_x_p).reshape(N*(N+1), num_q_plot)
    p1y_p = np.kron(e_x_p, h_x_p).reshape(N*(N+1), num_q_plot)
    psi1_plot[:N*(N+1), :, 0] = p1x_p
    psi1_plot[N*(N+1):, :, 1] = -1.0 * p1y_p 

    return {
        'M0': M0, 'M1': M1, 'M2': M2, 'M2_inv': M2_inv,
        'E10': E10, 'E21': E21,
        'psi0_plot': psi0_plot, 'psi1_plot': psi1_plot, 'psi2_plot': psi2_plot,
        'psi0_dual_plot': psi0_dual_plot,
        'detJ': detJ, 'dx': dx, 'dy': dy, 'w2D': q_weights_2D,
        'x_range': x_range, 'y_range': y_range
    }

# =============================================================================
# 3. ADDITIVE SCHWARZ SMOOTHER (SETUP & APPLY)
# =============================================================================

def setup_schwarz_smoother(A, B, eps=1e-8):
    """
    Performs the one-time setup for the Additive Schwarz Smoother.
    Computes global preconditioners (A_hat, S_hat) and local operators.
    
    Args:
        A (np.ndarray): Block (1,1) of the system (Mass matrix M1).
        B (np.ndarray): Block (2,1) of the system (Divergence E21).
        eps (float): Safety factor for strict inequalities.
        
    Returns:
        dict: Contains 'P_ops', 'Q_ops', 'A_hat_i_ops', 'B_i_ops', 'S_hat_i_ops'.
    """
    n, m = B.shape[1], B.shape[0]

    # --- 1. Decomposition Operators (Q, P) ---
    Q_operators = []
    for i in range(m):
        q_i = np.zeros((m, 1)); q_i[i] = 1.0
        Q_operators.append(q_i)

    # Overlap counting (mu)
    mu = np.zeros(n)
    for j in range(n):
        count = np.count_nonzero(B[:, j])
        mu[j] = max(count, 1)

    P_hat_operators = [] # Boolean
    P_operators = []     # Scaled
    D_sqrt_operators = []

    for i in range(m):
        local_u_indices = np.where(B[i, :] != 0)[0]
        n_i = len(local_u_indices)
        
        p_hat_i = np.zeros((n, n_i))
        p_i = np.zeros((n, n_i))
        
        if n_i > 0:
            local_mu = mu[local_u_indices]
            D_sqrt = np.diag(np.sqrt(local_mu))
            
            for k, j in enumerate(local_u_indices):
                p_hat_i[j, k] = 1.0
                p_i[j, k] = 1.0 / np.sqrt(mu[j]) # Scaled P
                
            D_sqrt_operators.append(D_sqrt)
        else:
            D_sqrt_operators.append(np.zeros((0,0)))

        P_hat_operators.append(p_hat_i)
        P_operators.append(p_i)

    # --- 2. Global Preconditioners (A_hat, S_hat) ---
    d_A = np.diag(A)
    # Calculate Sigma
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_A))
    lambda_max_A = np.max(la.eigvalsh(D_inv_sqrt @ A @ D_inv_sqrt))
    sigma = 1.0 / ((1.0 + eps) * lambda_max_A)
    A_hat = (1.0 / sigma) * np.diag(d_A)
    A_hat_inv = np.diag(1.0 / np.diag(A_hat))

    # Calculate Tau (via D_mu)
    d_mu_diag = np.zeros(m)
    A_hat_diag = np.diag(A_hat)
    
    for i in range(m):
        p_hat = P_hat_operators[i]
        if p_hat.shape[1] == 0: continue
        
        local_edges = np.where(np.sum(p_hat, axis=1) > 0)[0]
        val = 0
        for j in local_edges:
            val += mu[j] * (B[i, j]**2) / A_hat_diag[j]
        d_mu_diag[i] = max(val, 1e-15)

    D_mu = np.diag(d_mu_diag)
    S_exact = B @ A_hat_inv @ B.T
    
    D_mu_inv_sqrt = np.diag(1.0 / np.sqrt(d_mu_diag))
    lambda_max_S = np.max(la.eigvalsh(D_mu_inv_sqrt @ S_exact @ D_mu_inv_sqrt))
    tau = 1.0 / ((1.0 + eps) * lambda_max_S)
    
    S_hat = (1.0 / tau) * D_mu

    # --- 3. Local Operators (A_hat_i, B_i, S_hat_i) ---
    A_hat_i_ops = []
    B_i_ops = []
    S_hat_i_ops = []
    
    s_hat_diag = np.diag(S_hat)

    for i in range(m):
        p_hat = P_hat_operators[i]
        q_i = Q_operators[i]
        
        # Local A_hat
        if p_hat.shape[1] > 0:
            a_hat_i = p_hat.T @ A_hat @ p_hat
        else:
            a_hat_i = np.zeros((0,0))
        A_hat_i_ops.append(a_hat_i)
        
        # Local B_i (Scaled)
        # B_i = Q^T B P_hat D_sqrt
        if p_hat.shape[1] > 0:
            b_hat = q_i.T @ B @ p_hat
            b_i = b_hat @ D_sqrt_operators[i]
        else:
            b_i = np.zeros((1,0))
        B_i_ops.append(b_i)
        
        # Local S_hat (Diagonal element)
        S_hat_i_ops.append(np.array([[s_hat_diag[i]]]))

    return {
        'P_ops': P_operators, 'Q_ops': Q_operators,
        'A_hat_i_ops': A_hat_i_ops, 'B_i_ops': B_i_ops, 'S_hat_i_ops': S_hat_i_ops
    }

def apply_schwarz_smoother(smoother_data, num_steps, u, p, f, g, A, B):
    """
    Applies 'num_steps' of the Additive Schwarz Smoother.
    """
    P_ops = smoother_data['P_ops']
    Q_ops = smoother_data['Q_ops']
    A_hat_i_ops = smoother_data['A_hat_i_ops']
    B_i_ops = smoother_data['B_i_ops']
    S_hat_i_ops = smoother_data['S_hat_i_ops']
    
    m, n = B.shape[0], B.shape[1]
    u_new, p_new = u.copy(), p.copy()

    for _ in range(num_steps):
        # 1. Global Residuals
        res_u = f - (A @ u_new + B.T @ p_new)
        res_p = g - (B @ u_new)
        
        delta_u = np.zeros((n, 1))
        delta_p = np.zeros((m, 1))

        # 2. Local Solves
        for i in range(m):
            P_i, Q_i = P_ops[i], Q_ops[i]
            A_hat_i = A_hat_i_ops[i]
            B_i = B_i_ops[i]
            S_hat_i = S_hat_i_ops[i]
            
            n_i = A_hat_i.shape[0]
            
            # Restrict Residual
            rhs_loc = np.vstack([P_i.T @ res_u, Q_i.T @ res_p])
            
            # Form Local System
            A_inv_loc = np.diag(1.0 / np.diag(A_hat_i))
            S_schur_loc = B_i @ A_inv_loc @ B_i.T - S_hat_i
            K_loc = np.block([[A_hat_i, B_i.T], [B_i, S_schur_loc]])
            
            # Solve
            sol_loc = la.solve(K_loc, rhs_loc)
            
            # Accumulate
            delta_u += P_i @ sol_loc[:n_i]
            delta_p += Q_i @ sol_loc[n_i:]
            
        u_new += delta_u
        p_new += delta_p
        
    return u_new, p_new

# =============================================================================
# 4. RHS HELPERS
# =============================================================================

def gauss_quadrature_2forms(a, b, c, d, N_quad, func):
    """Integrates a function over a 2D rectangle using Gauss quadrature."""
    nodes, w = bf_polynomials.gauss_quad(N_quad)
    val = 0
    for i in range(N_quad):
        for j in range(N_quad):
            x = 0.5 * (b - a) * nodes[i] + 0.5 * (a + b)
            y = 0.5 * (d - c) * nodes[j] + 0.5 * (c + d)
            val += func(x, y) * w[i] * w[j]
    return val * (b-a) * (d-c) / 4.0

def build_rhs_dual(N, quad_degree, func, ref_nodes, x_range, y_range):
    """Constructs the RHS vector for dual variables (cells)."""
    x_min, x_max = x_range
    y_min, y_max = y_range
    
    # Map reference nodes to physical
    x_phys = 0.5 * (x_max - x_min) * ref_nodes + 0.5 * (x_max + x_min)
    y_phys = 0.5 * (y_max - y_min) * ref_nodes + 0.5 * (y_max + y_min)
    
    rhs = np.zeros(N * N)
    for i in range(N):
        for j in range(N):
            x0, x1 = x_phys[i], x_phys[i+1]
            y0, y1 = y_phys[j], y_phys[j+1]
            rhs[i*N + j] = gauss_quadrature_2forms(x0, x1, y0, y1, quad_degree, func)
            
    return rhs

# =============================================================================
# 5. MULTIGRID SOLVERS (CYCLES & DRIVERS)
# =============================================================================

def w_cycle_recursive(level, u, p, f, g, level_data, gamma=2):
    """
    Executes a recursive Multigrid Cycle (V-Cycle if gamma=1, W-Cycle if gamma=2).

    Args:
        level (int): Current level index (0 is coarsest).
        u, p (np.ndarray): Current solution approximations.
        f, g (np.ndarray): RHS components.
        level_data (dict): Hierarchy data structure.
        gamma (int): Cycle index (1 for V, 2 for W).

    Returns:
        tuple: (u_corrected, p_corrected)
    """
    # Base Case: Exact Solve on Coarsest Grid
    if level == 0:
        K_coarse = level_data[level]['K']
        rhs_coarse = np.vstack([f, g])
        # Direct solve on the coarsest level
        e_coarse = la.solve(K_coarse, rhs_coarse)
        n_coarse = level_data[level]['n']
        return e_coarse[:n_coarse].reshape(-1, 1), e_coarse[n_coarse:].reshape(-1, 1)

    # --- Recursive Step ---
    A, B = level_data[level]['A'], level_data[level]['B']
    smoother_data = level_data[level]['smoother_data']
    pre_steps = level_data[level]['pre_steps']
    post_steps = level_data[level]['post_steps']

    # 1. Pre-Smoothing
    u_smooth, p_smooth = apply_schwarz_smoother(
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
    u_final, p_final = apply_schwarz_smoother(
        smoother_data, post_steps, u_corrected, p_corrected, f, g, A, B)
        
    return u_final, p_final


def multigrid_solver(level_data, f_fine, g_fine, cycle_type='W', tol=1e-8, max_iter=50):
    """
    Main driver loop that iterates Multigrid cycles until convergence.

    Args:
        level_data (dict): Pre-computed MG hierarchy.
        f_fine, g_fine (np.ndarray): Fine grid RHS vectors.
        cycle_type (str): 'V' or 'W'.
        tol (float): Relative tolerance for stopping.
        max_iter (int): Maximum number of cycles.

    Returns:
        tuple: (u_solution, p_solution, convergence_rate)
    """
    gamma = 2 if cycle_type.upper() == 'W' else 1
    cycle_func = w_cycle_recursive
    
    finest_level = len(level_data) - 1
    K_fine = level_data[finest_level]['K']
    rhs_fine = np.vstack([f_fine, g_fine])
    n_fine = level_data[finest_level]['n']
    m_fine = level_data[finest_level]['m']
    
    # Initial Guess (Zero)
    u_sol = np.zeros((n_fine, 1))
    p_sol = np.zeros((m_fine, 1))

    # Initial Residual Norm
    initial_res_norm = la.norm(rhs_fine)
    if initial_res_norm < 1e-15: initial_res_norm = 1.0
    
    residual_history = [1.0]

    for i in range(max_iter):
        u_sol, p_sol = cycle_func(
            finest_level, u_sol, p_sol, f_fine, g_fine, level_data, gamma)
        
        # Check Convergence
        current_res = rhs_fine - K_fine @ np.vstack([u_sol, p_sol])
        rel_res = la.norm(current_res) / initial_res_norm
        residual_history.append(rel_res)
        
        if rel_res < tol:
            break
    else:
        print(f"Warning: Multigrid solver did not converge within {max_iter} iterations.")
        
    # Calculate Geometric Mean Convergence Rate
    num_iter = len(residual_history) - 1
    if num_iter > 0:
        conv_rate = (residual_history[-1] / residual_history[0])**(1.0 / num_iter)
    else:
        conv_rate = 0.0
        
    return u_sol, p_sol, conv_rate


def p_multigrid_solver(f_element_vec, p_multigrid_setup, solver_params):
    """
    Solves the single-element system Ax=b using the pre-computed p-multigrid.
    Wrapper for the 'multigrid_solver' function acting as a black-box.

    Args:
        f_element_vec (np.ndarray): Flattened RHS vector [f_u, f_p] for the element.
        p_multigrid_setup (dict): Pre-computed hierarchy from setup.
        solver_params (dict): Options {'cycle_type', 'tol', 'max_iter'}.

    Returns:
        tuple: (u_sol, p_sol, conv_rate)
            - u_sol (np.ndarray): Flux solution column vector.
            - p_sol (np.ndarray): Potential solution column vector.
            - conv_rate (float): Estimated geometric mean convergence rate.
    """
    # 1. Extract Dimensions
    finest_level = len(p_multigrid_setup) - 1
    n_fine = p_multigrid_setup[finest_level]['n']
    
    # 2. Unpack RHS (Assumes stacked vector [u; p])
    f_u = f_element_vec[:n_fine].reshape(-1, 1)
    f_p = f_element_vec[n_fine:].reshape(-1, 1)
    
    # 3. Run Core Multigrid Solver
    # We pass the parameters and capture the convergence rate as requested
    u_sol, p_sol, conv_rate = multigrid_solver(
        level_data=p_multigrid_setup,
        f_fine=f_u,
        g_fine=f_p,
        cycle_type=solver_params.get('cycle_type', 'W'),
        tol=solver_params.get('tol', 1e-9),
        max_iter=solver_params.get('max_iter', 20)
    )
    
    # 4. Return separate components and rate
    return u_sol, p_sol, conv_rate