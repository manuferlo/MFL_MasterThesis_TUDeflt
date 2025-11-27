"""
Theoretical Analysis and Validation of the Additive Schwarz Smoother.

This script performs a rigorous mathematical verification of the overlapping 
Additive Schwarz Preconditioner used as the smoother in the p-Multigrid hierarchy.
It validates the implementation against the convergence theory established by 
Schöberl & Zulehner (2003) for saddle-point problems.

Author: Manuel Fernandez Lopez
Master Thesis - TU Delft
"""


import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import multigrid_utilities as mg_utils
import bf_polynomials

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------
def check_condition(name, condition, error=None):
    """Prints a formatted pass/fail message."""
    status = "PASS" if condition else "FAIL"
    msg = f"{name:.<65} {status}"
    print(msg)

def get_spectral_radius(M):
    """Computes rho(M) = max|lambda(M)|."""
    evals = la.eigvals(M)
    return np.max(np.abs(evals))

# ---------------------------------------------------------
# Core Decomposition Logic
# ---------------------------------------------------------
def build_subspace_operators(m, n, B):
    """
    Constructs the partition of unity operators Q_i and P_i based on 
    the mesh connectivity (B matrix).
    """
    # 1. Q_i Operators (Cell decomposition - Trivial partition)
    # Each Q_i is a unit vector for cell i
    Q_operators = []
    for i in range(m):
        q_i = np.zeros((m, 1))
        q_i[i] = 1.0
        Q_operators.append(q_i)

    # 2. Calculate Overlap Counting (mu)
    # mu[j] = number of cells sharing edge j
    mu = np.zeros(n)
    for j in range(n):
        mu[j] = np.count_nonzero(B[:, j])

    # 3. P_i Operators (Edge decomposition)
    P_hat_operators = [] # Pure restriction (Boolean)
    P_operators = []     # Scaled restriction (Partition of Unity)

    for i in range(m):
        # Identify edges 'j' connected to cell 'i'
        local_u_indices = np.where(B[i, :] != 0)[0]
        n_i = len(local_u_indices)

        # 3a. Unscaled P_hat (Boolean matrix)
        p_hat_i = np.zeros((n, n_i))
        for k, j in enumerate(local_u_indices):
            p_hat_i[j, k] = 1.0
        P_hat_operators.append(p_hat_i)

        # 3b. Scaled P (Partition of Unity)
        # Scaled by 1/sqrt(mu) so that sum(P P^T) = I
        p_i = np.zeros((n, n_i))
        for k, j in enumerate(local_u_indices):
            p_i[j, k] = 1.0 / np.sqrt(mu[j])
        P_operators.append(p_i)

    return Q_operators, P_operators, P_hat_operators, mu

# ---------------------------------------------------------
# Main Verification Script
# ---------------------------------------------------------
def main():
    print("==========================================================")
    print("      Schöberl & Zulehner (2003) Theory Verification      ")
    print("==========================================================\n")

    # 1. Setup Problem 
    N = 16
    quad_degree = 18
    a,b,c,d = -2, 0, 0, 2
    print(f"Initializing MSEM Block (N={N})...")
    
    blk = mg_utils.build_mimetic_block(N, quad_degree, quad_degree, x_range=(a, b), y_range=(c, d))
    A = blk['M1']   # Mass matrix 1-form (A)
    B = blk['E21']  # Divergence (B)
    
    n, m = A.shape[0], B.shape[0] # n=edges, m=cells

    # 2. Build Decomposition Operators
    print("Building subspace operators (P_i, Q_i)...")
    Q_ops, P_ops, P_hat_ops, mu = build_subspace_operators(m, n, B)

    # ---------------------------------------------------------
    # CHECK 1: Partition of Unity
    # ---------------------------------------------------------
    print("\n--- 1. Partition of Unity Check ---")
    
    # Check Sum(Q Q^T) = I
    sum_QQ = sum(q @ q.T for q in Q_ops)
    check_condition("Sum(Q_i Q_i^T) == I", np.allclose(sum_QQ, np.eye(m)))

    # Check Sum(P P^T) = I (Requires 1/sqrt(mu) scaling)
    sum_PP = sum(p @ p.T for p in P_ops)
    check_condition("Sum(P_i P_i^T) == I", np.allclose(sum_PP, np.eye(n)))

    # ---------------------------------------------------------
    # CHECK 2: Operator Construction (A_hat, S_hat)
    # ---------------------------------------------------------
    print("\n--- 2. Construction of Global Preconditioners (A_hat, S_hat) ---")

    # --- Construct A_hat ---
    # Condition: A_hat >= A
    # Strategy: A_hat = (1/sigma) * diag(A)
    d_A = np.diag(A)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_A))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    lambda_max_A = np.max(la.eigvalsh(A_norm))
    
    sigma = 1.0 / ((1 + 1e-6) * lambda_max_A)
    A_hat = (1.0 / sigma) * np.diag(d_A)
    
    print(f"  -> Lambda_max(D^-0.5 A D^-0.5) = {lambda_max_A:.4f}")
    print(f"  -> Chosen sigma = {sigma:.4f}")

    # Check A_hat >= A
    min_eig_diff = np.min(la.eigvalsh(A_hat - A))
    check_condition("Condition (A_hat >= A)", min_eig_diff > -1e-9, min_eig_diff)

    # --- Construct S_hat ---
    # Condition: S_hat > B A_hat^-1 B^T
    # Strategy: S_hat = (1/tau) * D_mu
    # Formula for D_mu diagonal: sum (mu_e * B_ie^2 / A_hat_ee)
    
    d_mu_diag = np.zeros(m)
    A_hat_diag = np.diag(A_hat)
    
    for i in range(m):
        # Identify global edges for cell i
        local_edges = np.where(np.sum(P_hat_ops[i], axis=1) > 0)[0]
        val = 0
        for j in local_edges:
            # Theoretical construction from thesis/paper
            val += mu[j] * (B[i, j]**2) / A_hat_diag[j]
        d_mu_diag[i] = val

    D_mu = np.diag(d_mu_diag)
    
    # Calculate Tau
    A_hat_inv = np.diag(1.0 / np.diag(A_hat))
    S_exact = B @ A_hat_inv @ B.T  # The exact Schur complement of the preconditioner
    
    # Normalize S_exact by D_mu to find max eigenvalue
    D_mu_inv_sqrt = np.diag(1.0 / np.sqrt(d_mu_diag))
    S_norm = D_mu_inv_sqrt @ S_exact @ D_mu_inv_sqrt
    lambda_max_S = np.max(la.eigvalsh(S_norm))
    
    tau = 1.0 / ((1 + 1e-6) * lambda_max_S)
    S_hat = (1.0 / tau) * D_mu

    print(f"  -> Lambda_max(S_norm) = {lambda_max_S:.4f}")
    print(f"  -> Chosen tau = {tau:.4f}")

    # Check S_hat > S_exact
    min_eig_diff_S = np.min(la.eigvalsh(S_hat - S_exact))
    check_condition("Condition (S_hat >= B A_hat^-1 B^T)", min_eig_diff_S > -1e-9, min_eig_diff_S)

    # ---------------------------------------------------------
    # CHECK 3: Commutativity Properties
    # ---------------------------------------------------------
    print("\n--- 3. Commutativity Verification ---")

    # Local Operators Construction
    A_hat_local_ops = [p.T @ A_hat @ p for p in P_hat_ops] # Using P_hat
    
    # Check P_i^T * A_hat == A_hat_i * P_i^T 
    commute_A = True
    for i, p_i in enumerate(P_ops): # Using P scaled
        LHS = p_i.T @ A_hat
        RHS = A_hat_local_ops[i] @ p_i.T
        if not np.allclose(LHS, RHS):
            commute_A = False; break
    check_condition("Commutativity (P_i^T A_hat == A_hat_i P_i^T)", commute_A)

    # Check Q_i^T B == B_i P_i^T
    # Construct B_i local
    B_local_ops = []
    commute_B = True
    for i in range(m):
        q_i = Q_ops[i]; p_hat_i = P_hat_ops[i]; p_i = P_ops[i]
        
        # Construct B_i according to paper
        local_indices = np.where(np.sum(p_hat_i, axis=1) > 0)[0]
        D_mu_local = np.diag(np.sqrt(mu[local_indices]))
        
        b_hat_i = q_i.T @ B @ p_hat_i
        b_i = b_hat_i @ D_mu_local
        B_local_ops.append(b_i)

        if not np.allclose(q_i.T @ B, b_i @ p_i.T):
            commute_B = False; break
            
    check_condition("Commutativity (Q_i^T B == B_i P_i^T)", commute_B)

    # ---------------------------------------------------------
    # CHECK 4: Spectral Convergence Analysis
    # ---------------------------------------------------------
    print("\n--- 4. Spectral Convergence Analysis ---")

    # Construct Global Preconditioner K_hat
    # K_hat = [[A_hat, B^T], [B, B A_hat^-1 B^T - S_hat]]
    K_hat = np.zeros((n + m, n + m))
    
    Schur_complement_K_hat = S_exact - S_hat # B A^-1 B^T - S_hat
    
    K_hat[:n, :n] = A_hat
    K_hat[:n, n:] = B.T
    K_hat[n:, :n] = B
    K_hat[n:, n:] = Schur_complement_K_hat

    # Construct System Matrix K
    K = np.zeros((n + m, n + m))
    K[:n, :n] = A
    K[:n, n:] = B.T
    K[n:, :n] = B
    
    # Calculate Iteration Matrix M = I - K_hat^-1 * K
    I_glob = np.eye(n + m)
    M_iter = I_glob - la.inv(K_hat) @ K
    
    rho = get_spectral_radius(M_iter)
    
    print(f"System Size: {n+m} x {n+m}")
    print(f"Spectral Radius rho(M): {rho:.6f}")
    
    if rho < 1.0:
        print("RESULT: The Additive Schwarz Smoother is CONVERGENT.")
    else:
        print("RESULT: The smoother is DIVERGENT (Check scaling parameters sigma/tau).")

    # ---------------------------------------------------------
    # CHECK 5: Numerical Test (Running the Smoother)
    # ---------------------------------------------------------
    print("\n--- 5. Numerical Running Test (3 Steps) ---")
    
    # Setup Random RHS for testing
    u_sol = np.zeros((n, 1)); p_sol = np.zeros((m, 1))
    rhs_vec = np.random.rand(n + m, 1)
    
    # Initial Residual
    res_0 = la.norm(rhs_vec - K @ np.vstack([u_sol, p_sol]))
    print(f"Initial Residual: {res_0:.4e}")

    # Local S_hat_i (just diagonals of global S_hat)
    S_hat_local_ops = [np.array([[S_hat[i,i]]]) for i in range(m)]

    # Smoother Loop
    for step in range(3):
        # 1. Global Residuals
        r_u = rhs_vec[:n] - (A @ u_sol + B.T @ p_sol)
        r_p = rhs_vec[n:] - (B @ u_sol)

        du = np.zeros_like(u_sol)
        dp = np.zeros_like(p_sol)

        # 2. Local Solves
        for i in range(m):
            # Map global residuals to local
            ru_loc = P_ops[i].T @ r_u
            rp_loc = Q_ops[i].T @ r_p
            
            # Local System K_hat_i
            Ai = A_hat_local_ops[i]
            Bi = B_local_ops[i]
            Si = S_hat_local_ops[i]
            
            Ai_inv = np.diag(1.0/np.diag(Ai))
            S_loc_schur = Bi @ Ai_inv @ Bi.T - Si
            
            ni_loc, mi_loc = Ai.shape[0], Bi.shape[0]
            Ki = np.zeros((ni_loc+mi_loc, ni_loc+mi_loc))
            Ki[:ni_loc, :ni_loc] = Ai
            Ki[:ni_loc, ni_loc:] = Bi.T
            Ki[ni_loc:, :ni_loc] = Bi
            Ki[ni_loc:, ni_loc:] = S_loc_schur
            
            # Solve
            sol_loc = la.solve(Ki, np.vstack([ru_loc, rp_loc]))
            
            # Accumulate
            du += P_ops[i] @ sol_loc[:ni_loc]
            dp += Q_ops[i] @ sol_loc[ni_loc:]

        u_sol += du
        p_sol += dp
        
        res_k = la.norm(rhs_vec - K @ np.vstack([u_sol, p_sol]))
        print(f"Step {step+1}: Residual = {res_k:.4e} (Reduction: {res_k/res_0:.2f})")

if __name__ == "__main__":
    main()

    