"""
Verification of p-Multigrid Intergrid Transfer Operators.

This script mathematically validates the properties of the Prolongation (P) and 
Restriction (R) operators designed for the Mimetic Spectral Element Method.

It performs rigorous checks on:
1. Inverse Property: Verifies if R * P = Identity (essential for p-MG stability).
2. L2-Adjointness: Confirms that R is the discrete L2-adjoint of P.
3. Commutativity: Validates the structure-preserving property, ensuring 
   that the operators commute with the discrete exterior derivative:
   -  E10_fine * P0 = P1 * E10_coarse
   -  E21_fine * P1 = P2 * E21_coarse


Author: Manuel Fernandez Lopez
Master Thesis - TU Delft
"""

import numpy as np
import scipy.linalg
from scipy.linalg import block_diag
import multigrid_utilities as mg_utils
import bf_polynomials


def check_assertion(description, condition, tol=1e-10):
    """
    Helper function to print aligned validation messages.
    """
    status = "PASS" if condition else "FAIL"
    print(f"{description:.<65} {status}")

def get_1d_mass_matrices(N, Nq):
    """
    Computes 1D mass matrices (M0, M1) for L2-projection based restriction.
    """
    # 1. Quadrature and Nodes
    nodes, _ = bf_polynomials.lobatto_quad(N)
    nodes_q, wq = bf_polynomials.gauss_quad(Nq)
    
    # 2. Evaluate Basis at Quadrature Points
    h = bf_polynomials.lagrange_basis(nodes, nodes_q)
    e = bf_polynomials.edge_basis(nodes, nodes_q)
    
    # 3. Apply Integration Weights (Sum-Factorization style preparation)
    # M = B @ diag(w) @ B.T  ->  (B * sqrt(w)) @ (B * sqrt(w)).T
    sqrt_w = np.sqrt(wq)
    h_weighted = h * sqrt_w
    e_weighted = e * sqrt_w
    
    M0_1D = h_weighted @ h_weighted.T
    M1_1D = e_weighted @ e_weighted.T
    
    return M0_1D, M1_1D

def main():
    # ---------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------
    print("Initializing p-Multigrid Operator Verification...")
    Nc = 8   # Coarse p
    Nf = 12   # Fine p
    Nq = 16  # Integration Quadrature
    Np = 12  # Plotting parameter
    
    # ---------------------------------------------------------
    # 2. 1D Operator Construction
    # ---------------------------------------------------------
    # Prolongation (Interpolation)
    I0_1D = mg_utils.prolongate0_1D(Nc, Nf) 
    I1_1D = mg_utils.prolongate1_1D(Nc, Nf)

    # Mass Matrices for L2 Projection
    M0_1D_c, M1_1D_c = get_1d_mass_matrices(Nc, Nq)
    M0_1D_f, M1_1D_f = get_1d_mass_matrices(Nf, Nq)

    # Restriction (L2 Adjoint)
    # R = M_coarse^(-1) * I^T * M_fine
    R0_1D = mg_utils.restrict_L2(I0_1D, M0_1D_f, M0_1D_c)
    R1_1D = mg_utils.restrict_L2(I1_1D, M1_1D_f, M1_1D_c)

    # ---------------------------------------------------------
    # 3. 2D Operator Construction (Tensor Product)
    # ---------------------------------------------------------
    # Build 2D operators using tensor products of 1D ops
    # Returns P^(0), P^(1), P^(2) and corresponding R
    P0, P1, P2, R0, R1, R2 = mg_utils.build_2D_ops(I0_1D, I1_1D, R0_1D, R1_1D)

    # ---------------------------------------------------------
    # 4. Mimetic Blocks (Fine & Coarse) for Commutativity Check
    # ---------------------------------------------------------
    # Fine Grid Block
    blk_f = mg_utils.build_mimetic_block(N=Nf, quad_degree=Nq, plot_degree=Np)
    E21_f, E10_f = blk_f['E21'], blk_f['E10']

    # Coarse Grid Block
    blk_c = mg_utils.build_mimetic_block(N=Nc, quad_degree=Nq, plot_degree=Np)
    E21_c, E10_c = blk_c['E21'], blk_c['E10']

    # ---------------------------------------------------------
    # 5. Verification Report
    # ---------------------------------------------------------
    print("\n--- 1D Operator Properties ---")
    # Property: R*P = I in 1D
    check_assertion("1D R0 @ P0 == Identity", np.allclose(R0_1D @ I0_1D, np.eye(I0_1D.shape[1])))
    check_assertion("1D R1 @ P1 == Identity", np.allclose(R1_1D @ I1_1D, np.eye(I1_1D.shape[1])))
    
    # Property: L2 Adjoint Definition
    # <P u_c, v_f>_fine = <u_c, R v_f>_coarse
    check_assertion("1D Adjoint Property (0-forms): Mc * R0 == P0.T * Mf", 
                    np.allclose(M0_1D_c @ R0_1D, I0_1D.T @ M0_1D_f))
    check_assertion("1D Adjoint Property (1-forms): Mc * R1 == P1.T * Mf", 
                    np.allclose(M1_1D_c @ R1_1D, I1_1D.T @ M1_1D_f))

    print("\n--- 2D Operator Properties ---")
    
    # Property: R*P = I in 2D
    check_assertion("2D R0 @ P0 == Identity", np.allclose(R0 @ P0, np.eye(P0.shape[1])))
    check_assertion("2D R1 @ P1 == Identity", np.allclose(R1 @ P1, np.eye(P1.shape[1])))
    check_assertion("2D R2 @ P2 == Identity", np.allclose(R2 @ P2, np.eye(P2.shape[1])))

    print("\n--- Structure Preservation (Commutativity) ---")
    
    # Property: Commuting Diagram
   
    
    # E10_f * P0 == P1 * E10_c
    commutativity_grad = np.allclose(E10_f @ P0, P1 @ E10_c)
    check_assertion("(E10 * P0 == P1 * E10)", commutativity_grad)
    
    # E21_f * P1 == P2 * E21_c
    commutativity_div = np.allclose(E21_f @ P1, P2 @ E21_c)
    check_assertion("(E21 * P1 == P2 * E21)", commutativity_div)
    
    # 3. Galerkin Coarse Grid Operator Consistency
    # R2 * E21_f * P1 == E21_c ? 
    galerkin_check = np.allclose(R2 @ E21_f @ P1, E21_c)
    check_assertion("(R2 * E21_f * P1 == E21_c)", galerkin_check)

    print("\nVerification Complete.")

if __name__ == "__main__":
    main()

