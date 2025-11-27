"""
Single-Element Mimetic Spectral Element Solver.

This script validates the core MSEM discretization on a single high-order element.
It solves the mixed Poisson problem using the mimetic framework, verifying
convergence rates and the structure-preserving properties (commuting diagram).

Key functionalities:
- Construction of high-order 0-form (nodal), 1-form (edge), and 2-form (cell) bases.
- Assembly of Mass (M) and Incidence (E) matrices.
- Solution of the indefinite saddle-point system via direct solver.

Author: Manuel Fernandez Lopez
Master Thesis - TU Delft
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import bf_polynomials  # Custom library

# Set plotting style for thesis-quality figures
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'

def build_mimetic_block(N, quad_degree, plot_degree, x_range=(-1., 1.), y_range=(-1., 1.)):
    """
    Constructs all necessary matrices (Mass, Incidence) and basis functions 
    for a single Mimetic Spectral Element.
    """
    
    # 1. GLL Nodes
    nodes, _ = bf_polynomials.lobatto_quad(N)

    # ---------------------------
    # Quadrature Setup
    # ---------------------------
    q_nodes, q_weights = bf_polynomials.gauss_quad(quad_degree)
    q_weights_2D = np.outer(q_weights, q_weights).flatten()
    num_q = quad_degree**2

    # Scalar Bases Evaluation
    h_x = bf_polynomials.lagrange_basis(nodes, q_nodes)
    h_y = bf_polynomials.lagrange_basis(nodes, q_nodes)

    # Edge Bases Evaluation
    e_x = bf_polynomials.edge_basis(nodes, q_nodes)
    e_y = bf_polynomials.edge_basis(nodes, q_nodes)

    # Geometric Factors (Affine Mapping)
    a, b = x_range;  c, d = y_range
    dx, dy  = (b-a)/2.0, (d-c)/2.0          # Scaling factors
    detJ    = dx * dy                       # Jacobian determinant
    ax, ay  = dy/dx, dx/dy                  # Metric terms

    # Degrees of Freedom Counts
    num_dofs_0 = (N + 1)**2
    num_dofs_1 = 2 * N * (N + 1)
    num_dofs_2 = N**2

    # ---------------------------
    # psi^0 (0-forms) at quadrature points
    # ---------------------------
    psi0 = np.zeros((num_dofs_0, num_q))
    for i in range(N + 1):
        for j in range(N + 1):
            idx = i * (N + 1) + j
            # Tensor product construction
            for qx in range(quad_degree):
                for qy in range(quad_degree):
                    q = qx * quad_degree + qy
                    psi0[idx, q] = h_x[i, qx] * h_y[j, qy]

    # ---------------------------
    # psi^1 (1-forms) at quadrature points
    # ---------------------------
    psi1 = np.zeros((num_dofs_1, num_q, 2))
    count = 0
    # y-edges
    for i in range(N + 1):
        for j in range(1, N + 1):
            for qx in range(quad_degree):
                for qy in range(quad_degree):
                    q = qx * quad_degree + qy
                    psi1[count, q, 0] = h_x[i, qx] * e_y[j - 1, qy]
            count += 1
    # x-edges
    for i in range(1, N + 1):
        for j in range(N + 1):
            for qx in range(quad_degree):
                for qy in range(quad_degree):
                    q = qx * quad_degree + qy
                    psi1[count, q, 1] = e_x[i - 1, qx] * h_y[j, qy]
            count += 1

    # ---------------------------
    # psi^2 (2-forms) at quadrature points
    # ---------------------------
    psi2 = np.zeros((num_dofs_2, num_q))
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            for qx in range(quad_degree):
                for qy in range(quad_degree):
                    q = qx * quad_degree + qy
                    psi2[idx, q] = e_x[i, qx] * e_y[j, qy]

    # ---------------------------
    # Mass Matrices Assembly
    # ---------------------------
    # M0
    M0 = psi0 @ np.diag(q_weights_2D) @ psi0.T
    M0 = detJ * M0

    # M1 (Weighted by metric terms)
    M1 = np.zeros((num_dofs_1, num_dofs_1))              
    for i in range(num_dofs_1):
        for j in range(num_dofs_1):
            s  = ay * psi1[i,:,0] * psi1[j,:,0]
            s += ax * psi1[i,:,1] * psi1[j,:,1]
            M1[i,j] = np.sum(s * q_weights_2D)

    # M2
    M2 = psi2 @ np.diag(q_weights_2D) @ psi2.T
    M2 = (1.0/detJ) * M2
    M2_inv = np.linalg.inv(M2)

    # Dual 0-form basis (Projection)
    psi0_dual = M2_inv @ psi2  

    # ---------------------------
    # Incidence Matrices (E10 and E21)
    # ---------------------------
    num_nodes = (N + 1)**2
    num_edges = 2 * N * (N + 1)
    num_cells = N * N

    # E10
    E10 = np.zeros((num_edges, num_nodes), dtype=int)
    edge_id = 0
    # Vertical edges
    for i in range(N + 1):
        for j in range(1, N + 1):
            s = i * (N + 1) + (j - 1)
            e = i * (N + 1) + j
            E10[edge_id, s] = -1; E10[edge_id, e] = +1
            edge_id += 1
    # Horizontal edges
    for i in range(1, N + 1):
        for j in range(N + 1):
            s = (i - 1) * (N + 1) + j
            e = i * (N + 1) + j
            E10[edge_id, s] = -1; E10[edge_id, e] = +1
            edge_id += 1

    # E21
    E21 = np.zeros((num_cells, num_edges), dtype=int)
    for i in range(N):
        for j in range(N):
            cid = i * N + j
            # Map local edges to global indices
            b = N * (N + 1) + i * (N + 1) + j
            r = (i + 1) * N + j
            t = N * (N + 1) + i * (N + 1) + (j + 1)
            l = i * N + j
            
            E21[cid, b] = -1
            E21[cid, r] = +1
            E21[cid, t] = +1
            E21[cid, l] = -1

    # ---------------------------
    # Plotting / Reconstruction Data
    # ---------------------------
    plot_nodes, _ = bf_polynomials.gauss_quad(plot_degree)
    num_q_plot = plot_degree**2

    h_x_plot = bf_polynomials.lagrange_basis(nodes, plot_nodes)
    h_y_plot = bf_polynomials.lagrange_basis(nodes, plot_nodes)
    e_x_plot = bf_polynomials.edge_basis(nodes, plot_nodes)
    e_y_plot = bf_polynomials.edge_basis(nodes, plot_nodes)

    # psi0 for plotting
    psi0_plot = np.zeros((num_dofs_0, num_q_plot))
    for i in range(N + 1):
        for j in range(N + 1):
            idx = i * (N + 1) + j
            for qx in range(plot_degree):
                for qy in range(plot_degree):
                    q = qx * plot_degree + qy
                    psi0_plot[idx, q] = h_x_plot[i, qx] * h_y_plot[j, qy]

    # psi1 for plotting
    psi1_plot = np.zeros((num_dofs_1, num_q_plot, 2))
    count = 0
    for i in range(N + 1):
        for j in range(1, N + 1):
            for qx in range(plot_degree):
                for qy in range(plot_degree):
                    q = qx * plot_degree + qy
                    psi1_plot[count, q, 0] = h_x_plot[i, qx] * e_y_plot[j - 1, qy]
            count += 1
    for i in range(1, N + 1):
        for j in range(N + 1):
            for qx in range(plot_degree):
                for qy in range(plot_degree):
                    q = qx * plot_degree + qy
                    psi1_plot[count, q, 1] = e_x_plot[i - 1, qx] * h_y_plot[j, qy]
            count += 1

    # psi2 for plotting
    psi2_plot = np.zeros((num_dofs_2, num_q_plot))
    for i in range(N):
        for j in range(N):
            idx = i * N + j
            for qx in range(plot_degree):
                for qy in range(plot_degree):
                    q = qx * plot_degree + qy
                    psi2_plot[idx, q] = e_x_plot[i, qx] * e_y_plot[j, qy]

    psi0_dual_plot = M2_inv @ psi2_plot

    return {
        'psi0':psi0, 'psi1':psi1, 'psi2':psi2,
        'psi0_plot':psi0_plot, 'psi1_plot':psi1_plot, 'psi2_plot':psi2_plot,
        'psi0_dual'     : psi0_dual,
        'psi0_dual_plot': psi0_dual_plot,
        'M0':M0, 'M1':M1, 'M2':M2, 'M2_inv': M2_inv,
        'E10':E10, 'E21':E21,
        'detJ':detJ, 'dx':dx, 'dy':dy,
        'w2D': q_weights_2D, 
        'x_range':x_range, 'y_range':y_range
    }

def gauss_quadrature_2forms(a,b,c,d,N_quad, f):
    nodes_q, w = bf_polynomials.gauss_quad(N_quad)
    val = 0
    for i in range(N_quad):
        for j in range(N_quad):
            x = 0.5 * (b - a) * nodes_q[i] + 0.5 * (a + b)
            y = 0.5 * (d - c) * nodes_q[j] + 0.5 * (c + d)
            val += f(x, y) * w[i] * w[j]
            
    return val * (b-a) * (d-c) / 4.0

def build_rhs_dual(N, quad_degree, form_fun, ref_nodes, x_range, y_range):
    """
    Integrates the forcing function over each cell to build the dual RHS.
    """
    x_min, x_max = x_range
    y_min, y_max = y_range

    # Map reference nodes to physical space
    x_phys = 0.5 * (x_max - x_min) * ref_nodes + 0.5 * (x_max + x_min)
    y_phys = 0.5 * (y_max - y_min) * ref_nodes + 0.5 * (y_max + y_min)

    rhs = np.zeros(N * N)  # One DoF per 2-form cell

    for i in range(N):
        for j in range(N):
            x0, x1 = x_phys[i], x_phys[i+1]
            y0, y1 = y_phys[j], y_phys[j+1]
            integral = gauss_quadrature_2forms(x0, x1, y0, y1, quad_degree, form_fun)
            rhs[i * N + j] = integral
    return rhs

def reconstruct_solution(u_coeffs, phi_coeffs, blk):
    """
    Reconstructs the solution on a fine grid for plotting/error analysis.
    Returns: x, y meshes and fields phi_h, ux_h, uy_h.
    """
    # Number of plot points per cell
    num_q_plot = blk['psi2_plot'].shape[1]
    plot_deg   = int(np.sqrt(num_q_plot))

    # Plot nodes
    plot_nodes = bf_polynomials.gauss_quad(plot_deg)[0]
    xi, eta    = np.meshgrid(plot_nodes, plot_nodes, indexing='ij')

    # Physical coordinates
    a, b = blk['x_range'];  c, d = blk['y_range']
    dx, dy = blk['dx'], blk['dy']
    detJ   = blk['detJ']
    x = a + dx*(xi + 1.0)
    y = c + dy*(eta + 1.0)

    # Vector reconstruction
    phi_vec = phi_coeffs @ blk['psi0_dual_plot']
    ux_vec  = u_coeffs   @ blk['psi1_plot'][:,:,0]
    uy_vec  = u_coeffs   @ blk['psi1_plot'][:,:,1]

    # Reshape and Scale by Jacobian
    phi_h = phi_vec.reshape(plot_deg, plot_deg) / detJ
    ux_h  = ux_vec.reshape (plot_deg, plot_deg) / dy
    uy_h  = uy_vec.reshape (plot_deg, plot_deg) / dx

    return x, y, phi_h, ux_h, uy_h

def triple_plot(x, y, exact, approx, labels, fname=None):
    """
    Generates a 3-panel plot: Exact | Approximate | Error.
    """
    err = approx - exact
    data_list = [exact, approx, err]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), constrained_layout=True, dpi=180)
    for ax, data, lbl in zip(axes, data_list, labels):
        cmap = 'seismic' if 'Error' in lbl else 'coolwarm'
        # Shading gouraud gives a smoother look for spectral methods
        h = ax.pcolormesh(x, y, data, shading='gouraud', cmap=cmap)
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')
        ax.set_title(lbl, fontsize=14, pad=10)
        fig.colorbar(h, ax=ax, fraction=0.046, pad=0.04)

    if fname:
        plt.savefig(fname, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# ---------------------------------------------------------
# Analytical Solutions
# ---------------------------------------------------------
def exact_phi(x, y): 
    return np.sin(np.pi * x) * np.sin(np.pi * y)

def exact_u(x, y):  
    # Returns (u_x, u_y)
    ux = np.pi * np.cos(np.pi * x) * np.sin(np.pi * y)
    uy = np.pi * np.sin(np.pi * x) * np.cos(np.pi * y)
    return ux, uy

def forcing_f(x, y): 
    return -2 * (np.pi**2) * np.sin(np.pi*x) * np.sin(np.pi*y)

def calculate_errors(e_vals, w2D, detJ):
    """Weighted L2 norm helper."""
    e2 = np.asarray(e_vals, float)**2
    W  = (np.asarray(w2D, float).ravel() * np.asarray(detJ, float).ravel())
    return np.sqrt(np.dot(e2.ravel(), W))

# ---------------------------------------------------------
# Main Execution Block
# ---------------------------------------------------------
def main():
    # Parameters
    N = 12
    quad_degree = 44
    a, b, c, d = -2, 0, 0, 2  # Physical domain limits

    print(f"Initializing MSEM Block (N={N})...")
    blk = build_mimetic_block(N, quad_degree, quad_degree,
                              x_range=(a, b), y_range=(c, d))

    # Unpack necessary matrices
    M1, M2, E21 = blk['M1'], blk['M2'], blk['E21']
    detJ, w2D   = blk['detJ'], blk['w2D']

    # ---------------------------
    # Assemble Linear System (Saddle Point)
    # [ M1   E21^T ] [ u ] = [ 0 ]
    # [ E21    0   ] [ p ] = [ f ]
    # ---------------------------
    n1, n2 = M1.shape[0], M2.shape[0]
    A = np.zeros((n1+n2, n1+n2))
    A[:n1, :n1]   = M1
    A[:n1, n1:]   = E21.T
    A[n1:, :n1]   = E21      

    # Build RHS
    nodes, _ = bf_polynomials.lobatto_quad(N)
    F = build_rhs_dual(N, quad_degree, forcing_f, nodes, x_range=(a, b), y_range=(c, d))
    rhs = np.hstack((np.zeros(n1), F))

    # Solve System
    print("Solving system...")
    coeffs = np.linalg.solve(A, rhs)
    u_coeffs   = coeffs[:n1]
    phi_coeffs = coeffs[n1:]

    # ---------------------------
    # Reconstruction & Plotting
    # ---------------------------
    x, y, phi_h, ux_h, uy_h = reconstruct_solution(u_coeffs, phi_coeffs, blk)

    phi_ex = exact_phi(x, y)
    ux_ex, uy_ex = exact_u(x, y)

    # Plot phi
    labels_phi = [
        r'\textbf{Exact}: $\widetilde{\phi}^{(0)}$',
        r'\textbf{Reconstructed}: $\widetilde{\phi}^h$',
        r'\textbf{Error}: $\widetilde{\phi}^h - \widetilde{\phi}^{(0)}$'
    ]
    triple_plot(x, y, phi_ex, phi_h, labels_phi, fname='phi_result.png')

    # Plot Fluxes (qx, qy)
    labels_qx = [r'\textbf{Exact}: $q_x$', r'\textbf{Numeric}: $q_x^h$', r'\textbf{Error}']
    triple_plot(x, y, ux_ex, ux_h, labels_qx, fname='qx_result.png')
    
    labels_qy = [r'\textbf{Exact}: $q_y$', r'\textbf{Numeric}: $q_y^h$', r'\textbf{Error}']
    triple_plot(x, y, uy_ex, uy_h, labels_qy, fname='qy_result.png')

    # ---------------------------
    # Error Analysis
    # ---------------------------
    e_phi = (phi_h - phi_ex)
    e_qx  = (ux_h  - ux_ex)
    e_qy  = (uy_h  - uy_ex)

    norm_phi_err = calculate_errors(e_phi, w2D, detJ)
    norm_phi_ex  = calculate_errors(phi_ex, w2D, detJ)
    
    norm_qx_err = calculate_errors(e_qx, w2D, detJ)
    norm_qy_err = calculate_errors(e_qy, w2D, detJ)
    
    # Combined flux error
    # Note: Correct way to sum L2 norms of vector components
    norm_q_err = np.sqrt(norm_qx_err**2 + norm_qy_err**2)
    norm_q_ex  = calculate_errors(np.sqrt(ux_ex**2 + uy_ex**2), w2D, detJ)

    print("\n--- Convergence Results ---")
    print(f"L2 Error (phi): {norm_phi_err:.4e} (Rel: {norm_phi_err/norm_phi_ex:.4e})")
    print(f"L2 Error (q):   {norm_q_err:.4e}   (Rel: {norm_q_err/norm_q_ex:.4e})")

    # ---------------------------
    # Verification of H(div) Norm
    # ---------------------------
    # Checks if discrete divergence error is negligible (Commuting property)
    div_qh_coeffs = E21 @ u_coeffs
    div_error_coeffs = div_qh_coeffs - F
    div_error_L2 = np.sqrt(div_error_coeffs.T @ M2 @ div_error_coeffs)
    H_div_error = np.sqrt(norm_q_err**2 + div_error_L2**2)

    print("\n--- H(div) Verification ---")
    print(f"L2 Divergence Error: {div_error_L2:.4e}")

if __name__ == "__main__":
    main()

