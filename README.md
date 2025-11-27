================================================================================
FILE DESCRIPTION
================================================================================

This repository contains the source code accompanying the Master Thesis. 
The code implements a High-Order Mimetic Spectral Element solver for the 
mixed Poisson equation, featuring a custom p-Multigrid preconditioner 
within a Hybridized (Schur Complement) framework.

Author: Manuel Fernandez Lopez

--------------------------------------------------------------------------------
1. CORE LIBRARIES
--------------------------------------------------------------------------------

- bf_polynomials.py
  Mathematical backend. Contains definitions for high-order basis functions 
  (Nodal Lagrange, Edge functions) and numerical quadrature rules 
  (Gauss-Legendre-Lobatto, Gauss-Legendre).

- multigrid_utilities.py
  The central engine of the project. Handles the assembly of mimetic system 
  matrices (M, E), construction of intergrid transfer operators (P, R),
  the setup of the Additive Schwarz smoother and the multigrid cycle.

--------------------------------------------------------------------------------
2. SOLVERS (MAIN DRIVERS)
--------------------------------------------------------------------------------

- MSEM_Poisson2D_1element.py
  Benchmark script for a single spectral element. Validates the discretization 
  error convergence (p-refinement) using a direct solver.

- multigrid_solver.py
  Implements the p-Multigrid solver (V-Cycle/W-Cycle). This script runs 
  the iterative solver for a single element.

- hybrid_solver.py
  The full multi-element solver. Implements the Hybridization strategy using 
  Conjugate Gradient on the Schur Complement (outer solver) combined with 
  the p-Multigrid method (inner solver).

- hybrid_solver_RITZ.py
  An instrumented version of the hybrid solver that tracks Ritz values
  during the CG iterations to analyze spectral 
  clustering and convergence behavior.

--------------------------------------------------------------------------------
3. SETUP & VALIDATION
--------------------------------------------------------------------------------

- precompute_cycle_data.py
  Offline setup script. Pre-calculates setup for the multigrid hierarchy to 
  accelerate runtime performance.

- P_R_tests.py
  Validation script. Computationally verifies that the properties
  of the transfer operators 
  

- smoother_theoretical.py
  Theoretical validation script. Checks the conditions
  required for the convergence of the Additive Schwarz
  smoother (Schöberl & Zulehner theory).

