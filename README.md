================================================
FILE DESCRIPTION
================================================

This repository contains the source code accompanying the Master Thesis.
The code implements a High-Order Mimetic Spectral Element solver for the 
mixed Poisson equation, featuring a custom p-Multigrid preconditioner 
within a Hybridized (Schur Complement) framework. Only the key codes are
uploaded and the plotting routines are skipped most of the times,
but can be provided if requested.

Author: Manuel Fernandez Lopez

--------------------------------------------------------------------------------
1. CORE LIBRARIES
--------------------------------------------------------------------------------

- bf_polynomials.py
  Mathematical backend. Contains definitions for high-order basis functions 
  (Nodal Lagrange, Edge functions) and numerical quadrature rules 
  (Gauss-Legendre-Lobatto, Gauss-Legendre). Authors: Yi & Lorenzo
  Department of Aerodynamics, Faculty of Aerospace Engineering, TU Delft.

- multigrid_utilities.py
  The central engine of the project. Handles the assembly of mimetic system 
  matrices, construction of intergrid transfer operators,
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
  Offline setup script. Pre-calculates setup for the multigrid hierarchy.
  The desired levels to be precomputed should be changed inside the code.

- P_R_tests.py
  Validation script. Computationally verifies the properties
  of the transfer operators 
  
- smoother_theoretical.py
  Theoretical validation script. Checks the conditions
  required for the convergence of the Additive Schwarz
  smoother (Schöberl & Zulehner theory).
--------------------------------------------------------------------------------
4. DATA ARCHIVE (mg_level_data.7z) **IMPORTANT**
--------------------------------------------------------------------------------
  - mg_level_data.7z
  This archive contains the pre-computed '.npy' data files required by the 
  p-multigrid solver and the hybrid solver (system matrices, smoother data, and RHSs for
  various polynomial levels).

  **USAGE:**
  To save time and skip the pre-computation step, extract the 
  contents of this archive into a folder named 'mg_level_data' in the 
  root directory.

  Alternatively, you can generate these files from scratch by running 
  'precompute_cycle_data.py'.

