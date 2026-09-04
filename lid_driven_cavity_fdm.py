"""
Lid-Driven Cavity Flow Solver (2D, Incompressible)

Numerical methods:
- Streamfunction–Vorticity formulation (psi–omega)
- High-Performance Alternating Direction Implicit (ADI) scheme with exact tridiagonal boundary closures
- Direct Precomputed Sparse LU Decomposition for Streamfunction Poisson equation (exact to machine precision)
  (with optional Vectorized Red–Black SOR fallback)
- Dual Lid Profiles:
  * 'constant': Standard benchmark lid U(x, 1) = U (Ghia et al., 1982)
  * 'regularized': Singularity-free polynomial profile u(x, 1) = 16*U*(x/L)^2*(1 - x/L)^2
- Physical coordinate system with instant post-processing and Ghia et al. validation

Author: Kartikey Singh
Year: 2026
License: MIT
"""

import os
import sys
import io
import time
import pickle
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# ---- NumPy pickle compatibility patch ----
try:
    if hasattr(np, '_core') and hasattr(np._core, 'numeric'):
        sys.modules['numpy._core.numeric'] = np._core.numeric
    elif hasattr(np, 'core') and hasattr(np.core, 'numeric'):
        sys.modules['numpy._core.numeric'] = np.core.numeric
except Exception:
    pass

# UNICODE FIX FOR TERMINALS
if hasattr(sys, 'stdout') and hasattr(sys.stdout, 'buffer'):
    try:
        current_encoding = getattr(sys.stdout, 'encoding', None)
        if current_encoding and current_encoding.lower() != 'utf-8':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass


# ============================================================================
# BENCHMARK DATA: Ghia, Ghia & Shin (1982)
# ============================================================================
GHIA_DATA = {
    'y_u': np.array([1.0000, 0.9766, 0.9688, 0.9609, 0.9531, 0.8516, 0.7344, 0.6172,
                     0.5000, 0.4531, 0.2813, 0.1719, 0.1016, 0.0703, 0.0625, 0.0547, 0.0000]),
    'u_Re100': np.array([1.00000, 0.84123, 0.78871, 0.73722, 0.68717, 0.23151, 0.00332, -0.13641,
                         -0.20581, -0.21090, -0.15662, -0.10150, -0.06434, -0.04775, -0.04192, -0.03717, 0.00000]),
    'u_Re1000': np.array([1.00000, 0.65928, 0.57492, 0.51117, 0.46604, 0.33304, 0.18719, 0.05702,
                          -0.06080, -0.10648, -0.27805, -0.38289, -0.29730, -0.22220, -0.20196, -0.18109, 0.00000]),
    'u_Re3200': np.array([1.00000, 0.53236, 0.48296, 0.46547, 0.46101, 0.34682, 0.19791, 0.07156,
                          -0.05702, -0.10656, -0.24427, -0.34314, -0.41933, -0.42768, -0.41528, -0.39269, 0.00000]),
    'u_Re5000': np.array([1.00000, 0.48223, 0.46120, 0.45992, 0.46036, 0.33556, 0.19456, 0.07283,
                          -0.04302, -0.09140, -0.22445, -0.30836, -0.40435, -0.43501, -0.43643, -0.42878, 0.00000]),
    'x_v': np.array([1.0000, 0.9688, 0.9609, 0.9531, 0.9453, 0.9063, 0.8594, 0.8047,
                     0.5000, 0.2344, 0.2266, 0.1563, 0.0938, 0.0781, 0.0703, 0.0625, 0.0000]),
    'v_Re100': np.array([0.00000, -0.05906, -0.07391, -0.08864, -0.10313, -0.16914, -0.22445, -0.24533,
                         0.05454, 0.17527, 0.17507, 0.16077, 0.12317, 0.10890, 0.10090, 0.09233, 0.00000]),
    'v_Re1000': np.array([0.00000, -0.21388, -0.27669, -0.32079, -0.34228, -0.42665, -0.51692, -0.38598,
                          0.02526, 0.32627, 0.33304, 0.37095, 0.32627, 0.29012, 0.27485, 0.25686, 0.00000]),
    'v_Re3200': np.array([0.00000, -0.32407, -0.38389, -0.41920, -0.43590, -0.47820, -0.52357, -0.54053,
                          0.00945, 0.38324, 0.39088, 0.41496, 0.37801, 0.34184, 0.32622, 0.30690, 0.00000]),
    'v_Re5000': np.array([0.00000, -0.38458, -0.43448, -0.45543, -0.46387, -0.49099, -0.52987, -0.55408,
                          0.00838, 0.40028, 0.40797, 0.42951, 0.39276, 0.35414, 0.33784, 0.31818, 0.00000]),
}


class LidDrivenCavitySolver:
    """
    Solves 2D incompressible lid-driven cavity flow using the streamfunction-vorticity (psi-omega) method.
    
    Features:
    - Precomputed direct sparse LU Poisson solver (O(ms) per step, machine precision).
    - Fully vectorized ADI vorticity transport with exact tridiagonal boundary closures.
    - Support for constant (U=1) or regularized singularity-free (16*x^2*(1-x)^2) lid profiles.
    - Automated pressure Poisson solver with Neumann zero-gradient boundary conditions.
    """

    def __init__(self, N=129, Re=1000, lid_velocity=1.0, L=1.0, lid_profile='constant', poisson_solver='lu'):
        """
        Parameters:
        -----------
        N : int
            Number of grid points along each axis (N x N mesh).
        Re : float
            Reynolds number (Re = U * L / nu).
        lid_velocity : float
            Peak velocity of the moving lid.
        L : float
            Cavity dimension (domain [0, L] x [0, L]).
        lid_profile : str
            'constant': classic benchmark U(x, 1) = U.
            'regularized': smooth profile u(x, 1) = 16*U*(x/L)^2*(1 - x/L)^2.
        poisson_solver : str
            'lu': direct sparse LU factorized solve (fastest, exact).
            'sor': vectorized Red-Black Successive Over-Relaxation.
        """
        self.N = N
        self.M = N - 2
        self.Re = Re
        self.U = lid_velocity
        self.L = L
        self.lid_profile = lid_profile.lower()
        self.poisson_solver = poisson_solver.lower()

        # Grid parameters
        self.h = L / (N - 1)
        self.nu = self.U * self.L / self.Re

        # Grid coordinates
        self.x = np.linspace(0, L, N)
        self.y = np.linspace(0, L, N)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing='xy')

        # Lid boundary profile
        x_norm = self.x / self.L
        if self.lid_profile == 'regularized':
            self.u_lid = 16.0 * self.U * (x_norm**2) * ((1.0 - x_norm)**2)
        else:
            self.u_lid = np.full(N, self.U)

        # Flow fields (array indexing: [y_idx, x_idx])
        self.psi = np.zeros((N, N))
        self.omega = np.zeros((N, N))
        self.u = np.zeros((N, N))
        self.v = np.zeros((N, N))
        self.p = np.zeros((N, N))

        # Time step (CFL condition for explicit upwind convection)
        cfl_factor = 0.2 if self.poisson_solver == 'lu' else 0.1
        self.dt = cfl_factor * self.h / (self.U if self.U > 0 else 1.0)
        self.alpha_adi = (self.nu * self.dt) / (2.0 * self.h**2)

        print(f"[INFO] Mesh: {N}x{N} | Re: {Re} | Lid: '{self.lid_profile}' | Poisson: '{self.poisson_solver}'")
        print(f"[INFO] h = {self.h:.4e} | dt = {self.dt:.4e} | nu = {self.nu:.4e} | alpha_adi = {self.alpha_adi:.4e}")

        # Interior velocities for ADI convection
        self.u_c = np.zeros((self.M, self.M))
        self.v_c = np.zeros((self.M, self.M))

        # Convergence tracking
        self.history = {'iterations': [], 'max_change': [], 'psi_min': []}

        # Initialize Poisson Solver
        if self.poisson_solver == 'lu':
            self._init_sparse_lu_poisson()
        else:
            self._init_sor_masks()

        # Precompute 1D Thomas recurrence coefficients for ADI
        self._init_adi_coefficients()

    def _init_sparse_lu_poisson(self):
        """Precompute sparse LU factorization of the 2D Laplacian operator."""
        M = self.M
        h2 = self.h**2
        main_diag = -2.0 * np.ones(M) / h2
        off_diag = 1.0 * np.ones(M - 1) / h2
        T = sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(M, M))
        I = sp.eye(M)
        L_2d = (sp.kron(I, T) + sp.kron(T, I)).tocsc()
        self.lu_poisson = spla.splu(L_2d)

    def _init_sor_masks(self):
        """Create checkerboard masks for Red-Black SOR."""
        N = self.N
        i_vals, j_vals = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
        self.red_interior_mask = ((i_vals + j_vals) % 2 == 0) & (i_vals > 0) & (i_vals < N-1) & (j_vals > 0) & (j_vals < N-1)
        self.black_interior_mask = ~self.red_interior_mask & (i_vals > 0) & (i_vals < N-1) & (j_vals > 0) & (j_vals < N-1)

    def _init_adi_coefficients(self):
        """Precompute recurrence factors for vectorized tridiagonal solver."""
        M = self.M
        alpha = self.alpha_adi
        A = -alpha
        B = 1.0 + 2.0 * alpha
        C = -alpha
        self.A_adi, self.B_adi, self.C_adi = A, B, C

        self.P_adi = np.zeros(M)
        self.denom_adi = np.zeros(M)
        self.denom_adi[0] = B
        self.P_adi[0] = C / B
        for j in range(1, M):
            self.denom_adi[j] = B - A * self.P_adi[j - 1]
            self.P_adi[j] = C / self.denom_adi[j]

    def apply_boundary_conditions(self):
        """Apply boundary conditions with Thom's formula for wall vorticity."""
        h = self.h

        # Streamfunction BCs: all walls are no-slip impermeable (psi = 0)
        self.psi[0, :] = 0.0
        self.psi[-1, :] = 0.0
        self.psi[:, 0] = 0.0
        self.psi[:, -1] = 0.0

        # Vorticity BCs (Thom's first-order formula):
        # Top moving lid (y = L, row index = N-1)
        self.omega[-1, 1:-1] = -2.0 * self.psi[-2, 1:-1] / (h**2) - 2.0 * self.u_lid[1:-1] / h
        # Bottom stationary wall (y = 0, row index = 0)
        self.omega[0, 1:-1] = -2.0 * self.psi[1, 1:-1] / (h**2)
        # Left stationary wall (x = 0, col index = 0)
        self.omega[1:-1, 0] = -2.0 * self.psi[1:-1, 1] / (h**2)
        # Right stationary wall (x = L, col index = N-1)
        self.omega[1:-1, -1] = -2.0 * self.psi[1:-1, -2] / (h**2)

        # Corner vorticity closure: average adjacent wall points
        self.omega[0, 0] = 0.5 * (self.omega[1, 0] + self.omega[0, 1])
        self.omega[0, -1] = 0.5 * (self.omega[1, -1] + self.omega[0, -2])
        self.omega[-1, 0] = 0.5 * (self.omega[-2, 0] + self.omega[-1, 1])
        self.omega[-1, -1] = 0.5 * (self.omega[-2, -1] + self.omega[-1, -2])

    def solve_streamfunction(self, max_iterations=1000, tolerance=1e-5, omega_relaxation=1.8):
        """Solve Poisson equation del^2(psi) = -omega."""
        if self.poisson_solver == 'lu':
            rhs = -self.omega[1:-1, 1:-1].ravel()
            psi_inner = self.lu_poisson.solve(rhs).reshape((self.M, self.M))
            max_change = np.max(np.abs(self.psi[1:-1, 1:-1] - psi_inner))
            self.psi[1:-1, 1:-1] = psi_inner
            return max_change
        else:
            # Fallback Red-Black SOR
            source_term = (self.h**2) * self.omega
            for iteration in range(max_iterations):
                psi_old = self.psi.copy()
                psi_new_red = 0.25 * (
                    np.roll(self.psi, 1, axis=0)[self.red_interior_mask] +
                    np.roll(self.psi, -1, axis=0)[self.red_interior_mask] +
                    np.roll(self.psi, 1, axis=1)[self.red_interior_mask] +
                    np.roll(self.psi, -1, axis=1)[self.red_interior_mask] +
                    source_term[self.red_interior_mask]
                )
                self.psi[self.red_interior_mask] = (1 - omega_relaxation) * self.psi[self.red_interior_mask] + omega_relaxation * psi_new_red

                psi_new_black = 0.25 * (
                    np.roll(self.psi, 1, axis=0)[self.black_interior_mask] +
                    np.roll(self.psi, -1, axis=0)[self.black_interior_mask] +
                    np.roll(self.psi, 1, axis=1)[self.black_interior_mask] +
                    np.roll(self.psi, -1, axis=1)[self.black_interior_mask] +
                    source_term[self.black_interior_mask]
                )
                self.psi[self.black_interior_mask] = (1 - omega_relaxation) * self.psi[self.black_interior_mask] + omega_relaxation * psi_new_black

                max_change = np.max(np.abs(self.psi - psi_old))
                if max_change < tolerance:
                    break
            return max_change

    def calculate_velocities(self):
        """Calculate velocities u = d(psi)/dy and v = -d(psi)/dx."""
        h = self.h
        self.u[1:-1, 1:-1] = (self.psi[2:, 1:-1] - self.psi[:-2, 1:-1]) / (2.0 * h)
        self.v[1:-1, 1:-1] = -(self.psi[1:-1, 2:] - self.psi[1:-1, :-2]) / (2.0 * h)

        self.u_c = self.u[1:-1, 1:-1]
        self.v_c = self.v[1:-1, 1:-1]

        # Apply boundary velocities
        self.u[-1, :] = self.u_lid
        self.v[-1, :] = 0.0
        self.u[0, :] = 0.0
        self.v[0, :] = 0.0
        self.u[:, 0] = 0.0
        self.v[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[:, -1] = 0.0

    def solve_vorticity_transport_ADI(self):
        """
        Solve vorticity transport equation using vectorized Alternating Direction Implicit (ADI).
        - Implicit central-difference diffusion (unconditionally stable in 1D).
        - Explicit upwind convection.
        - Exact Dirichlet wall-vorticity boundary closures incorporated directly into tridiagonal RHS.
        """
        M = self.M
        h = self.h
        nu = self.nu
        dt = self.dt
        A, B, C = self.A_adi, self.B_adi, self.C_adi
        P, denom = self.P_adi, self.denom_adi
        omega_old = self.omega.copy()

        # 1. Explicit upwind convection
        domega_dx = np.where(
            self.u_c > 0,
            (omega_old[1:-1, 1:-1] - omega_old[1:-1, :-2]) / h,
            (omega_old[1:-1, 2:] - omega_old[1:-1, 1:-1]) / h
        )
        domega_dy = np.where(
            self.v_c > 0,
            (omega_old[1:-1, 1:-1] - omega_old[:-2, 1:-1]) / h,
            (omega_old[2:, 1:-1] - omega_old[1:-1, 1:-1]) / h
        )
        conv = self.u_c * domega_dx + self.v_c * domega_dy

        # 2. Step 1: Implicit X (row sweeps), Explicit Y
        diff_y = (nu / (h**2)) * (omega_old[2:, 1:-1] + omega_old[:-2, 1:-1] - 2.0 * omega_old[1:-1, 1:-1])
        RHS_x = omega_old[1:-1, 1:-1] + dt * (diff_y - conv)

        # Boundary closures on left (col 0) and right (col -1)
        RHS_x[:, 0] -= A * omega_old[1:-1, 0]
        RHS_x[:, -1] -= C * omega_old[1:-1, -1]

        # Vectorized Thomas algorithm across all M rows concurrently
        Q_x = np.zeros((M, M))
        Q_x[:, 0] = RHS_x[:, 0] / denom[0]
        for j in range(1, M):
            Q_x[:, j] = (RHS_x[:, j] - A * Q_x[:, j - 1]) / denom[j]

        omega_star_inner = np.zeros((M, M))
        omega_star_inner[:, -1] = Q_x[:, -1]
        for j in range(M - 2, -1, -1):
            omega_star_inner[:, j] = Q_x[:, j] - P[j] * omega_star_inner[:, j + 1]

        omega_star = self.omega.copy()
        omega_star[1:-1, 1:-1] = omega_star_inner

        # 3. Step 2: Implicit Y (col sweeps), Explicit X
        diff_x = (nu / (h**2)) * (omega_star[1:-1, 2:] + omega_star[1:-1, :-2] - 2.0 * omega_star[1:-1, 1:-1])
        RHS_y = omega_star[1:-1, 1:-1] + dt * (diff_x - conv)

        # Boundary closures on bottom (row 0) and top (row -1)
        RHS_y[0, :] -= A * omega_star[0, 1:-1]
        RHS_y[-1, :] -= C * omega_star[-1, 1:-1]

        # Vectorized Thomas algorithm across all M columns concurrently
        Q_y = np.zeros((M, M))
        Q_y[0, :] = RHS_y[0, :] / denom[0]
        for i in range(1, M):
            Q_y[i, :] = (RHS_y[i, :] - A * Q_y[i - 1, :]) / denom[i]

        omega_new_inner = np.zeros((M, M))
        omega_new_inner[-1, :] = Q_y[-1, :]
        for i in range(M - 2, -1, -1):
            omega_new_inner[i, :] = Q_y[i, :] - P[i] * omega_new_inner[i + 1, :]

        omega_new = self.omega.copy()
        omega_new[1:-1, 1:-1] = omega_new_inner
        return omega_new

    def calculate_pressure(self, max_iter=2500, tol=1e-5):
        """
        Recover the pressure field by solving the pressure Poisson equation:
            del^2(p) = - [ (du/dx)^2 + 2*(du/dy)*(dv/dx) + (dv/dy)^2 ]
        with homogeneous Neumann boundary conditions dp/dn = 0 and mean-zero gauge.
        """
        h = self.h
        dudx = (self.u[1:-1, 2:] - self.u[1:-1, :-2]) / (2.0 * h)
        dudy = (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2.0 * h)
        dvdx = (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2.0 * h)
        dvdy = (self.v[2:, 1:-1] - self.v[:-2, 1:-1]) / (2.0 * h)

        rhs = -(dudx**2 + 2.0 * dudy * dvdx + dvdy**2)

        p_new = self.p.copy()
        for _ in range(max_iter):
            p_old = p_new.copy()
            p_new[1:-1, 1:-1] = 0.25 * (
                p_old[2:, 1:-1] + p_old[:-2, 1:-1] +
                p_old[1:-1, 2:] + p_old[1:-1, :-2] - (h**2) * rhs
            )
            # Neumann boundary conditions: zero normal derivative
            p_new[0, :] = p_new[1, :]
            p_new[-1, :] = p_new[-2, :]
            p_new[:, 0] = p_new[:, 1]
            p_new[:, -1] = p_new[:, -2]

            if np.max(np.abs(p_new[1:-1, 1:-1] - p_old[1:-1, 1:-1])) < tol:
                break

        self.p = p_new - np.mean(p_new[1:-1, 1:-1])

    def solve(self, max_iterations=25000, tolerance=1e-5, log_interval=500):
        """Main iterative coupling loop between vorticity transport and streamfunction."""
        print(f"\n{'='*65}")
        print(f"LID-DRIVEN CAVITY SOLVER (Re={self.Re}, Grid={self.N}x{self.N}, Lid='{self.lid_profile}')")
        print(f"{'='*65}")

        start_time = time.time()
        for iteration in range(max_iterations):
            omega_old = self.omega.copy()

            # 1. Update velocities
            self.calculate_velocities()

            # 2. Enforce wall vorticity
            self.apply_boundary_conditions()

            # 3. Vectorized ADI step for omega
            self.omega = self.solve_vorticity_transport_ADI()

            # 4. Enforce wall vorticity
            self.apply_boundary_conditions()

            # 5. Solve streamfunction Poisson equation
            self.solve_streamfunction()

            # 6. Check convergence on interior vorticity
            max_change = np.max(np.abs(self.omega[1:-1, 1:-1] - omega_old[1:-1, 1:-1]))

            if iteration % log_interval == 0 or iteration == max_iterations - 1:
                self.history['iterations'].append(iteration)
                self.history['max_change'].append(max_change)
                self.history['psi_min'].append(float(self.psi.min()))
                print(f"Iter {iteration:6d} | max d(omega): {max_change:.2e} | psi_min: {self.psi.min():.6f}")

            if max_change < tolerance and iteration > 500:
                elapsed = time.time() - start_time
                print(f"\n[SUCCESS] Converged in {iteration} iterations ({elapsed:.2f} s). Final max d(omega): {max_change:.2e}")
                self.calculate_velocities()
                self.calculate_pressure()
                return True, iteration

        elapsed = time.time() - start_time
        print(f"\n[INFO] Reached max iterations ({max_iterations}) in {elapsed:.2f} s. Final max d(omega): {max_change:.2e}")
        self.calculate_velocities()
        self.calculate_pressure()
        return False, max_iterations

    def get_vortex_center(self):
        """Return (x, y) physical coordinates of the primary vortex center."""
        min_idx = np.unravel_index(np.argmin(self.psi), self.psi.shape)
        return float(self.x[min_idx[1]]), float(self.y[min_idx[0]])

    def print_summary(self):
        """Display physical summary of the solution."""
        vx, vy = self.get_vortex_center()
        vel_mag = np.sqrt(self.u**2 + self.v**2)

        print(f"\n{'='*65}")
        print("SOLUTION SUMMARY")
        print(f"{'='*65}")
        print(f"Grid Size: {self.N} x {self.N} | Reynolds: Re = {self.Re}")
        print(f"Lid Velocity Profile: {self.lid_profile} (Peak U = {self.U})")
        print(f"Primary Vortex Center: x = {vx:.4f}, y = {vy:.4f}")
        print(f"Streamfunction: min = {self.psi.min():.6f}, max = {self.psi.max():.6f}")
        print(f"Vorticity:      min = {self.omega.min():.2f}, max = {self.omega.max():.2f}")
        print(f"Max Velocity:   |V|_max = {vel_mag.max():.4f}")
        print(f"Pressure Range: min = {self.p.min():.4f}, max = {self.p.max():.4f}")
        print(f"{'='*65}\n")

    def create_plots(self, save_fig=True, show=True):
        """Generate comprehensive 6-panel flow diagnostics figure."""
        try:
            plt.style.use('seaborn-v0_8-whitegrid')
        except Exception:
            pass

        fig = plt.figure(figsize=(16, 10), dpi=300)

        # 1. Streamlines
        ax1 = plt.subplot(231)
        levels = np.linspace(self.psi.min(), 0, 25)
        ax1.contourf(self.X, self.Y, self.psi, levels=levels, cmap='viridis', alpha=0.7)
        ax1.contour(self.X, self.Y, self.psi, levels=levels, colors='#2E86AB', linewidths=0.8)
        ax1.set_xlabel('x (horizontal)')
        ax1.set_ylabel('y (vertical)')
        ax1.set_title(r'Streamlines $\psi$', fontweight='bold')
        ax1.set_aspect('equal')

        # 2. Vorticity
        ax2 = plt.subplot(232)
        c2 = ax2.contourf(self.X, self.Y, self.omega, levels=35, cmap='coolwarm', alpha=0.85)
        ax2.set_xlabel('x (horizontal)')
        ax2.set_ylabel('y (vertical)')
        ax2.set_title(r'Vorticity $\omega$', fontweight='bold')
        ax2.set_aspect('equal')
        plt.colorbar(c2, ax=ax2, fraction=0.046, pad=0.04)

        # 3. Velocity Magnitude
        ax3 = plt.subplot(233)
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        c3 = ax3.contourf(self.X, self.Y, vel_mag, levels=30, cmap='plasma', alpha=0.85)
        ax3.set_xlabel('x (horizontal)')
        ax3.set_ylabel('y (vertical)')
        ax3.set_title('Velocity Magnitude |V|', fontweight='bold')
        ax3.set_aspect('equal')
        plt.colorbar(c3, ax=ax3, fraction=0.046, pad=0.04)

        # 4. Vertical Centerline u(y)
        ax4 = plt.subplot(234)
        x_idx = np.argmin(np.abs(self.x - self.L / 2.0))
        ax4.plot(self.u[:, x_idx], self.y, 'b-', lw=2.2, label=f'FDM u at x={self.L/2:.2f}')
        if self.lid_profile == 'constant' and self.Re in [100, 1000, 3200, 5000]:
            ghia_u = GHIA_DATA[f'u_Re{self.Re}']
            ax4.scatter(ghia_u, GHIA_DATA['y_u'], color='black', s=25, zorder=5, label='Ghia et al. (1982)')
        ax4.set_xlabel('u-velocity')
        ax4.set_ylabel('y (vertical)')
        ax4.set_title('u-Velocity at Vertical Centerline', fontweight='bold')
        ax4.grid(True, ls='--', alpha=0.4)
        ax4.legend(framealpha=0.9, fontsize=9)

        # 5. Horizontal Centerline v(x)
        ax5 = plt.subplot(235)
        y_idx = np.argmin(np.abs(self.y - self.L / 2.0))
        ax5.plot(self.x, self.v[y_idx, :], 'r-', lw=2.2, label=f'FDM v at y={self.L/2:.2f}')
        if self.lid_profile == 'constant' and self.Re in [100, 1000, 3200, 5000]:
            ghia_v = GHIA_DATA[f'v_Re{self.Re}']
            ax5.scatter(GHIA_DATA['x_v'], ghia_v, color='black', s=25, zorder=5, label='Ghia et al. (1982)')
        ax5.set_xlabel('x (horizontal)')
        ax5.set_ylabel('v-velocity')
        ax5.set_title('v-Velocity at Horizontal Centerline', fontweight='bold')
        ax5.grid(True, ls='--', alpha=0.4)
        ax5.legend(framealpha=0.9, fontsize=9)

        # 6. Pressure
        ax6 = plt.subplot(236)
        c6 = ax6.contourf(self.X, self.Y, self.p, levels=30, cmap='RdGy', alpha=0.85)
        ax6.set_xlabel('x (horizontal)')
        ax6.set_ylabel('y (vertical)')
        ax6.set_title('Pressure Field p', fontweight='bold')
        ax6.set_aspect('equal')
        plt.colorbar(c6, ax=ax6, fraction=0.046, pad=0.04)

        plt.suptitle(f"2D Lid-Driven Cavity Flow | Re = {self.Re} | Grid = {self.N}×{self.N} | Lid: {self.lid_profile.capitalize()}",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_fig:
            fname = f"lid_driven_diagnostic_Re{self.Re}_N{self.N}_{self.lid_profile}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[SUCCESS] Diagnostic plot saved: {fname}")

        if show:
            plt.show()
        else:
            plt.close()

    def create_showcase_plots(self, save_fig=True, show=True):
        """Generate high-contrast publication showcase plot comparing streamlines and centerline."""
        fig, axs = plt.subplots(1, 2, figsize=(18, 8), dpi=300)

        # 1. Streamlines
        ax = axs[0]
        levels = np.linspace(self.psi.min(), 0, 30)
        ax.contourf(self.X, self.Y, self.psi, levels=levels, cmap='viridis', alpha=0.85)
        ax.contour(self.X, self.Y, self.psi, levels=levels, colors='black', linewidths=1.0, alpha=0.6)
        vx, vy = self.get_vortex_center()
        ax.plot(vx, vy, 'r*', markersize=12, label=f'Vortex ({vx:.3f}, {vy:.3f})')
        ax.set_title(r"Streamlines ($\psi$)", fontsize=14, fontweight='bold')
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_aspect('equal')
        ax.set_xlim(0, self.L)
        ax.set_ylim(0, self.L)
        ax.legend(loc='lower left', framealpha=0.9)

        # 2. Centerline validation
        ax = axs[1]
        x_idx = np.argmin(np.abs(self.x - self.L / 2.0))
        ax.plot(self.u[:, x_idx], self.y, 'k-', lw=2.8, label=rf"$u(y)$ at $x = {self.L/2:.2f}$ (FDM)")
        if self.lid_profile == 'constant' and self.Re in [100, 1000, 3200, 5000]:
            ghia_u = GHIA_DATA[f'u_Re{self.Re}']
            ax.scatter(ghia_u, GHIA_DATA['y_u'], facecolors='none', edgecolors='red', s=45, lw=1.5,
                       label='Ghia et al. (1982) Benchmark')
        ax.set_title("Centerline Velocity Profile", fontsize=14, fontweight='bold')
        ax.set_xlabel("u-velocity", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.grid(True, ls='--', alpha=0.4)
        ax.legend(fontsize=11, framealpha=0.95)

        plt.suptitle(f"Lid-Driven Cavity Flow — Re = {self.Re}, Grid = {self.N}×{self.N} ({self.lid_profile.capitalize()} Lid)",
                     fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_fig:
            fname = f"lid_driven_showcase_Re{self.Re}_N{self.N}.png"
            plt.savefig(fname, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"[SUCCESS] Showcase plot saved: {fname}")

        if show:
            plt.show()
        else:
            plt.close()

    def save_model(self, filename=None):
        """Save flow fields and parameters to both Python pickle (.pkl) and NumPy archive (.npz)."""
        if filename is None:
            filename = f"lid_driven_model_Re{self.Re}_N{self.N}.pkl"

        vx, vy = self.get_vortex_center()

        model_data = {
            'parameters': {
                'N': self.N,
                'Re': self.Re,
                'U': self.U,
                'L': self.L,
                'h': self.h,
                'nu': self.nu,
                'dt': self.dt,
                'lid_profile': self.lid_profile,
                'poisson_solver': self.poisson_solver
            },
            'coordinates': {
                'x': self.x,
                'y': self.y,
                'X': self.X,
                'Y': self.Y
            },
            'fields': {
                'psi': self.psi,
                'omega': self.omega,
                'u': self.u,
                'v': self.v,
                'p': self.p
            },
            'vortex_center': {
                'x': vx,
                'y': vy,
                'array_indices': np.unravel_index(np.argmin(self.psi), self.psi.shape)
            },
            'convergence': self.history,
            'metadata': {
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'coordinate_system': 'x=horizontal [0, L], y=vertical [0, L]',
                'moving_lid': f'top wall (y = {self.L}) with profile {self.lid_profile}',
                'provenance': 'High-resolution FDM solver with precomputed sparse LU Poisson and vectorized ADI closures'
            }
        }

        with open(filename, 'wb') as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"[SUCCESS] Model pickle saved: {filename}")

        # Save NumPy npz archive (compatible with plot_from_npz.py)
        npz_name = f"flow_fields_Re{self.Re}_N{self.N}.npz"
        np.savez(
            npz_name,
            x=self.x, y=self.y,
            X=self.X, Y=self.Y,
            psi=self.psi, omega=self.omega,
            u=self.u, v=self.v, p=self.p,
            vortex_x=vx, vortex_y=vy,
            Re=self.Re, N=self.N,
            lid_profile=self.lid_profile
        )
        print(f"[SUCCESS] Flow fields NumPy archive saved: {npz_name}")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="High-resolution 2D Lid-Driven Cavity FDM Solver.")
    parser.add_argument('--N', type=int, default=129, help="Grid size along each axis (default: 129)")
    parser.add_argument('--Re', type=int, default=1000, help="Reynolds number (default: 1000)")
    parser.add_argument('--lid_profile', type=str, default='constant', choices=['constant', 'regularized'],
                        help="Top wall velocity profile: 'constant' (Ghia benchmark) or 'regularized' (singularity-free)")
    parser.add_argument('--solver', type=str, default='lu', choices=['lu', 'sor'],
                        help="Streamfunction Poisson solver: 'lu' (precomputed sparse LU, exact) or 'sor' (RB-SOR)")
    parser.add_argument('--max_iterations', type=int, default=25000, help="Maximum solver iterations (default: 25000)")
    parser.add_argument('--tolerance', type=float, default=1e-5, help="Vorticity convergence tolerance (default: 1e-5)")
    parser.add_argument('--no_plots', action='store_true', help="Suppress interactive plot windows")
    parser.add_argument('--save_fig', action='store_true', default=True, help="Save diagnostic and showcase figures")

    args = parser.parse_args()

    solver = LidDrivenCavitySolver(
        N=args.N,
        Re=args.Re,
        lid_profile=args.lid_profile,
        poisson_solver=args.solver
    )

    converged, iters = solver.solve(
        max_iterations=args.max_iterations,
        tolerance=args.tolerance
    )

    solver.print_summary()
    solver.save_model()

    if not args.no_plots:
        solver.create_plots(save_fig=args.save_fig, show=True)
        solver.create_showcase_plots(save_fig=args.save_fig, show=True)
    else:
        if args.save_fig:
            solver.create_plots(save_fig=True, show=False)
            solver.create_showcase_plots(save_fig=True, show=False)
