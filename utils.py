import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.optimize import fsolve

import numpy as np

import torch

def analytical_solution(evaluation_points, n_core, n_substrate, n_cladding, w_core, k0, TM=False, num_modes=1):
    """
    Analytical mode solver for asymmetric slab waveguide.

    Parameters:
        evaluation_points: (N,) spatial grid (in microns)
        n_core, n_substrate, n_cladding: refractive indices
        w_core: core width in microns
        k0: free-space wave number (2π / λ)
        TM: bool, whether to compute TM modes
        num_modes: int, number of modes (default: 1)

    Returns:
        analytical_EIs: (num_modes, N) array of electric field profiles
        analytical_modes: (num_modes,) array of effective indices
    """

    def dispersion_eq(n_eff, mode):
        gamma_1 = np.sqrt(n_core**2 - n_eff**2)
        gamma_2 = np.sqrt(n_eff**2 - n_substrate**2)
        gamma_3 = np.sqrt(n_eff**2 - n_cladding**2)
        a = k0 * gamma_1 * w_core
        b = mode * np.pi
        if TM:
            c = np.arctan((n_core**2 * gamma_3) / (n_cladding**2 * gamma_1))
            d = np.arctan((n_core**2 * gamma_2) / (n_substrate**2 * gamma_1))
        else:
            c = np.arctan(gamma_3 / gamma_1)
            d = np.arctan(gamma_2 / gamma_1)
        return a - b - c - d

    analytical_modes = np.zeros((num_modes,))
    analytical_EIs = np.zeros((num_modes, len(evaluation_points)))

    # Permittivity factors for TM mode field scaling
    pdFL = (n_core**2 / n_substrate**2) if TM else 1

    # Core thickness in microns
    thickness = w_core
    hn = np.array([thickness])  # single layer
    hs = np.concatenate(([0], np.cumsum(hn)))

    x = evaluation_points.reshape(-1).copy()
    x_shifted = x + w_core / 2  # shift so core is [0, w_core]

    for mode in range(num_modes):
        guess = n_core * 0.999
        n_eff = fsolve(lambda neff: dispersion_eq(neff, mode), guess)[0]
        analytical_modes[mode] = n_eff

        beta = k0 * n_eff
        gL = np.sqrt(beta**2 - (n_substrate * k0)**2)
        gR = np.sqrt(beta**2 - (n_cladding * k0)**2)
        kn = np.sqrt((n_core * k0)**2 - beta**2)

        A = 1
        theta = np.arctan(gL / kn * pdFL)
        AL = A * np.cos(-theta)
        AR = A * np.cos(kn * hn[-1] - theta)

        field = np.zeros_like(x_shifted)

        # Regions
        left = x_shifted < hs[0]
        core = (x_shifted >= hs[0]) & (x_shifted < hs[1])
        right = x_shifted >= hs[1]

        # Field expressions
        field[left] = AL * np.exp(gL * (x_shifted[left] - hs[0]))
        field[core] = A * np.cos(kn * x_shifted[core] - theta)
        field[right] = AR * np.exp(-gR * (x_shifted[right] - hs[-1]))

        # Normalize the mode
        norm = np.sqrt(np.trapz(np.abs(field)**2, x))
        field /= norm
        analytical_EIs[mode, :] = field

    return analytical_EIs, analytical_modes


def d1_1d(n, d):
    """1st derivative central difference matrix"""
    off_diag_p = np.ones(n - 1)
    off_diag_m = -np.ones(n - 1)
    return sp.diags([off_diag_m, np.zeros(n), off_diag_p], [-1, 0, 1]) / (2 * d)

def d2_1d(n, d):
    """2nd derivative central difference matrix"""
    main_diag = -2 * np.ones(n)
    off_diag = np.ones(n - 1)
    return sp.diags([off_diag, main_diag, off_diag], [-1, 0, 1]) / d**2

def is_symmetric(A, tol=1e-10):
    if not sp.issparse(A):
        A = sp.csr_matrix(A)  # convert to sparse if not already
    diff = A - A.T
    if diff.nnz == 0:
        return True
    else:
        return np.allclose(diff.data, 0, atol=tol)

def full_vectorial_fd(xy, eps_map, dx, dy, k0, num_modes=6):
    """
    Full-vectorial eigenmode solver (2D FD) based on the equations:
    ∂²Ex/∂x² + ∂²Ex/∂y² + ∂/∂x[(1/ε)(∂ε/∂x)Ex] + ∂/∂x[(1/ε)(∂ε/∂y)Ey] + (β² - k₀²ε)Ex = 0
    ∂²Ey/∂x² + ∂²Ey/∂y² + ∂/∂y[(1/ε)(∂ε/∂y)Ey] + ∂/∂y[(1/ε)(∂ε/∂x)Ex] + (β² - k₀²ε)Ey = 0

    Args:
        xy: (Ny, Nx, 2) torch or np array with coordinates
        eps_map: (Ny, Nx) tensor or ndarray, epsilon = n²
        dx, dy: step size
        k0: 2*pi / lambda
        num_modes: number of modes to compute
    Returns:
        neff: (num_modes,) array of effective indices
        Ex_all: (num_modes, Ny, Nx) array of Ex field components
        Ey_all: (num_modes, Ny, Nx) array of Ey field components
        eps_map: (Ny, Nx) array of permittivity
    """

    # Convert tensors to numpy arrays if needed
    if isinstance(xy, torch.Tensor):
        xy = xy.detach().cpu().numpy()
    if isinstance(eps_map, torch.Tensor):
        eps_map = eps_map.detach().cpu().numpy()

    Ny, Nx = eps_map.shape
    N = Nx * Ny

    # Flatten the permittivity map
    eps_flat = eps_map.flatten()
    eps_diag = sp.diags(eps_flat)

    # Create identity matrices
    Ix = sp.eye(Nx)
    Iy = sp.eye(Ny)

    # Second derivative operators (for Laplacian terms ∂²E/∂x² + ∂²E/∂y²)
    D2x = d2_1d(Nx, dx)
    D2y = d2_1d(Ny, dy)

    # Laplacian operator
    Lx_op = sp.kron(Iy, D2x)
    Ly_op = sp.kron(D2y, Ix)
    Laplacian = Lx_op + Ly_op

    # First derivative operators
    Dx_1d = d1_1d(Nx, dx)
    Dy_1d = d1_1d(Ny, dy)
    Dx = sp.kron(Iy, Dx_1d)
    Dy = sp.kron(Dy_1d, Ix)

    # Compute spatial derivatives of permittivity: ∂ε/∂x and ∂ε/∂y
    eps_x = Dx @ eps_flat
    eps_y = Dy @ eps_flat

    # Create diagonal matrices for calculations
    Eps_inv = sp.diags(1.0 / eps_flat)
    Eps_x = sp.diags(eps_x)
    Eps_y = sp.diags(eps_y)

    # ---- Constructing the operator terms for the vectorial wave equation ----

    # Term: ∂/∂x[(1/ε)(∂ε/∂x)Ex] for Ex equation
    T_xx = Dx @ (Eps_inv @ Eps_x)

    # Term: ∂/∂x[(1/ε)(∂ε/∂y)Ey] for Ex equation
    T_xy = Dx @ (Eps_inv @ Eps_y)

    # Term: ∂/∂y[(1/ε)(∂ε/∂y)Ey] for Ey equation
    T_yy = Dy @ (Eps_inv @ Eps_y)

    # Term: ∂/∂y[(1/ε)(∂ε/∂x)Ex] for Ey equation
    T_yx = Dy @ (Eps_inv @ Eps_x)

    # Constructing the diagonal blocks (self-coupling terms)
    A11 = Laplacian + T_xx + k0**2 * eps_diag  # Operator for Ex
    A22 = Laplacian + T_yy + k0**2 * eps_diag  # Operator for Ey

    # Constructing the off-diagonal blocks (cross-coupling terms)
    A12 = T_xy  # Coupling from Ey to Ex
    A21 = T_yx  # Coupling from Ex to Ey

    # Assemble the full system matrix
    top = sp.hstack([A11, A12])
    bottom = sp.hstack([A21, A22])
    A = sp.vstack([top, bottom]).tocsc()

    # Target eigenvalue - use sigma close to the expected eigenvalues
    # β² will be our eigenvalues, so set sigma near k0²ε
    sigma = (k0**2 * np.max(eps_flat)) - 0.1

    # Solve the eigenvalue problem: A*[Ex; Ey] = β²*[Ex; Ey]
    try:
        eigvals, eigvecs = spla.eigs(A, k=num_modes, sigma=sigma, which='LM')
    except Exception as e:
        print(f"Eigenvalue computation failed: {e}")
        # Fallback to a more robust but slower method
        eigvals, eigvecs = spla.eigs(A, k=num_modes, which='LM')

    # Sort eigenvalues in descending order
    idx = np.argsort(eigvals.real)[::-1]
    beta2 = eigvals.real[idx]
    neff = np.sqrt(beta2) / k0  # Convert β to effective index

    # Extract all modes
    Ex_all = np.zeros((num_modes, Ny, Nx), dtype=complex)
    Ey_all = np.zeros((num_modes, Ny, Nx), dtype=complex)

    for i, idx_i in enumerate(idx[:num_modes]):
        E_mode = eigvecs[:, idx_i]
        Ex_all[i] = E_mode[:N].reshape(Ny, Nx)
        Ey_all[i] = E_mode[N:].reshape(Ny, Nx)

    return neff[:num_modes], Ex_all, Ey_all, eps_map

def solve_fd_mode(n_squared_profile, dx, k0, TM,  num_modes=1):
    """
    Solve the 1D scalar Helmholtz eigenvalue problem for TE or TM modes using finite differences.

    Parameters:
        n_squared_profile: (N,) refractive index squared (n^2) profile
        dx: float, spatial step in microns
        k0: float, vacuum wave number
        num_modes: int, number of guided modes to return
        plot: bool, whether to plot the fundamental mode
        TM: bool, whether to solve for TM instead of TE

    Returns:
        beta: (M,) propagation constants
        E_modes: (M, N) normalized field profiles
    """
    N = len(n_squared_profile)
    x = np.linspace(0, dx * (N - 1), N)

    if not TM:
        # Standard central difference Laplacian (TE case)
        diagonals = [
            -2 * np.ones(N),
            np.ones(N - 1),
            np.ones(N - 1)
        ]
        L = diags(diagonals, [0, -1, 1], format='csc') / dx**2

    else: # we solve to get H
        # Non-uniform TM Laplacian
        n2 = n_squared_profile
        a = np.copy(n2)
        a[:-1] += n2[1:]
        a = 1 / a

        b = np.copy(n2)
        b[1:] += n2[:-1]
        b = 1 / b

        diagonal_elements = -2 * (a + b) * n2
        upper_diagonal = a[:-1] * 2 * n2[:-1]
        lower_diagonal = b[1:] * 2 * n2[1:]
        L = diags([diagonal_elements, lower_diagonal, upper_diagonal], [0, -1, 1], format='csc') / dx**2

    # Impose Dirichlet boundary conditions
    L = L.copy()
    L[0, :] = 0
    L[0, 0] = 1
    L[-1, :] = 0
    L[-1, -1] = 1

    # Build scale matrix (k₀² * n²(x))
    K = diags([k0**2 * n_squared_profile], [0], format='csc')
    K = K.copy()
    K[0, 0] = 0
    K[-1, -1] = 0

    # Operator A = -d²/dx² + k₀²n²(x)
    A = L + K

    # Solve eigenvalue problem
    eigvals, eigvecs = eigs(A, k=num_modes, sigma=(k0**2 * np.max(n_squared_profile)))
    eigvals = eigvals.real
    eigvecs = eigvecs.real

    # Sort modes by descending eigenvalue (guided first)
    sorted_indices = np.argsort(-eigvals)
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Normalize modes
    E_modes = []
    for i in range(eigvecs.shape[1]):
        E = eigvecs[:, i]
        if TM: # get E from Hy
            E /= n_squared_profile  # TM field scaling (Ey ∝ Hy / n²)
        E /= np.sqrt(np.trapz(E**2, dx=dx))
        E_modes.append(E)
    E_modes = np.stack(E_modes, axis=0)

    beta = np.sqrt(eigvals)

    return beta, E_modes