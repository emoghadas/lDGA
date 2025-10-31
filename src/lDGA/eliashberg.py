import numpy as np
from numba import jit
from scipy.sparse.linalg import LinearOperator, gmres
from typing import Tuple
from lDGA.config import DGA_ConfigType
from lDGA.utilities import k2ik, build_nu_mats, build_w_mats, U_trans, Udyn, Udyn_arr, G_wq_given_nuk, ek
from lDGA.bse import asymp_chi


@jit(nopython=True)
def chi_pp_loc(dga_cfg : DGA_ConfigType) -> np.ndarray:
    '''
    Compute local pp chi for iw=0
    '''
    beta = dga_cfg.beta; chi=dga_cfg.chi_ph
    n4iwf=dga_cfg.n4iwf; n4iwb=dga_cfg.n4iwb
    chi0_w = dga_cfg.chi0_w

    chi_ud =  chi[1,...]

    nup = n4iwf//2
    chi_pp = np.empty((2*nup, 2*nup), dtype=np.complex128)
    for i, nu1 in enumerate(range(-nup, nup)):
        for j, nu2 in enumerate(range(-nup, nup)):
            iw_idx =  -nu1 - nu2 - 1 + n4iwb  
            chi_pp[i, j] = chi_ud[n4iwf+nu1, n4iwf+nu2, iw_idx]
    return chi_pp


def bse_pp(dga_cfg : DGA_ConfigType) -> np.ndarray:
    '''
    Compute local gamma_pp for iw=0
    '''
    beta = dga_cfg.beta
    n4iwf = dga_cfg.n4iwf
    n4iwb = dga_cfg.n4iwb
    chi_pp = dga_cfg.chi_pp
    nouter = dga_cfg.nouter
    niwf = dga_cfg.niwf
    g = dga_cfg.g_imp

    nup = n4iwf//2
    nu_range = slice(niwf-nup,niwf+nup)
    chi0_pp = -beta*g[nu_range]*np.flip(g[nu_range])

    gamma_pp = beta**2 * np.linalg.inv(np.linalg.inv(chi_pp - np.diag(chi0_pp)) + np.diag(1/chi0_pp))

    return np.flip(gamma_pp, axis=-1)


@jit(nopython=True)
def get_pairing_vertex(dga_cfg:DGA_ConfigType, gamma_irr_d:np.ndarray, gamma_irr_m:np.ndarray, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d:np.ndarray, chi_m:np.ndarray, chi0_w_q:np.ndarray) -> np.ndarray:
    '''
    Compute ladder pairing vertex for singlet and triplet channels
    '''
    u=dga_cfg.U; beta=dga_cfg.beta; dens=dga_cfg.occ_imp
    g0=dga_cfg.g0; omega0=dga_cfg.w0
    ts=dga_cfg.ts
    n4iwf=dga_cfg.n4iwf; n4iwb=dga_cfg.n4iwb
    self_old=dga_cfg.s_imp
    g_old=dga_cfg.g_imp
    F_d_loc = dga_cfg.F_d_loc
    F_m_loc = dga_cfg.F_m_loc
    dim = dga_cfg.kdim
    irrbz = dga_cfg.irrbz
    all_q_sym = dga_cfg.all_q_sym
    symq_weights = dga_cfg.symq_weights
    qpoints = dga_cfg.q_grid_loc
    Nk = dga_cfg.n_kpoints
    Nqtot = dga_cfg.n_qpoints_fullbz
    Nq = chi_d.shape[1]
    asymp = dga_cfg.asymp
    niwf = dga_cfg.nouter
    gamma_pp = dga_cfg.gamma_pp

    #chi_d -= asymp_chi(2*niwf, beta)
    #chi_m -= asymp_chi(2*niwf, beta)

    nup = n4iwf//2

    wmats  = build_w_mats(n4iwb,beta)

    Uw = Udyn_arr(wmats,omega0,g0,u).astype(np.complex128)

    u_d = 2*Uw - u
    u_m = - u

    gamma_s = np.empty((2*nup,2*nup,qpoints.shape[0]), dtype=np.complex128)
    gamma_t = np.empty((2*nup,2*nup,qpoints.shape[0]), dtype=np.complex128)
    for q_idx, q in enumerate(qpoints):
        for i,inu1 in enumerate(range(-nup, nup)):
            for j,inu2 in enumerate(range(-nup, nup)):
                nu1_idx = n4iwf + inu1
                nu2_idx = n4iwf + inu2
                w_idx = -inu1 - inu2 + n4iwb - 1

                phi_d = np.linalg.inv(np.diag(1/chi0_w_q[:,w_idx,q_idx]) + (gamma_irr_d[:,:,w_idx]-u_d[w_idx]*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)
                phi_m = np.linalg.inv(np.diag(1/chi0_w_q[:,w_idx,q_idx]) + (gamma_irr_m[:,:,w_idx]-u_m*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)
                #one = np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128)
                #phi_d = phi_d - phi_d@(u_d[w_idx]*one)@phi_d*(1-u_d[w_idx]*chi_d[w_idx,q_idx])/beta**2 + phi_d@(u_d[w_idx]*one)@phi_d*(1-u_d[w_idx]*chi_d[w_idx,q_idx])**2/(1-u_d[w_idx]*(chi_d[w_idx,q_idx]+asymp_chi(2*niwf, beta)))/beta**2
                #phi_m = phi_m - phi_m@(u_m*one)@phi_m*(1-u_m*chi_m[w_idx,q_idx])/beta**2 + phi_m@(u_m*one)@phi_m*(1-u_m*chi_m[w_idx,q_idx])**2/(1-u_m*(chi_m[w_idx,q_idx]+asymp_chi(2*niwf, beta)))/beta**2

                #chi_m[w_idx,q_idx] += asymp_chi(2*niwf, beta)
                #chi_d[w_idx,q_idx] += asymp_chi(2*niwf, beta)

                phi_slice_d = phi_d[nu1_idx, nu2_idx]
                phi_slice_m = phi_m[nu1_idx, nu2_idx]
                chi0_nu1 = chi0_w_q[nu1_idx,w_idx,q_idx]/beta
                chi0_nu2 = chi0_w_q[nu2_idx,w_idx,q_idx]/beta
                gamma_nu1_d = gamma_d[nu1_idx,w_idx,q_idx]
                gamma_nu2_d = gamma_d[nu2_idx,w_idx,q_idx]
                gamma_nu1_m = gamma_m[nu1_idx,w_idx,q_idx]
                gamma_nu2_m = gamma_m[nu2_idx,w_idx,q_idx]
                f_pp = (F_d_loc - F_m_loc)[nu1_idx,nu2_idx,w_idx]

                f_d = 1.0*(inu1==inu2)*beta/chi0_nu1 - phi_slice_d/(chi0_nu1*chi0_nu2) + u_d[w_idx] * (1-u_d[w_idx]*(chi_d[w_idx,q_idx])) * gamma_nu1_d * gamma_nu2_d
                f_m = 1.0*(inu1==inu2)*beta/chi0_nu1 - phi_slice_m/(chi0_nu1*chi0_nu2) + u_m * (1-u_m*(chi_m[w_idx,q_idx])) * gamma_nu1_m * gamma_nu2_m

                gamma_s[i,j,q_idx] = 0.5*f_d - 1.5*f_m - 2*f_pp - gamma_pp[i,j]
                #gamma_t[i,j,q_idx] = 0.5*f_d + 0.5*f_m
    
    return gamma_s


############## power iteration routines ###############

def get_gap_start(nup:int, nk:int, ktype:str) -> np.float64:
    '''
    starting guess for gap with even frequencies but s/d-wave momentum structure
    '''

    kgrid = np.linspace(-np.pi,np.pi,nk)

    gap0 = np.zeros((2*nup,nk,nk), dtype=np.complex128)

    if ktype=='d':
        gap0[nup:,...] = (np.cos(kgrid[:,None]) - np.cos(kgrid[None,:]))
    elif ktype=='s':
        gap0[nup:,...] = np.ones((nk,nk), dtype=np.complex128)
    else:
        raise ValueError("Only s- and d-wave supported !")
    
    gap0[:nup,...] = gap0[nup:,...]
    gap0 /= np.sqrt(np.vdot(gap0, gap0).real)

    return gap0

def power_iteration_old(dga_cfg:DGA_ConfigType, gamma:np.ndarray, gk:np.ndarray, mode:str):
    '''
    Power iteration solver for specific for linearized Eliashberg kernel
    '''
    nup = gamma.shape[0]//2
    niwf = gk.shape[0]//2
    nk = dga_cfg.nk
    kdim = dga_cfg.kdim
    beta = dga_cfg.beta

    g = gk[niwf-nup:niwf+nup,...]
    g = g.reshape(2*nup,nk,nk)
    gamma_s = ( 0.5 * (gamma + np.flip(gamma, axis=-2)) )
    gamma_s = gamma_s.reshape(2*nup,2*nup,nk,nk)
    gammax = np.fft.fftn(gamma_s, axes=(-1,-2))

    gap_old = get_gap_start(nup, nk, ktype=mode) 
    lambda_old = 1  # Randomly start with
    converged = False
    eps = 1e-3
    max_count=10000
    count = 0
    while not converged:
        count += 1
        gap_gg = np.fft.ifftn(gap_old * np.abs(g) ** 2, axes=(-1,-2))
        gap_new = 1. / (beta*nk**kdim) * np.einsum('ijkl,jkl->ikl', gammax, gap_gg)
        gap_new = np.fft.fftn(gap_new, axes=(-1,-2))
        lambda_new = np.sum(np.conj(gap_old) * gap_new) / np.sum(np.conj(gap_old) * gap_old)
        gap_old = gap_new / lambda_new

        if (np.abs(lambda_new - lambda_old) < eps or count > max_count):
            converged = True
        lambda_old = lambda_new

    gap_old /= np.sqrt(np.vdot(gap_old, gap_old).real)
    return lambda_new, gap_old

def _make_form_factor(nk, mode):
    # 2D square lattice BZ form factors on an nk×nk grid
    k = 2*np.pi*np.arange(nk)/nk
    KX, KY = np.meshgrid(k, k, indexing='ij')
    if mode.lower() in ('a1g','s','s-wave','swave'):
        f = np.ones((nk, nk), dtype=np.complex128)
    elif mode.lower() in ('b1g','d','d-wave','dwave'):
        f = (np.cos(KX) - np.cos(KY)).astype(np.complex128)
    else:
        raise ValueError("mode must be one of: 'A1g'/'s', 'B1g'/'d'")
    den = np.vdot(f, f) + 1e-30  # <f|f>
    return f, den

def _project_channel(gap, f, den):
    """
    Project each Matsubara slice gap[iw, :, :] onto the form factor f:
      gap_iw -> f * (<f|gap_iw> / <f|f>)
    gap shape: (2*nup, nk, nk)
    f shape: (nk, nk)
    """
    proj = np.empty_like(gap)
    # vectorized over iw: compute <f|gap_iw> for all iw at once
    # result shape (2*nup,)
    num = np.tensordot(np.conj(f), gap, axes=([0,1],[1,2]))  # sum_{k} f*(k) gap[iw,k]
    # broadcast back: proj[iw] = f * (num[iw]/den)
    proj = f[None, ...] * (num[:, None, None] / den)
    return proj

def _enforce_singlet_even_freq(gap):
    """
    Enforce even-frequency (singlet) symmetry along Matsubara axis:
      gap(ω) = gap(-ω).
    Assumes ordering [... -n, ..., -1, +1, ..., +n ] with no ω=0 slice in this array,
    OR symmetric content; if you *do* have ω=0 included inside the 2*nup block,
    this still works (the center element is averaged with itself).
    """
    return 0.5*(gap + gap[::-1, ...])

def power_iteration(dga_cfg, gamma, gk, mode:str,
                                   eps=1e-3, max_count=10000,
                                   gmres_atol=1e-8, gmres_maxiter=None):
    """
    Like your power_iteration, but returns the eigenvalue of K closest to 1
    by iterating on M = (I - K)^(-1).
    """
    nup = gamma.shape[0] // 2
    niwf = gk.shape[0] // 2
    nk   = dga_cfg.nk
    kdim = dga_cfg.kdim
    beta = dga_cfg.beta

    # --- assemble the ingredients exactly like your code does ---
    g = gk[niwf - nup : niwf + nup, ...].reshape(2*nup, nk, nk)
    gamma_s = 0.5 * (gamma + np.flip(gamma, axis=-2))
    gamma_s = gamma_s.reshape(2*nup, 2*nup, nk, nk)
    gammax  = np.fft.fftn(gamma_s, axes=(-1, -2))

    # helper: apply K (matrix-free), returns same shape as gap
    def apply_K(gap):
        gap_gg  = np.fft.ifftn(gap * np.abs(g)**2, axes=(-1, -2))
        Kg_r    = (1.0 / (beta * nk**kdim)) * np.einsum('ijkl,jkl->ikl', gammax, gap_gg)
        Kg      = np.fft.fftn(Kg_r, axes=(-1, -2))
        return Kg

    # --- G-metric inner products (optional but recommended) ---
    def inner_G(x, y): return np.sum(np.conj(x) * (np.abs(g)**2) * y)
    def norm_G(x):     return np.sqrt(np.real(inner_G(x, x)) + 0.0)

    # starting vector (keep your channel-specific initializer)
    v = get_gap_start(nup, nk, ktype=mode)          # shape: (2*nup, nk, nk)
    v /= (norm_G(v) + 1e-30)

    # Flatten/unflatten for GMRES LinearOperator
    shp = v.shape
    N   = v.size
    def flat(x): return x.reshape(N)
    def unflat(x): return x.reshape(shp)

    # Build a LinearOperator for (I - K)
    def Ik_matvec(x_flat):
        x = unflat(x_flat)
        y = x - apply_K(x)
        return flat(y)

    Ik = LinearOperator((N, N), matvec=Ik_matvec, dtype=np.complex128)

    # iteration on M = (I - K)^(-1)
    lam_old = None
    count   = 0
    while True:
        count += 1

        # Solve (I - K) x = v  -> x = M v
        x_flat, info = gmres(Ik, flat(v), atol=gmres_atol, rtol=0.0, maxiter=gmres_maxiter)
        if info != 0:
            raise RuntimeError(f"GMRES did not converge (info={info}). "
                               f"Try relaxing gmres_atol or adding a preconditioner.")

        x = unflat(x_flat)

        # Power step on M
        # Rayleigh estimate for μ (eigenvalue of M): μ ≈ <v, x>_G / <v, v>_G
        mu = inner_G(v, x) / (inner_G(v, v) + 1e-30)

        # Map back to λ of K:  λ = 1 - 1/μ
        lam_new = 1.0 - 1.0 / mu

        # Next iterate
        v = x / (norm_G(x) + 1e-30)

        # Convergence (on λ and residual of K)
        # residual r = ||K v - λ v||_G / ||v||_G
        Kv = apply_K(v)
        r  = norm_G(Kv - lam_new * v) / (norm_G(v) + 1e-30)

        if ((lam_old is not None and np.abs(lam_new - lam_old) < eps) and (r < 5*eps)) \
           or (count >= max_count):
            break

        lam_old = lam_new

    # normalize output vector
    v /= (norm_G(v) + 1e-30)
    return lam_new, v
