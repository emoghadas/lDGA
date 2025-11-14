import numpy as np
from numba import jit
from scipy.sparse.linalg import LinearOperator, gmres, eigs, eigsh
from typing import Tuple
from lDGA.config import DGA_ConfigType
from lDGA.utilities import k2ik, build_nu_mats, build_w_mats, U_trans, Udyn, Udyn_arr, G_wq_given_nuk, ek, wrap_k
from lDGA.bse import asymp_chi


@jit(nopython=True)
def get_F_q(dga_cfg:DGA_ConfigType, gamma_irr_d:np.ndarray, gamma_irr_m:np.ndarray, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d:np.ndarray, chi_m:np.ndarray, chi0_w_q:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute ladder F for density and magnetic channels at w0=, v=pi*T
    '''
    u=dga_cfg.U; beta=dga_cfg.beta; dens=dga_cfg.occ_imp
    g0=dga_cfg.g0; omega0=dga_cfg.w0
    n4iwf=dga_cfg.n4iwf; n4iwb=dga_cfg.n4iwb
    qpoints = dga_cfg.q_grid_loc

    wmats  = build_w_mats(n4iwb,beta)
    Uw = Udyn_arr(wmats,omega0,g0,u).astype(np.complex128)
    u_d = 2*Uw - u
    u_m = - u

    F_w_q_d = np.empty((2*n4iwf,2*n4iwb+1,qpoints.shape[0]), dtype=np.complex128)
    F_w_q_m = np.empty((2*n4iwf,2*n4iwb+1,qpoints.shape[0]), dtype=np.complex128)
    for q_idx, q in enumerate(qpoints):
        for w_idx, iw in enumerate(range(-n4iwb,n4iwb+1)):
            phi_d = np.linalg.inv(np.diag(1/chi0_w_q[:,w_idx,q_idx]) + (gamma_irr_d[:,:,w_idx]-u_d[w_idx]*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)
            phi_m = np.linalg.inv(np.diag(1/chi0_w_q[:,w_idx,q_idx]) + (gamma_irr_m[:,:,w_idx]-u_m*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)

            f_d = np.diag(beta**2/chi0_w_q[:,w_idx,q_idx]) - np.diag(beta/chi0_w_q[:,w_idx,q_idx])@phi_d@np.diag(beta/chi0_w_q[:,w_idx,q_idx]) + u_d[w_idx] * (1-u_d[w_idx]*chi_d[w_idx,q_idx])*np.outer(gamma_d[:,w_idx,q_idx], gamma_d[:,w_idx,q_idx])
            f_m = np.diag(beta**2/chi0_w_q[:,w_idx,q_idx]) - np.diag(beta/chi0_w_q[:,w_idx,q_idx])@phi_m@np.diag(beta/chi0_w_q[:,w_idx,q_idx]) + u_m * (1-u_m*chi_m[w_idx,q_idx])*np.outer(gamma_m[:,w_idx,q_idx], gamma_m[:,w_idx,q_idx])

            F_w_q_d[:,w_idx,q_idx] = f_d[n4iwf,:]
            F_w_q_m[:,w_idx,q_idx] = f_m[n4iwf,:]
    
    return F_w_q_d, F_w_q_m


@jit(nopython=True)
def get_epc(dga_cfg, F_w_q_d, F_w_q_m, gk, nseg) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = dga_cfg.U; beta = dga_cfg.beta; dens = dga_cfg.occ_imp
    g0 = dga_cfg.g0; omega0 = dga_cfg.w0
    k_grid = np.asarray(dga_cfg.k_grid, dtype=np.float64)  # ensure ndarray
    dim = dga_cfg.kdim
    Nk = dga_cfg.n_kpoints
    Nk_lin = int(np.round(Nk**(1.0/dim)))

    n4iwf = dga_cfg.n4iwf; n4iwb = dga_cfg.n4iwb
    F_d_loc = dga_cfg.F_d_loc

    g = gk[gk.shape[0] // 2 - n4iwf : gk.shape[0] // 2 + n4iwf, :]

    # ---- build kpath with pure NumPy (no lists, no np.append) ----
    t = np.linspace(0.0, np.pi, nseg)
    n1 = nseg               # (0, t)
    n2 = nseg - 1           # (t[1:], pi)
    n3 = nseg - 2           # (t[::-1][1:-1], same)
    total = n1 + n2 + n3
    kpath = np.empty((total, 2), dtype=np.float64)
    idx = 0
    # segment 1: (0, k_y)
    kpath[idx:idx+n1, 0] = 0.0
    kpath[idx:idx+n1, 1] = t
    idx += n1
    # segment 2: (k_x, pi)
    kpath[idx:idx+n2, 0] = t[1:]
    kpath[idx:idx+n2, 1] = np.pi
    idx += n2
    # segment 3: (k, k) backwards
    back = t[::-1][1:-1]
    kpath[idx:idx+n3, 0] = back
    kpath[idx:idx+n3, 1] = back

    idx_list = np.array([0, nseg-1, 2*nseg-2], dtype=np.int32)

    # --------------------------------------------------------------

    L = kpath.shape[0]
    epc_d_l = np.empty((L, L), dtype=np.complex128)
    epc_d_t = np.empty((L, L), dtype=np.complex128)
    epc_m_t = np.empty((L, L), dtype=np.complex128)
    epc_loc = np.empty((L, L), dtype=np.complex128)

    nu_start = n4iwf - n4iwb
    nu_stop  = n4iwf + n4iwb
    for i in range(L):
        k = kpath[i]                  # ndarray(float64, 1d)
        for j in range(L):
            q = kpath[j]              # ndarray(float64, 1d)
            iq = k2ik(q, Nk_lin)

            # make these complex from the start
            f_q_d    = 0.0 + 0.0j
            f_kpmk_d = 0.0 + 0.0j
            f_kpmk_m = 0.0 + 0.0j
            f_loc    = 0.0 + 0.0j

            for p in range(k_grid.shape[0]):
                kp = k_grid[p]
                ikp   = k2ik(kp, Nk_lin)
                ikpmk = k2ik(wrap_k(kp - k), Nk_lin)    # both arrays now
                ikpq  = k2ik(wrap_k(kp + q), Nk_lin)

                f_q_d    += (1.0/beta) * np.sum(F_w_q_d[:, n4iwb, iq]      * g[:, ikp]           * g[:, ikpq])
                f_kpmk_d -= (1.0/(2.0*beta)) * np.sum(F_w_q_d[n4iwf, :-1, ikpmk] * g[nu_start:nu_stop, ikp] * g[nu_start:nu_stop, ikpq])
                f_kpmk_m -= (3.0/(2.0*beta)) * np.sum(F_w_q_m[n4iwf, :-1, ikpmk] * g[nu_start:nu_stop, ikp] * g[nu_start:nu_stop, ikpq])
                f_loc    -= (1.0/beta) * np.sum(F_d_loc[n4iwf, :, n4iwb]   * g[:, ikp]           * g[:, ikpq])

            epc_d_l[i, j] = f_q_d / Nk
            epc_d_t[i, j] = f_kpmk_d / Nk
            epc_m_t[i, j] = f_kpmk_m / Nk
            epc_loc[i, j] = f_loc / Nk

    return epc_d_l, epc_d_t, epc_m_t, epc_loc