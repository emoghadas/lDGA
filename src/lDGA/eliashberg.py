import numpy as np
from numba import jit
from scipy.sparse.linalg import LinearOperator, gmres, eigs, eigsh
from typing import Tuple
from lDGA.config import DGA_ConfigType
from lDGA.utilities import k2ik, build_nu_mats, build_w_mats, U_trans, Udyn, Udyn_arr, G_wq_given_nuk, ek
from lDGA.bse import asymp_chi


@jit(nopython=True)
def chi_pp_loc(dga_cfg : DGA_ConfigType) -> Tuple[np.ndarray,np.ndarray]:
    '''
    Compute local pp chi for iw=0
    '''
    beta = dga_cfg.beta; chi=dga_cfg.chi_ph
    n4iwf=dga_cfg.n4iwf; n4iwb=dga_cfg.n4iwb
    chi0_w = dga_cfg.chi0_w
    F_d_loc = dga_cfg.F_d_loc
    F_m_loc = dga_cfg.F_m_loc

    chi_ud =  chi[1,...]
    #F_ud = 0.5*(F_d_loc + F_m_loc)

    nup = n4iwf//2
    chi_pp = np.empty((2*nup, 2*nup), dtype=np.complex128)
    F_d_pp_loc = np.empty((2*nup, 2*nup), dtype=np.complex128)
    F_m_pp_loc = np.empty((2*nup, 2*nup), dtype=np.complex128)
    for i, nu1 in enumerate(range(-nup, nup)):
        for j, nu2 in enumerate(range(-nup, nup)):
            iw_idx =  -nu1 - nu2 - 1 + n4iwb  
            chi_pp[i, j] = chi_ud[n4iwf+nu1, n4iwf+nu2, iw_idx]
            F_d_pp_loc[i, j] = F_d_loc[n4iwf+nu1, n4iwf+nu2, iw_idx]
            F_m_pp_loc[i, j] = F_m_loc[n4iwf+nu1, n4iwf+nu2, iw_idx]
    return chi_pp, F_d_pp_loc, F_m_pp_loc


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

    gamma_pp = beta**2 * (np.linalg.inv(np.flip(chi_pp, axis=1) - np.diag(chi0_pp)) + np.diag(1/chi0_pp))

    return np.flip(gamma_pp, axis=-1)


@jit(nopython=True)
def get_pairing_vertex(dga_cfg:DGA_ConfigType, gamma_irr_d:np.ndarray, gamma_irr_m:np.ndarray, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d:np.ndarray, chi_m:np.ndarray, chi0_w_q:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
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

    #chi_d -= asymp_chi(2*niwf, beta) # this is readded again at the end
    #chi_m -= asymp_chi(2*niwf, beta)

    nup = n4iwf//2

    wmats  = build_w_mats(n4iwb,beta)

    Uw = Udyn_arr(wmats,omega0,g0,u).astype(np.complex128)

    u_d = 2*Uw - u
    u_m = - u

    #gamma_s = np.empty((2*nup,2*nup,qpoints.shape[0]), dtype=np.complex128)
    #gamma_t = np.empty((2*nup,2*nup,qpoints.shape[0]), dtype=np.complex128)
    F_d = np.empty((2*nup,2*nup,qpoints.shape[0]), dtype=np.complex128)
    F_m = np.empty((2*nup,2*nup,qpoints.shape[0]), dtype=np.complex128)
    for q_idx, q in enumerate(qpoints):
        for i,inu1 in enumerate(range(-nup, nup)):
            for j,inu2 in enumerate(range(-nup, nup)):
                nu1_idx = n4iwf + inu1
                nu2_idx = n4iwf + inu2
                w_idx = -inu1 - inu2 + n4iwb - 1

                phi_d = np.linalg.inv(np.diag(1/chi0_w_q[:,w_idx,q_idx]) + (gamma_irr_d[:,:,w_idx]-u_d[w_idx]*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)
                phi_m = np.linalg.inv(np.diag(1/chi0_w_q[:,w_idx,q_idx]) + (gamma_irr_m[:,:,w_idx]-u_m*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)

                # asymptotics for phi
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
                #f_pp = 0.5*(F_d_loc - F_m_loc)[nu1_idx,nu2_idx,w_idx] # local double counting, only relevant for s-wave

                f_d = 1.0*(inu1==inu2)*beta/chi0_nu1 - phi_slice_d/(chi0_nu1*chi0_nu2) + u_d[w_idx] * (1-u_d[w_idx]*(chi_d[w_idx,q_idx])) * gamma_nu1_d * gamma_nu2_d
                f_m = 1.0*(inu1==inu2)*beta/chi0_nu1 - phi_slice_m/(chi0_nu1*chi0_nu2) + u_m * (1-u_m*(chi_m[w_idx,q_idx])) * gamma_nu1_m * gamma_nu2_m

                #gamma_s[i,j,q_idx] = 0.5*f_d - 1.5*f_m #- 2*f_pp - gamma_pp[i,j]
                #gamma_t[i,j,q_idx] = 0.5*f_d + 0.5*f_m
                F_d[i,j,q_idx] = f_d
                F_m[i,j,q_idx] = f_m
    
    return F_d, F_m


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

def get_eig(dga_cfg, gamma, g):
    ''' calculate leading eigenvalues for eliashberg kernel '''
    nup = gamma.shape[0] // 2
    niwf = g.shape[0] // 2
    nk   = dga_cfg.nk
    kdim = dga_cfg.kdim
    beta = dga_cfg.beta

    gk = np.roll(g[niwf - nup : niwf + nup, ...].reshape(2*nup, nk, nk), shift=nk//2, axis=(-1,-2))

    gamma_s = -np.roll(np.flip(gamma, axis=1).reshape(2*nup, 2*nup, nk, nk), shift=nk//2, axis=(-1,-2))

    gammax  = np.fft.fftn(gamma_s, axes=(-1, -2))

    norm = nk**kdim * beta
       
    def mv(gap):
        gap=gap.reshape(np.shape(gk))
        gap_gg = np.fft.fftn(gap * np.abs(gk) ** 2, axes=(-1,-2))
        # symmetrization of the pairing vertex, see M. Kitatani, PhD Thesis Eq. (6.31) performed in the real space
        # sing/trip = 0.5 ( \Gamma(x,nu)\DeltaGG(x,nu) \pm \Gamma(-x,-nu)DeltaGG(-x,nu) )
        gap_new = 1. / norm * np.sum((gammax * gap_gg[None, :, ...] # \Gamma(x,nu)\DeltaGG(x,nu)
                                            + np.flip(np.roll(np.flip(gammax, axis=(-1,-2)), shift=1, axis=(-1,-2)), axis=1) # \Gamma(-x,-nu)
                                            * np.roll(np.flip(gap_gg[None, :, ...], axis=(-1,-2)), shift=1, axis=(-1,-2)))/2, axis=1) # \DeltaGG(-x,nu)
        gap_new = np.fft.ifftn(gap_new, axes=(-1, -2))
        return gap_new.flatten()

    A = LinearOperator((np.prod(np.shape(gk)),np.prod(np.shape(gk))), matvec=mv)

    #v = get_gap_start(nup, nk, ktype='d').real 
    
    lam, gap = eigsh(A, k=5, which='LA', ncv=100, tol=1e-10, maxiter=100000)
    idx = np.abs(lam-1).argsort()   
    lam = lam[idx]
    gap = gap[:,idx]

    return lam, gap

