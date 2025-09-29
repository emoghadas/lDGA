import numpy as np
from numba import jit
from typing import Tuple
from lDGA.config import DGA_ConfigType
from lDGA.utilities import k2ik, build_nu_mats, build_w_mats, U_trans, Udyn, Udyn_arr, G_wq_given_nuk, ek
#from ._fast_bubble import ek_3d, calc_bubble, calc_bubble_gl

@jit(nopython=True)
def asymp_chi(nu, beta) -> np.float64:
    """
    Returns bubble asymptotic -2*beta/nu^2,
    excluding inner fermionic Matsubara frequencies up
    to nu for +/-omega_max = pi/beta*(2*nu+1)
    """
    summ = np.sum(1/(2*np.arange(nu//2)+1)**2)
    return 2*beta*(1/8. - summ/np.pi**2)


@jit(nopython=True)
def chi0_loc_w(dga_cfg : DGA_ConfigType ) -> np.ndarray:
    '''
    Compute local bubble for all iw
    '''
    beta = dga_cfg.beta; g=dga_cfg.g_imp
    n4iwf=dga_cfg.n4iwf; n4iwb=dga_cfg.n4iwb
    chi0_w = np.empty((2*n4iwf,2*n4iwb+1), dtype=np.complex128)
    for w_idx, iw in enumerate(range(-n4iwb,n4iwb+1)):
        if iw>0:
            g_nu = g[iw:-iw]
            g_nu_omega = g[2*iw:]
        elif iw<0:
            g_nu = g[abs(iw):-abs(iw)]
            g_nu_omega = g[:-2*abs(iw)]
        else:
            g_nu = g
            g_nu_omega = g
        chi0 = -beta*g_nu*g_nu_omega
        nu_range = slice(chi0.shape[0]//2-n4iwf,chi0.shape[0]//2+n4iwf)
        chi0_w[...,w_idx] = chi0[nu_range]
    return chi0_w


@jit(nopython=True)
def chi0_loc_w_full(dga_cfg:DGA_ConfigType) -> np.ndarray:  #beta: float, g:np.ndarray, n4iwf:int, n4iwb:int, nouter) -> np.ndarray:
    '''
    Compute local bubble for larger fermionic grid up to n_outer 
    '''
    beta = dga_cfg.beta 
    g = dga_cfg.g_imp
    n4iwb = dga_cfg.n4iwb
    nouter = dga_cfg.nouter
    chi0_w = np.empty((2*nouter,2*n4iwb+1), dtype=np.complex128)
    for w_idx, iw in enumerate(range(-n4iwb,n4iwb+1)):
        if iw>0:
            g_nu = g[iw:-iw]
            g_nu_omega = g[2*iw:]
        elif iw<0:
            g_nu = g[abs(iw):-abs(iw)]
            g_nu_omega = g[:-2*abs(iw)]
        else:
            g_nu = g
            g_nu_omega = g
        chi0 = -beta*g_nu*g_nu_omega
        nu_range = slice(chi0.shape[0]//2-nouter,chi0.shape[0]//2+nouter)
        chi0_w[...,w_idx] = chi0[nu_range]
    return chi0_w


@jit(nopython=True)
def F_r_loc(dga_cfg : DGA_ConfigType) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Compute local vertex F for all iw
    '''
    beta = dga_cfg.beta; chi=dga_cfg.chi_ph
    n4iwf=dga_cfg.n4iwf; n4iwb=dga_cfg.n4iwb
    chi0_w = dga_cfg.chi0_w

    chi_m = chi[0,...] - chi[1,...]
    chi_d = chi[0,...] + chi[1,...]

    F_d_w = np.empty((2*n4iwf, 2*n4iwf, 2*n4iwb+1), dtype=np.complex128)
    F_m_w = np.empty((2*n4iwf, 2*n4iwf, 2*n4iwb+1), dtype=np.complex128)
    for w_idx, iw in enumerate(range(-n4iwb,n4iwb+1)):
        chi_vc = chi_d[:,:,w_idx] - np.diag(chi0_w[:,w_idx])
        F_d_w[:,:,w_idx] = -np.diag(1/chi0_w[:,w_idx])@chi_vc@np.diag(1/chi0_w[:,w_idx])*beta**2

        chi_vc = chi_m[:,:,w_idx] - np.diag(chi0_w[:,w_idx])
        F_m_w[:,:,w_idx] = -np.diag(1/chi0_w[:,w_idx])@chi_vc@np.diag(1/chi0_w[:,w_idx])*beta**2

    return F_d_w, F_m_w

@jit(nopython=True)
def chi_r_loc(dga_cfg:DGA_ConfigType) -> Tuple[np.ndarray, np.ndarray]:
    ''' compute local physical susceptibilities with bubble or bare-u asymptotics '''
    beta = dga_cfg.beta
    u = dga_cfg.U
    omega0 = dga_cfg.w0
    g0 = dga_cfg.g0
    n4iwf = dga_cfg.n4iwf
    n4iwb = dga_cfg.n4iwb
    chi = dga_cfg.chi_ph
    chi0_w = dga_cfg.chi0_w_full
    gamma_d = dga_cfg.gamma_d
    gamma_m = dga_cfg.gamma_m
    niwf = dga_cfg.nouter
    asymp = dga_cfg.asymp

    nu_inner = slice(niwf-n4iwf,niwf+n4iwf)
    wmats  = build_w_mats(n4iwb,beta)

    Uw = Udyn_arr(wmats,omega0,g0,u).astype(np.complex128)
    u_d = 2*Uw - u
    u_m = -u

    chi_d_loc = np.empty((2*n4iwb+1), dtype=np.complex128)
    chi_m_loc = np.empty((2*n4iwb+1), dtype=np.complex128)
    if asymp=='bare-u':
        for w_idx, iw in enumerate(range(-n4iwb, n4iwb+1)):
            phi_d = np.linalg.inv(np.diag(1/chi0_w[nu_inner,w_idx]) + (gamma_d[:,:,w_idx]-u_d[w_idx]*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)
            phi_m = np.linalg.inv(np.diag(1/chi0_w[nu_inner,w_idx]) + (gamma_m[:,:,w_idx]-u_m*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)

            bub_sum = (np.sum(chi0_w[niwf+n4iwf:, w_idx]) + np.sum(chi0_w[:niwf-n4iwf, w_idx])) / beta**2

            chi_d_loc[w_idx] = 1/(1/(np.sum(phi_d)/beta**2 + bub_sum) + u_d[w_idx]) + asymp_chi(2*niwf, beta)
            chi_m_loc[w_idx] = 1/(1/(np.sum(phi_m)/beta**2 + bub_sum) + u_m) + asymp_chi(2*niwf, beta)    
    else:
        for w_idx, iw in enumerate(range(-n4iwb, n4iwb+1)):
            chi_d_gen = chi[0,:,:,w_idx] + chi[1,:,:,w_idx]
            chi_d_phys = np.sum(chi_d_gen)/beta**2 + asymp_chi(2*n4iwf, beta)
            chi_d_loc[w_idx] = chi_d_phys

            chi_m_gen = chi[0,:,:,w_idx] - chi[1,:,:,w_idx]
            chi_m_phys = np.sum(chi_m_gen)/beta**2 + asymp_chi(2*n4iwf, beta)
            chi_m_loc[w_idx] = chi_m_phys

    return chi_d_loc, chi_m_loc


@jit(nopython=True)
def gamma_w(dga_cfg:DGA_ConfigType) -> Tuple[np.ndarray, np.ndarray]: # beta, U, w0, g0, chi0_w, chi, n4iwf, n4iwb, nouter
    ''' Invert local BSE for Gamma_r with asymptotic contribution from bare U(w) '''
    beta = dga_cfg.beta
    u = dga_cfg.U
    w0 = dga_cfg.w0
    g0 = dga_cfg.g0
    n4iwf = dga_cfg.n4iwf
    n4iwb = dga_cfg.n4iwb
    chi = dga_cfg.chi_ph
    chi0_w = dga_cfg.chi0_w_full
    nouter = dga_cfg.nouter


    chi_d = chi[0,...] + chi[1,...]
    chi_m = chi[0,...] - chi[1,...]
    gamma_d = np.empty((2*n4iwf, 2*n4iwf, 2*n4iwb+1), dtype=np.complex128)
    gamma_m = np.empty((2*n4iwf, 2*n4iwf, 2*n4iwb+1), dtype=np.complex128)
    nu_mats = build_nu_mats(nouter, beta)
    w_mats = build_w_mats(n4iwb, beta)
    for w_idx, iw in enumerate(range(-n4iwb,n4iwb+1)):
        Uw = Udyn(w_mats[w_idx], w0, g0, u)
        Uw_array = np.ones((2*nouter, 2*nouter))*Uw
        Unu = U_trans(nu_mats, nu_mats, w0, g0, u)
        U_d = 2*Uw_array - Unu
        U_m = -Unu
        
        chi_t_d = np.linalg.inv(np.diag(1/chi0_w[:,w_idx]) + U_d/beta**2)
        chi_t_m = np.linalg.inv(np.diag(1/chi0_w[:,w_idx]) + U_m/beta**2)

        nu_range = slice(nouter-n4iwf,nouter+n4iwf)

        dgamma_d = beta**2*(np.linalg.inv(chi_d[:,:,w_idx]) - np.linalg.inv(chi_t_d[nu_range, nu_range]))
        dgamma_m = beta**2*(np.linalg.inv(chi_m[:,:,w_idx]) - np.linalg.inv(chi_t_m[nu_range, nu_range]))

        gamma_m[:,:,w_idx] = dgamma_m + U_m[nu_range, nu_range]
        gamma_d[:,:,w_idx] = dgamma_d + U_d[nu_range, nu_range]

    return gamma_d, gamma_m


@jit(nopython=True)
def chi0_w_q(dga_cfg : DGA_ConfigType , mu:np.float64, s_dga:np.ndarray=None) -> np.ndarray:
    '''
    Compute lattice bubble chi0 for all iw and range of q-points
    '''
    beta=dga_cfg.beta; s_dmft=dga_cfg.s_imp
    n4iwf=dga_cfg.n4iwf;  n4iwb=dga_cfg.n4iwb
    ts=dga_cfg.ts
    t1=ts[0]
    t2=ts[1]
    k_grid = dga_cfg.k_grid
    nk = k_grid.shape[0]
    qpoints = dga_cfg.q_grid_loc
    niwf  = s_dmft.shape[0]//2
    chi0_wq = np.zeros((2*n4iwf,2*n4iwb+1,qpoints.shape[0]), dtype=np.complex128)

    nu_array=build_nu_mats(n4iwf, beta)
    for inu,nu in enumerate(nu_array):
        for ik,k in enumerate(k_grid):
            G_nuw_kq = G_wq_given_nuk(nu,k,s_dmft,n4iwb,qpoints,beta,mu,ts,s_dga)
            if(s_dga is None):
                G_nu_k = 1.0/(1j*nu - ek(k,t=t1,tpr=t2) + mu - s_dmft[inu-n4iwf+niwf] )
            else:
                G_nu_k = 1.0/(1j*nu - ek(k, t=t1,tpr=t2) + mu - s_dga[inu,ik] )
            chi0_wq[inu,:,:] += G_nu_k*G_nuw_kq
    chi0_wq *= -beta/nk
    return chi0_wq


@jit(nopython=True)
def chi_v_r_w_q(dga_cfg:DGA_ConfigType , chi0_w_q:np.ndarray) \
         -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # chi_d, gamma_d, A,  chi_m, gamma_m, A
    '''
    Compute physical susceptibility and three leg vertex of the lattice for all iw and given q-points
    '''
    beta=dga_cfg.beta; u=dga_cfg.U
    omega0=dga_cfg.w0; g0=dga_cfg.g0
    n4iwf=dga_cfg.n4iwf; n4iwb=dga_cfg.n4iwb
    chi=dga_cfg.chi_ph
    chi0_w = dga_cfg.chi0_w
    qpoints = dga_cfg.q_grid_loc
    
    chi_d = chi[0,...] + chi[1,...]
    chi_m = chi[0,...] - chi[1,...]

    chi_d_w_q  = np.empty((2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    v_d_w_q    = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    A_d_w_q = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    
    chi_m_w_q  = np.empty((2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    v_m_w_q    = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    A_m_w_q = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    
    numats = build_nu_mats(n4iwf,beta)
    wmats  = build_w_mats(n4iwb,beta)

    Uw = Udyn_arr(wmats,omega0,g0,u).astype(np.complex128)
    Ununup = U_trans(nu=numats,nup=numats, omega0=omega0, g=g0, u=u).astype(np.complex128)

    u_d = 2*Uw - u
    u_m = -u

    for q_idx, q in enumerate(qpoints):
        
        for w_idx, iw in enumerate(range(-n4iwb, n4iwb+1)):
            
            chi_d_q = np.linalg.inv(np.linalg.inv(chi_d[...,w_idx]) + np.diag(1/(chi0_w_q[:,w_idx,q_idx]) - 1/(chi0_w[...,w_idx])))
            chi_m_q = np.linalg.inv(np.linalg.inv(chi_m[...,w_idx]) + np.diag(1/(chi0_w_q[:,w_idx,q_idx]) - 1/(chi0_w[...,w_idx])))

            chi_phys_d_q = np.sum(chi_d_q)/beta**2 + asymp_chi(2*n4iwf, beta)
            chi_phys_m_q = np.sum(chi_m_q)/beta**2 + asymp_chi(2*n4iwf, beta)

            # compute three-leg vertex
            v_d_q  = np.sum(np.diag(1/chi0_w_q[:,w_idx,q_idx])@chi_d_q, axis=1)/(1 - u_d[w_idx]*chi_phys_d_q)
            v_m_q  = np.sum(np.diag(1/chi0_w_q[:,w_idx,q_idx])@chi_m_q, axis=1)/(1 - u_m*chi_phys_m_q)

            #compute phi
            phi_d_q = chi_d_q + np.outer(chi0_w_q[:,w_idx,q_idx]*v_d_q, chi0_w_q[:,w_idx,q_idx]*v_d_q)*u_d[w_idx]*(1-u_d[w_idx]*chi_phys_d_q)/beta**2
            phi_m_q = chi_m_q + np.outer(chi0_w_q[:,w_idx,q_idx]*v_m_q, chi0_w_q[:,w_idx,q_idx]*v_m_q)*u_m*(1-u_m*chi_phys_m_q)/beta**2

            # compute three-leg vertex A
            A_d_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_d_q @ Ununup )*beta
            A_m_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_m_q @ Ununup )*beta

            # store quantities
            chi_d_w_q[w_idx,q_idx] = chi_phys_d_q
            v_d_w_q[:,w_idx,q_idx] = v_d_q
            A_d_w_q[:,w_idx,q_idx] = A_d_q
            
            chi_m_w_q[w_idx,q_idx] = chi_phys_m_q
            v_m_w_q[:,w_idx,q_idx] = v_m_q
            A_m_w_q[:,w_idx,q_idx] = A_m_q

    return chi_d_w_q, v_d_w_q, A_d_w_q,   chi_m_w_q, v_m_w_q, A_m_w_q


@jit(nopython=True)
def bse_asymp(dga_cfg:DGA_ConfigType, chi0_w_q:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ''' non-local BSE with asymptotic knowledge of bare U(w) '''
    beta=dga_cfg.beta 
    u=dga_cfg.U
    omega0=dga_cfg.w0
    g0=dga_cfg.g0
    niwf = dga_cfg.nouter
    n4iwf=dga_cfg.n4iwf
    n4iwb=dga_cfg.n4iwb
    chi=dga_cfg.chi_ph
    gamma_d = dga_cfg.gamma_d
    gamma_m = dga_cfg.gamma_m
    chi0_w = dga_cfg.chi0_w_full
    qpoints = dga_cfg.q_grid_loc

    chi_d_gen = chi[0,...] + chi[1,...]
    chi_m_gen = chi[0,...] - chi[1,...]

    chi_d_w_q  = np.empty((2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    v_d_w_q    = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    A_d_w_q = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    
    chi_m_w_q  = np.empty((2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    v_m_w_q    = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    A_m_w_q = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    
    numats = build_nu_mats(n4iwf,beta)
    wmats  = build_w_mats(n4iwb,beta)

    Uw = Udyn_arr(wmats,omega0,g0,u).astype(np.complex128)
    Ununup = U_trans(nu=numats,nup=numats, omega0=omega0, g=g0, u=0).astype(np.complex128)

    u_d = 2*Uw - u
    u_m = -u

    for q_idx, q in enumerate(qpoints):
        for w_idx, iw in enumerate(range(-n4iwb, n4iwb+1)):
            phi_d = np.linalg.inv(np.diag(1/chi0_w_q[:,w_idx,q_idx]) + (gamma_d[:,:,w_idx]-u_d[w_idx]*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)
            phi_m = np.linalg.inv(np.diag(1/chi0_w_q[:,w_idx,q_idx]) + (gamma_m[:,:,w_idx]-u_m*np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128))/beta**2)

            bub_sum = (np.sum(chi0_w[niwf+n4iwf:, w_idx]) + np.sum(chi0_w[:niwf-n4iwf, w_idx])) / beta**2

            chi_d = 1/(1/(np.sum(phi_d)/beta**2 + bub_sum) + u_d[w_idx]) 
            chi_m = 1/(1/(np.sum(phi_m)/beta**2 + bub_sum) + u_m)

            v_d = np.sum(np.diag(1/chi0_w_q[:,w_idx,q_idx])@phi_d, axis=1) * (1-u_d[w_idx]*chi_d)/(1-u_d[w_idx]*(chi_d+asymp_chi(2*niwf, beta)))
            v_m = np.sum(np.diag(1/chi0_w_q[:,w_idx,q_idx])@phi_m, axis=1) * (1-u_m*chi_m)/(1-u_m*(chi_m+asymp_chi(2*niwf, beta)))
            #v_d = np.sum(np.diag(1/chi0_w_q[:,w_idx,q_idx])@chi_d_gen[...,w_idx], axis=1) /(1-u_d[w_idx]*(chi_d+asymp_chi(2*niwf, beta)))
            #v_m = np.sum(np.diag(1/chi0_w_q[:,w_idx,q_idx])@chi_m_gen[...,w_idx], axis=1) /(1-u_m*(chi_m+asymp_chi(2*niwf, beta)))

            one = np.ones((2*n4iwf,2*n4iwf), dtype=np.complex128)
            phi_d = phi_d - phi_d@(u_d[w_idx]*one)@phi_d*(1-u_d[w_idx]*chi_d)/beta**2 + phi_d@(u_d[w_idx]*one)@phi_d*(1-u_d[w_idx]*chi_d)**2/(1-u_d[w_idx]*(chi_d+asymp_chi(2*niwf, beta)))/beta**2
            phi_m = phi_m - phi_m@(u_m*one)@phi_m*(1-u_m*chi_m)/beta**2 + phi_m@(u_m*one)@phi_m*(1-u_m*chi_m)**2/(1-u_m*(chi_m+asymp_chi(2*niwf, beta)))/beta**2

            #compute phi
            chi_d += asymp_chi(2*niwf, beta)
            chi_m += asymp_chi(2*niwf, beta)

            #phi_d = chi_d_gen[...,w_idx] + np.outer(chi0_w_q[:,w_idx,q_idx]*v_d, chi0_w_q[:,w_idx,q_idx]*v_d)*u_d[w_idx]*(1-u_d[w_idx]*chi_d)/beta**2
            #phi_m = chi_m_gen[...,w_idx] + np.outer(chi0_w_q[:,w_idx,q_idx]*v_m, chi0_w_q[:,w_idx,q_idx]*v_m)*u_m*(1-u_m*chi_m)/beta**2

            # compute three-leg vertex A
            A_d_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_d @ Ununup )*beta
            A_m_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_m @ Ununup )*beta

            chi_d_w_q[w_idx,q_idx] = chi_d
            v_d_w_q[:,w_idx,q_idx] = v_d
            A_d_w_q[:,w_idx,q_idx] = A_d_q

            chi_m_w_q[w_idx,q_idx] = chi_m 
            v_m_w_q[:,w_idx,q_idx] = v_m
            A_m_w_q[:,w_idx,q_idx] = A_m_q

    return chi_d_w_q, v_d_w_q, A_d_w_q, chi_m_w_q, v_m_w_q, A_m_w_q


@jit(nopython=True)
def dual_bse(dga_cfg:DGA_ConfigType, chi0_w_q:np.ndarray):
    ''' Calculates physical lattice susceptibility and lattice hedin vertex for fixed q and omega based on the dual-scheme'''
    beta = dga_cfg.beta 
    u = dga_cfg.U
    omega0 = dga_cfg.w0
    g0 = dga_cfg.g0
    chi0_w = dga_cfg.chi0_w
    n2iwb = dga_cfg.n3iwb
    n3iwf = dga_cfg.n3iwf
    n3iwb = dga_cfg.n3iwb
    n4iwf = dga_cfg.n4iwf
    n4iwb = dga_cfg.n4iwb
    chi_d_loc = dga_cfg.chi_d_loc    
    chi_m_loc = dga_cfg.chi_m_loc
    p3ph = dga_cfg.p3ph
    l3_d = p3ph[0,...] + p3ph[1,...]
    l3_m = p3ph[0,...] - p3ph[1,...]
    chi_ph = dga_cfg.chi_ph
    chi_d = chi_ph[0,...] + chi_ph[1,...]
    chi_m = chi_ph[0,...] - chi_ph[1,...]
    F_d_loc = dga_cfg.F_d_loc
    F_m_loc = dga_cfg.F_m_loc
    qpoints = dga_cfg.q_grid_loc

    numats = build_nu_mats(n4iwf,beta)
    wmats  = build_w_mats(n4iwb,beta)

    Uw = Udyn_arr(wmats,omega0,g0,u).astype(np.complex128)
    Ununup = U_trans(nu=numats,nup=numats, omega0=omega0, g=g0, u=u).astype(np.complex128)

    u_d = 2*Uw - u
    u_m = -u

    chi_d_w_q = np.empty((2*n4iwb+1, len(qpoints)), dtype=np.complex128)
    chi_m_w_q = np.empty((2*n4iwb+1, len(qpoints)), dtype=np.complex128)
    v_d_w_q = np.empty((2*n4iwf, 2*n4iwb+1, len(qpoints)), dtype=np.complex128)
    v_m_w_q = np.empty((2*n4iwf, 2*n4iwb+1, len(qpoints)), dtype=np.complex128)
    A_d_w_q = np.empty((2*n4iwf, 2*n4iwb+1, len(qpoints)), dtype=np.complex128)
    A_m_w_q = np.empty((2*n4iwf, 2*n4iwb+1, len(qpoints)), dtype=np.complex128)
    for q_idx, q in enumerate(qpoints):
        for w_idx, iw in enumerate(range(-n4iwb, n4iwb+1)):
            ##### dual bse #####

            # nonlocal bubble
            chi0_nl = chi0_w_q[:,w_idx,q_idx] - chi0_w[:,w_idx]

            # nonlocal 4point
            chi_nl_d = np.linalg.inv(np.linalg.inv(np.diag(chi0_nl)) + F_d_loc[...,w_idx]/beta**2)
            chi_nl_m = np.linalg.inv(np.linalg.inv(np.diag(chi0_nl)) + F_m_loc[...,w_idx]/beta**2)

            # include local chi
            chi_d_q = chi_d_loc[w_idx] + l3_d[:,w_idx]@chi_nl_d@l3_d[:,w_idx]
            chi_m_q = chi_m_loc[w_idx] + l3_m[:,w_idx]@chi_nl_m@l3_m[:,w_idx]
            chi_d_w_q[w_idx, q_idx] = chi_d_q
            chi_m_w_q[w_idx, q_idx] = chi_m_q

            #### ladder equation for lattice hedin vertex ####
            hedin_d = beta*l3_d/(1 - u_d[w_idx]*chi_d_loc[w_idx])
            hedin_m = beta*l3_m/(1 - u_m*chi_m_loc[w_idx])

            u_scr_d = u_d[w_idx] - (u_d[w_idx]**2*chi_d_loc[w_idx])
            u_scr_m = u_m - (u_m**2*chi_m_loc[w_idx])

            F_d_irr_loc = F_d_loc[...,w_idx] - u_scr_d * np.outer(hedin_d[:,w_idx], hedin_d[:,w_idx])
            F_m_irr_loc = F_m_loc[...,w_idx] - u_scr_m * np.outer(hedin_m[:,w_idx], hedin_m[:,w_idx])

            v_d = np.linalg.solve(np.eye(2*n4iwf) + F_d_irr_loc.T@np.diag(chi0_nl)/(beta**2), hedin_d[:,w_idx])
            v_m = np.linalg.solve(np.eye(2*n4iwf) + F_m_irr_loc.T@np.diag(chi0_nl)/(beta**2), hedin_m[:,w_idx])  

            v_d_w_q[:, w_idx, q_idx] = v_d 
            v_m_w_q[:, w_idx, q_idx] = v_m 

            # lattice polarization
            F_d_q = np.linalg.inv(np.eye(2*n4iwf) + F_d_loc[...,w_idx]@np.diag(chi0_nl/beta)/beta)@F_d_loc[...,w_idx]
            F_m_q = np.linalg.inv(np.eye(2*n4iwf) + F_m_loc[...,w_idx]@np.diag(chi0_nl/beta)/beta)@F_m_loc[...,w_idx]

            u_scr_d_q = u_d[w_idx] - (u_d[w_idx]**2*chi_d_q)
            u_scr_m_q = u_m - (u_m**2*chi_m_q)

            F_d_irr_q = F_d_q - u_scr_d_q * np.outer(v_d, v_d)
            F_m_irr_q = F_m_q - u_scr_m_q * np.outer(v_m, v_m)
            phi_d = np.diag(chi0_w_q[:,w_idx,q_idx]) - np.diag(chi0_w_q[:,w_idx,q_idx])@F_d_irr_q@np.diag(chi0_w_q[:,w_idx,q_idx])/beta**2
            phi_m = np.diag(chi0_w_q[:,w_idx,q_idx]) - np.diag(chi0_w_q[:,w_idx,q_idx])@F_m_irr_q@np.diag(chi0_w_q[:,w_idx,q_idx])/beta**2

            # compute three-leg vertex A
            A_d_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_d @ Ununup )*beta
            A_m_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_m @ Ununup )*beta

            A_d_w_q[:, w_idx, q_idx] = A_d_q
            A_m_w_q[:, w_idx, q_idx] = A_m_q

    return chi_d_w_q, v_d_w_q, A_d_w_q, chi_m_w_q, v_m_w_q, A_m_w_q