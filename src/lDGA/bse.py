import numpy as np
import numba
from lDGA.utilities import k2ik, build_nu_mats, build_w_mats, U_trans, Udyn_arr
#from ._fast_bubble import ek_3d, calc_bubble, calc_bubble_gl

@numba.njit
def asymp_chi(nu, beta):
    """
    Returns bubble asymptotic -2*beta/nu^2,
    excluding inner fermionic Matsubara frequencies up
    to nu for +/-omega_max = pi/beta*(2*nu+1)
    """
    summ = np.sum(1/(2*np.arange(nu//2)+1)**2)
    return 2*beta*(1/8. - summ/np.pi**2)

@numba.njit
def ek_2d(k:np.ndarray, t:float=0.25, tpr:float=0, tsec:float=0) -> np.float64:
    '''
    return 2d sqaured lattice Hamiltonian evaluated at give k-point
    '''
    kx = k[0]
    ky = k[1]
    
    ek = - 2*t*(np.cos(kx) + np.cos(ky))\
               - 4*tpr*np.cos(kx)*np.cos(ky)\
               - 2*tsec*(np.cos(2*kx)+np.cos(2*ky))
    return ek


@numba.njit
def chi0_w_q(beta:float, mu:float, s:np.ndarray, k_grid:np.ndarray, qpoints: np.ndarray, niwf:int, n4iwf:int, n4iwb:int) -> np.ndarray:
    '''
    Compute lattice bubble chi0 for all iw and range of q-points
    '''
    nk = k_grid.shape[0]
    chi0_wq = np.empty((2*n4iwf,2*n4iwb+1,qpoints.shape[0]), dtype=np.complex128)
    for q_idx,q in enumerate(qpoints):
        for w_idx,iw in enumerate(range(-n4iwb,n4iwb+1)):
            # correct frequency shifts
            iW = 1j*2*np.pi*iw / beta
            iv = 1j*(2*np.arange(-niwf,niwf)+1)*np.pi/beta 
            if iw!=0:
                iv = iv[abs(iw):-abs(iw)]
            else:
                iv = iv
            iv_w = iv + iW

            if iw>0:
                s_v = s[abs(iw):-abs(iw)]
                s_v_w = s[2*iw:]
            elif iw<0:
                s_v = s[abs(iw):-abs(iw)]
                s_v_w = s[:-2*abs(iw)]
            else:
                s_v = s
                s_v_w = s

            bub = np.zeros((s_v.shape[0],), dtype=np.complex128)
            for k in k_grid:
                g_v = 1/(iv - ek_2d(k, t=1) + mu - s_v)
                g_v_w = 1/(iv_w - ek_2d(k+q, t=1) + mu - s_v_w)
                bub = bub - beta * (g_v*g_v_w) / nk

            nu_range = slice(bub.shape[0]//2-n4iwf, bub.shape[0]//2+n4iwf)
            bub = bub[nu_range]
            
            chi0_wq[:,w_idx,q_idx] = bub
    return chi0_wq


@numba.njit
def chi0_loc_w(beta: float, g:np.ndarray, n4iwf:int, n4iwb:int) -> np.ndarray:
    '''
    Compute local bubble for all iw
    '''
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


@numba.njit
def F_r_loc(beta:float, chi0_w:np.ndarray, chi:np.ndarray, n4iwf:int, n4iwb:int) -> tuple[np.ndarray, np.ndarray]:
    '''
    Compute local vertex F for all iw
    '''
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


@numba.njit
def chi_v_r_w_q(beta:float, u:np.float64, omega0:np.float64, g:np.float64 , chi0_w:np.ndarray, chi0_w_q:np.ndarray, chi:np.ndarray, n4iwf:int, n4iwb:int, qpoints:np.ndarray) \
         -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # chi_d, gamma_d, A,  chi_m, gamma_m, A
    '''
    Compute physical susceptibility and three leg vertex of the lattice for all iw and given q-points
    '''
    
    chi_d = chi[0,...] + chi[1,...]
    chi_m = chi[0,...] - chi[1,...]

    chi_d_w_q  = np.empty((2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    v_d_w_q    = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    vR_d_w_q   = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    A_d_w_q = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    
    chi_m_w_q  = np.empty((2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    v_m_w_q    = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    vR_m_w_q   = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    A_m_w_q = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    
    numats = build_nu_mats(n4iwf,beta)
    wmats  = build_w_mats(n4iwb,beta)

    Uw = Udyn_arr(wmats,omega0,g,u).astype(np.complex128)
    Ununup = U_trans(nu=numats,nup=numats, omega0=omega0, g=g, u=u).astype(np.complex128)

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

            vR_d_q = np.sum(chi_d_q@np.diag(1/chi0_w_q[:,w_idx,q_idx]), axis=0)/(1 - u_d[w_idx]*chi_phys_d_q)
            vR_m_q = np.sum(chi_m_q@np.diag(1/chi0_w_q[:,w_idx,q_idx]), axis=0)/(1 - u_m*chi_phys_m_q)

            #compute phi
            phi_d_q = chi_d_q + np.outer(chi0_w_q[:,w_idx,q_idx]*v_d_q, chi0_w_q[:,w_idx,q_idx]*v_d_q)*u_d[w_idx]*(1-u_d[w_idx]*chi_phys_d_q)/beta**2
            phi_m_q = chi_m_q + np.outer(chi0_w_q[:,w_idx,q_idx]*v_m_q, chi0_w_q[:,w_idx,q_idx]*v_m_q)*u_m*(1-u_m*chi_phys_m_q)/beta**2

            # compute three-leg vertex A
            A_d_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_d_q @ Ununup )*beta
            A_m_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_m_q @ Ununup )*beta

            # store quantities
            chi_d_w_q[w_idx,q_idx] =  chi_phys_d_q
            v_d_w_q[:,w_idx,q_idx] = v_d_q
            vR_d_w_q[:,w_idx,q_idx] = vR_d_q
            A_d_w_q[:,w_idx,q_idx] = A_d_q
            #
            chi_m_w_q[w_idx,q_idx] =  chi_phys_m_q
            v_m_w_q[:,w_idx,q_idx] = v_m_q
            vR_m_w_q[:,w_idx,q_idx] = vR_m_q
            A_m_w_q[:,w_idx,q_idx] = A_m_q

    return chi_d_w_q, v_d_w_q, vR_d_w_q, A_d_w_q,   chi_m_w_q, v_m_w_q, vR_m_w_q, A_m_w_q