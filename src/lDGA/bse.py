import numpy as np
from numba import jit
from lDGA.utilities import k2ik, build_nu_mats, build_w_mats, U_trans, Udyn_arr, G_wq_given_nuk, ek_2d
#from ._fast_bubble import ek_3d, calc_bubble, calc_bubble_gl

@jit(nopython=True)
def asymp_chi(nu, beta):
    """
    Returns bubble asymptotic -2*beta/nu^2,
    excluding inner fermionic Matsubara frequencies up
    to nu for +/-omega_max = pi/beta*(2*nu+1)
    """
    summ = np.sum(1/(2*np.arange(nu//2)+1)**2)
    return 2*beta*(1/8. - summ/np.pi**2)




@jit(nopython=True)
def chi0_w_q(beta:float, mu:float, s_dmft:np.ndarray, k_grid:np.ndarray, qpoints: np.ndarray, n4iwf:int, n4iwb:int, s_dga:np.ndarray=None) -> np.ndarray:
    '''
    Compute lattice bubble chi0 for all iw and range of q-points
    '''
    nk = k_grid.shape[0]
    niwf  = s_dmft.shape[0]//2
    chi0_wq = np.zeros((2*n4iwf,2*n4iwb+1,qpoints.shape[0]), dtype=np.complex128)
    

    nu_array=build_nu_mats(n4iwf, beta)
    for inu,nu in enumerate(nu_array):
        for ik,k in enumerate(k_grid):
            G_nuw_kq = G_wq_given_nuk(nu,k,s_dmft,n4iwb,qpoints,beta,mu, s_dga)
            if(s_dga is None):
                G_nu_k = 1.0/(1j*nu - ek_2d(k, t=1) + mu - s_dmft[inu-n4iwf+niwf] )
            else:
                G_nu_k = 1.0/(1j*nu - ek_2d(k, t=1) + mu - s_dga[inu,ik] )
            chi0_wq[inu,:,:] += G_nu_k*G_nuw_kq
    chi0_wq *= -beta/nk
    return chi0_wq

@jit(nopython=True)
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


@jit(nopython=True)
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


@jit(nopython=True)
def chi_v_r_w_q(beta:float, u:np.float64, omega0:np.float64, g:np.float64 , chi0_w:np.ndarray, chi0_w_q:np.ndarray, chi:np.ndarray, n4iwf:int, n4iwb:int, qpoints:np.ndarray) \
         -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: # chi_d, gamma_d, A,  chi_m, gamma_m, A
    '''
    Compute physical susceptibility and three leg vertex of the lattice for all iw and given q-points
    '''
    
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

            #compute phi
            phi_d_q = chi_d_q + np.outer(chi0_w_q[:,w_idx,q_idx]*v_d_q, chi0_w_q[:,w_idx,q_idx]*v_d_q)*u_d[w_idx]*(1-u_d[w_idx]*chi_phys_d_q)/beta**2
            phi_m_q = chi_m_q + np.outer(chi0_w_q[:,w_idx,q_idx]*v_m_q, chi0_w_q[:,w_idx,q_idx]*v_m_q)*u_m*(1-u_m*chi_phys_m_q)/beta**2

            # compute three-leg vertex A
            A_d_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_d_q @ Ununup )*beta
            A_m_q = (1/chi0_w_q[:,w_idx,q_idx])*np.diag(  phi_m_q @ Ununup )*beta

            # store quantities
            chi_d_w_q[w_idx,q_idx] =  chi_phys_d_q
            v_d_w_q[:,w_idx,q_idx] = v_d_q
            A_d_w_q[:,w_idx,q_idx] = A_d_q
            #
            chi_m_w_q[w_idx,q_idx] =  chi_phys_m_q
            v_m_w_q[:,w_idx,q_idx] = v_m_q
            A_m_w_q[:,w_idx,q_idx] = A_m_q

    return chi_d_w_q, v_d_w_q, A_d_w_q,   chi_m_w_q, v_m_w_q, A_m_w_q