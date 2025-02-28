import numpy as np
import numba
from lDGA.utilities import k2ik
#from ._fast_bubble import ek_3d, calc_bubble, calc_bubble_gl


def ek_2d(t:float=0.25,tpr:float=0,tsec:float=0,kpoints:int=48,q:list[float,float]=[0.,0.]) -> np.ndarray:
    "return 2d sqaured lattice Hamiltonian"
    k = np.linspace(-np.pi,np.pi, kpoints, endpoint=False)
    kx = np.array(k+q[0])[:,None]
    ky = np.array(k+q[1])[None,:]
    
    ek = - 2*t*(np.cos(kx) + np.cos(ky))\
               - 4*tpr*np.cos(kx)*np.cos(ky)\
               - 2*tsec*(np.cos(2*kx)+np.cos(2*ky))
    return ek


def gk_w(beta:float, mu:float, g:np.ndarray, s:np.ndarray, kdim:int, nk:int, iw:int, niwf:int) -> np.ndarray:
    '''
    Compute lattice GF for given iw and q
    '''
    iW = 1j*2*np.pi*iw / beta
    ivs = 1j*(2*np.arange(-niwf,niwf)+1)*np.pi/beta 
    if iw!=0:
        ivs = ivs[abs(iw):-abs(iw)]
    else:
        ivs = ivs
    iv_w = ivs + iW

    kdim = np.ones((kdim,),dtype=np.int32)

    if iw>0:
        s_v_w = s[2*iw:]
    elif iw<0:
        s_v_w = s[:-2*abs(iw)]
    else:
        s_v_w = s

    ek = ek_2d(kpoints=nk).flatten()

    iv_w = iv_w.reshape(-1,1)
    s_v_w = s_v_w.reshape(-1,1)
    ek = ek.reshape(1,-1)
    
    gk_v_w = 1/(iv_w + mu - s_v_w - ek)
    
    return gk_v_w


# TODO: test with looped bubble calculation, as fft underestimates chi_q
def chi0_q_w(beta:float, mu:float, g:np.ndarray, s:np.ndarray, kdim:int, nk:int, niwf:int, n4iwf:int, n4iwb:int) -> np.ndarray:
    '''
    Compute lattice bubble for all q-points and iw
    '''
    chi0_qw = np.empty((2*n4iwf,2*n4iwb+1,nk**kdim), dtype=np.complex128)
    for w_idx,iw in enumerate(range(-n4iwb,n4iwb+1)):
        # get lattic gf with correct shifts
        gk_v = gk_w(beta, mu, g, s, kdim, nk, 0, niwf)
        if iw!=0:
            gk_v = gk_v[abs(iw):-abs(iw),...]
        gk_v_w = gk_w(beta, mu, g, s, kdim, nk, iw, niwf)

        # fft
        gr = np.fft.fftn(gk_v, axes=(-1,))
        gr_w = np.fft.fftn(gk_v_w, axes=(-1,))
        chi0 = -beta*np.fft.ifftn(gr*gr_w, axes=(-1,))/nk**kdim

        # truncate nu tails
        nu_range = slice(chi0.shape[0]//2-n4iwf,chi0.shape[0]//2+n4iwf)
        chi0 = chi0[nu_range]

        chi0_qw[:,w_idx,:] = chi0
    return chi0_qw


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


#TODO: extract loca lFs for all w
def F_r_loc():
    pass


@numba.jit(nopython=True)
def chi_v_r_q_w(beta:float, u:np.ndarray, chi0_w:np.ndarray, chi0_q_w:np.ndarray, g2:np.ndarray, niwf:int, n4iwf:int, n4iwb:int, qpoints:np.ndarray, nk:int) \
         -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Compute physical susceptibility and three leg vertex of the lattice for all iw and given q-points
    '''
    
    chi_d = g2[0,...] + g2[1,...]
    chi_m = g2[0,...] - g2[1,...]

    chi_d_q_w = np.empty((2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    v_d_q_w = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    
    chi_m_q_w = np.empty((2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    v_m_q_w = np.empty((2*n4iwf, 2*n4iwb+1, qpoints.shape[0]), dtype=np.complex128)
    for q in qpoints:
        for w_idx, iw in enumerate(range(-n4iwb, n4iwb+1)):
            # find q idx
            q_idx = k2ik(q, nk)
            q_idx = 0

            chi_d_q = np.linalg.inv(np.linalg.inv(chi_d[...,w_idx]) + np.diag(1/(chi0_q_w[:,w_idx,q_idx]) - 1/(chi0_w[...,w_idx])))
            chi_m_q = np.linalg.inv(np.linalg.inv(chi_m[...,w_idx]) + np.diag(1/(chi0_q_w[:,w_idx,q_idx]) - 1/(chi0_w[...,w_idx])))

            chi_phys_d_q = np.sum(chi_d_q)/beta**2
            chi_phys_m_q = np.sum(chi_m_q)/beta**2

            # compute three-leg vertex
            v_d_q = np.sum(np.diag(1/chi0_q_w[:,w_idx,q_idx])@chi_d_q, axis=1)/(1 - u[w_idx]*chi_phys_d_q)
            v_m_q = np.sum(np.diag(1/chi0_q_w[:,w_idx,q_idx])@chi_m_q, axis=1)/(1 + u[w_idx]*chi_phys_m_q)

            # store quantities
            chi_d_q_w[w_idx, q_idx] =  chi_phys_d_q
            v_d_q_w[:,w_idx,q_idx] = v_d_q
            chi_m_q_w[w_idx, q_idx] =  chi_phys_m_q
            v_m_q_w[:,w_idx,q_idx] = v_m_q
        
    return chi_d_q_w, v_d_q_w, chi_m_q_w, v_m_q_w