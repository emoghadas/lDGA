import numpy as np
from ._fast_bubble import ek_3d, calc_bubble, calc_bubble_gl

# this will be removed
def ek_2d(t=0.25,tpr=0,tsec=0,kpoints=48,q=[0.,0.]):
    "return 2d sqaured lattice Hamiltonian"
    k = np.linspace(0.,2*np.pi,kpoints,endpoint=False)
    kx = np.array(k+q[0])[:,None]
    ky = np.array(k+q[1])[None,:]
    
    return - 2*t*(np.cos(kx) + np.cos(ky))\
               - 4*tpr*np.cos(kx)*np.cos(ky)\
               - 2*tsec*(np.cos(2*kx)+np.cos(2*ky))


def local_bubble(beta: float, g:np.ndarray, iw:int) -> np.ndarray:
    ''' Compute local bubble '''

    if iw>0:
        g_nu = g[iw:-iw]
        g_nu_omega = g[:,:,2*iw:]
    elif iw<0:
        g_nu = g[abs(iw):-abs(iw)]
        g_nu_omega = g[:,:,:-2*abs(iw)]
    else:
        g_nu = g
        g_nu_omega = g

    return -beta * g_nu * g_nu_omega

#TODO: implement 2d fst bubble
def lattice_bubble(beta:float, mu:float, g:np.ndarray, s:np.ndarray, kdim:int, nk:int, iw:int, q:list, niwf:int) -> np.ndarray:
    ''' Compute lattice bubble '''

    # lattice gk
    iW = 1j*2*np.pi*iw / beta
    iws = 1j*(2*np.arange(-niwf,niwf)+1)*np.pi/beta 
    if iw!=0:
        i_nu = iws[abs(iw):-abs(iw)]
    else:
        i_nu = iws
    i_nu_omega = i_nu + iW

    kdim = np.ones(kdim,dtype=int)

    if iw>0:
        s_nu = s[iw:-iw]
        s_nu_omega = s[:,:,2*iw:]
    elif iw<0:
        s_nu = s[abs(iw):-abs(iw)]
        s = s[:,:,:-2*abs(iw)]
    else:
        s_nu = s
        s_nu_omega = s

    ek = ek_2d()
    ekq = ek_2d(q=q)

    i_nu = i_nu.reshape(-1,*kdim)
    gk_nu = 1/(i_nu + mu -ek.reshape(1,*ek.shape) \
                        - s_nu.reshape(*s_nu.shape,*kdim))

    i_nu_omega = i_nu_omega.reshape(-1,*kdim)
    gk_nu_omega = 1/(i_nu_omega + mu -ekq.reshape(1,*ek.shape) \
                        - s_nu_omega.reshape(*s_nu_omega.shape,*kdim))

    return -beta * np.sum(gk_nu*gk_nu_omega, axis=(-1,-2))/ek.size

    
def F_r_local(beta:float, chi0:np.ndarray, g2:np.ndarray, g:np.ndarray, niwf:int, n4iwf:int) -> tuple[np.ndarray, np.ndarray]:
    ''' compute local F_r '''
    # trim bubbles and g
    nu_range = slice(niwf-n4iwf, niwf+n4iwf)
    chi0_tr = chi0[nu_range]
    g_tr = g[nu_range]
    
    # construct local chi
    gg_straight = np.tensordot(g_tr,g_tr,((),()))
    chi_ph = g2 - gg_straight.reshape(1,*gg_straight.shape)
    chi_ph *= beta

    # compute charge and spin F_locs
    chi_c = chi_ph[0,...] + chi_ph[1,...]
    chi_vc = chi_c - np.diag(chi0_tr)
    F_c_loc = -np.diag(1/chi0_tr)@chi_vc@np.diag(1/chi0_tr)*beta**2

    chi_s = chi_ph[0,...] - chi_ph[1,...]
    chi_vc = chi_s - np.diag(chi0_tr)
    F_s_loc = -np.diag(1/chi0_tr)@chi_vc@np.diag(1/chi0_tr)*beta**2

    return F_c_loc, F_s_loc


def chi_ladder_q(beta:float, chi0:np.ndarray, chi0_q:np.ndarray, F_loc_c:np.ndarray, F_loc_s:np.ndarray, niwf:int, n4iwf:int) -> tuple[np.ndarray, np.ndarray]:
    ''' compute lattice chi_q_r '''

    # trim bubbles
    nu_range = slice(niwf-n4iwf, niwf+n4iwf)
    chi0_tr = chi0[nu_range]
    chi0_q_tr = chi0_q[nu_range]

    # non local bubble
    chi0_nl = -(chi0_q_tr - chi0_tr)/beta

    # compute ladder F
    F_q_c = np.linalg.inv(np.eye(chi0_tr.shape[0]) - F_loc_c@np.diag(chi0_nl)/beta)@F_loc_c
    F_q_s = np.linalg.inv(np.eye(chi0_tr.shape[0]) - F_loc_s@np.diag(chi0_nl)/beta)@F_loc_s

    # compute chi_q
    chi_q_c = chi0_q_tr - 1/beta**2 * np.diag(chi0_q_tr)@F_q_c@np.diag(chi0_q_tr)
    chi_q_s = chi0_q_tr - 1/beta**2 * np.diag(chi0_q_tr)@F_q_s@np.diag(chi0_q_tr)

    return chi_q_c, chi_q_s

# TODO: implement asymptotics and three-leg quantities
def gamma_q():
    pass