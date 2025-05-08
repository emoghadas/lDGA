import numpy as np
from scipy.optimize import root
from numba import jit

##### LATTICE UTILITIES #####

@jit(nopython=True)
def ik2k(ik:int, dim:int, Nk:int) -> np.ndarray:
    k = np.zeros(dim, dtype=np.float64)
    ikres=ik*1
    for idim in range(dim):
        ik_dim = ikres % Nk
        k[idim] = np.pi*(-1.0 + 2.0*ik_dim/Nk)
        ikres = ikres // Nk
    return k

@jit(nopython=True)
def k2ik(k:np.ndarray, Nk:int) -> int:
    dim = len(k)
    ik = 0
    for idim in range(dim):
        ik += (Nk*k[idim]/(2.0*np.pi) + 0.5*Nk) % Nk * Nk**idim
    return int(np.round(ik))

@jit(nopython=True)
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

# DYNAMICAL UTILS
@jit(nopython=True)
def inu2nu(inu:int, beta:np.float64) -> np.float64:
    return np.pi*(2.0*inu + 1.0)/beta

@jit(nopython=True)
def nu2inu(nu:np.float64, beta:np.float64) -> int:
    return int(np.round((nu*beta/np.pi - 1.0)/2.0))

@jit(nopython=True)
def Udyn(omega:np.float64, omega0:np.float64, g:np.float64, u:np.float64 = 0.0) -> np.float64:
    return u - 2*omega0*g**2/(omega0**2 + omega**2)

@jit(nopython=True)
def Udyn_arr(omega:np.ndarray , omega0:np.float64, g:np.float64, u:np.float64 = 0.0) -> np.ndarray:
    return u - 2*omega0*g**2/(omega0**2 + omega**2)

@jit(nopython=True)
def U_trans(nu:np.ndarray, nup:np.ndarray, omega0:np.float64, g:np.float64, u:np.float64 = 0.0) -> np.ndarray:
    omega = nup.reshape(1, nup.shape[0]) - nu.reshape(nu.shape[0], 1)
    return u - 2*omega0*g**2/(omega0**2 + omega**2)

@jit(nopython=True)
def build_w_mats(Nw:int, beta:np.float64) -> np.ndarray:
    return (2.0*np.pi/beta)*np.linspace(-Nw,Nw,2*Nw+1)

@jit(nopython=True)
def build_nu_mats(Nnu:int, beta:np.float64) -> np.ndarray:
    return (2.0*np.pi/beta)*(np.linspace(-Nnu,Nnu-1,2*Nnu)+0.5)

##### GREEN'S FUNCTION UTILITIES #####

#For the moment square (dim-dimensional hypercubic) dispersion only (using t=1)
#we may want to generalize it later using jitclass for a "model" object
#containing a method retrieving the dispersion

#N.B. this works only for DMFT self-energy with enough frequencies.
# Auxiliary function for Swinger-Dyson Equations
@jit(nopython=True)
def G_wq_given_nuk(nu:np.float64, k:np.ndarray, sigma:np.ndarray, n4iwb:int, qpoints:np.ndarray, beta:np.float64, mu:np.float64, sigma_dga:np.ndarray=None )-> np.ndarray:
    dim = len(k); inu=nu2inu(nu, beta)
    Nq, dimq = qpoints.shape
    Gres = np.zeros( (2*n4iwb+1,Nq), dtype=np.complex128 )
    niwf = sigma.shape[0]//2
    n4iwf = 0
    if(not(sigma_dga is None)):
        Nk = sigma_dga.shape[1]
        Nk_lin = int(np.round(Nk**(1/dim)))
        n4iwf = sigma_dga.shape[0]//2
    for iq,q in enumerate(qpoints):
        eps_kq = np.complex128(ek_2d(k+q, 1.0))

        for iw in range(-n4iwb,1+n4iwb):
            nu_plus_w = nu+np.pi*(2.0*iw)/beta
            i_nuw = nu2inu(nu_plus_w, beta) #Here if nu+w is beyond our sigma we may want to implement a "tail" version of sigma
            if( not(sigma_dga is None) and (i_nuw >= -n4iwf and i_nuw < n4iwf)): 
                i_qk = k2ik(k+q,Nk_lin)
                Gres[iw+n4iwb,iq] = 1.0 / ( 1j*nu_plus_w + mu - eps_kq - sigma_dga[i_nuw+n4iwf,i_qk] )
            elif( (sigma_dga is None) or (i_nuw >= -niwf and i_nuw < niwf) ):
                Gres[iw+n4iwb,iq] = 1.0 / ( 1j*nu_plus_w + mu - eps_kq - sigma[i_nuw+niwf] )
    return Gres


@jit(nopython=True)
def G_w_given_nu(nu:np.float64, g_loc:np.ndarray, Nw:int, beta:np.float64)-> np.ndarray:
    Gres = np.zeros( (2*Nw+1), dtype=np.complex128 )
    Nnu = g_loc.shape[0]//2
    for iw in range(-Nw,Nw+1):
        nu_plus_w = nu+np.pi*(2.0*iw)/beta
        i_nuw = nu2inu(nu_plus_w, beta) #Here if nu+w is beyond our sigma we may want to implement a "tail" version of sigma
        if(i_nuw < -Nnu or i_nuw >= Nnu ): continue
        Gres[iw+Nw] = g_loc[i_nuw+Nnu]
    return Gres

# MU SEARCH
@jit(nopython=True)
def mu_root(mu:np.float64,n_target:np.float64, sigma_dga:np.ndarray, eps_kgrid:np.ndarray, beta:np.float64):
    Nk = eps_kgrid.shape[0]; n4iwf = sigma_dga.shape[0]//2
    nu_array=build_nu_mats(n4iwf,beta).reshape(2*n4iwf,1)
    print("testing mu=",mu)
    return n_target - (1.0/Nk/beta)*np.sum(  1.0/( 1j*nu_array +mu - eps_kgrid.reshape(1,Nk) -sigma_dga  ) - 1.0/(1j*nu_array ) ).real -0.5


def get_mu( mu_start:np.float64, n_target:np.float64, sigma_dga:np.ndarray, k_grid:np.ndarray, beta:np.float64 ):
    print("Searching for new chemical potential...")
    eps_kgrid=np.zeros(k_grid.shape[0])
    for ik,k in enumerate(k_grid):
        eps_kgrid[ik] = ek_2d(k, 1.0)
    root_sol = root(mu_root,args=(n_target,sigma_dga,eps_kgrid,beta),x0=mu_start,method="lm",tol=1e-10)
    mu_sol = root_sol.x[0]
    if(root_sol.success):
        print("After ",root_sol.nfev," function evaluations, the root is found to be ",mu_sol)
    else:
        print("Root finding did not converge. The best estimate is ",mu_sol)
    return mu_sol