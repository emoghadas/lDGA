import numpy as np
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
def inu2nu(inu:int, beta:np.float64) -> np.float64:
    return np.pi*(2.0*inu + 1.0)/beta

@jit(nopython=True)
def nu2inu(nu:np.float64, beta:np.float64) -> int:
    return int(np.round((nu*beta/np.pi - 1.0)/2.0))

@jit(nopython=True)
def square_ek(k:np.ndarray,t:np.float64) -> np.float64:
    return -2.0*t*np.sum( np.cos(k) )

#Maybe use multipledispatch for this and array?
@jit(nopython=True)
def Udyn(omega:np.float64, omega0:np.float64, g:np.float64, U:np.float64 = 0.0) -> np.float64:
    return U - omega0*g**2/(omega0**2 + omega**2)

@jit(nopython=True)
def Udyn_arr(omegas:np.ndarray , omega0:np.float64, g:np.float64, U:np.float64 = 0.0) -> np.ndarray:
    return U - omega0*g**2/(omega0**2 + omegas**2)

##### GREEN'S FUNCTION UTILITIES #####

#For the moment square (dim-dimensional hypercubic) dispersion only (using t=1)
#we may want to generalize it later using jitclass for a "model" object
#containing a method retrieving the dispersion

#N.B. this works only for DMFT self-energy with enough frequencies.
# Auxiliary function for Swinger-Dyson Equations
@jit(nopython=True)
def G_wq_given_nuk(nu:np.float64, k:np.ndarray, Sigma:np.ndarray, Nw:int, Nq:int, beta:np.float64 )-> np.complex128:
    dim = len(k); inu=nu2inu(nu, beta)
    Gres = np.zeros( (2*Nw+1,Nq), dtype=np.complex128 )
    Nnu = Sigma.shape[0]
    for iq in range(Nq):
        k_plus_q = k+ik2k(iq, dim, Nq)
        eps_kq = np.complex128(square_ek(k_plus_q, 1.0))
        for iw in range(-Nw/2,1+Nw/2):
            nu_plus_w = nu+np.pi*(2.0*iw)/beta
            i_nuw = nu2inu(nu_plus_w, beta) #Here if nu+w is beyond our sigma we may want to implement a "tail" version of sigma
            if(i_nuw < -Nnu/2 or i_nuw >= Nnu/2 ): continue
            Gres[iw,iq] += 1.0/(1j*nu_plus_w - eps_kq - Sigma[i_nuw+Nnu/2] )
    return Gres


