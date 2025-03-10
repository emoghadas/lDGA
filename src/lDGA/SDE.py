import numpy as np
from lDGA.utilities import ik2k, k2ik, G_wq_given_nuk, Udyn_arr, G_w_given_nu
from numba import jit

#Without Hartree, should we insert it?
def Hubbard_SDE(U:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d_w_q:np.ndarray, chi_m_w_q:np.ndarray, F_d_loc:np.array, F_m_loc:np.array, chi0_nu_w_q:np.ndarray, self_old:np.ndarray, dens:np.float64, Nk:int, mu:np.float64, dim:int=2):
    #assert shapes
    Nnu = self_old.shape[0]
    Nw,Nq = chi_d_w_q.shape
    Nw=(Nw+1)//2; Nnu//=2;
    self_energy = np.zeros( (2*Nnu,Nk), dtype=np.complex128)
    F_updn = F_d_loc - F_m_loc

    theta_nu_w_q = np.zeros( (2*Nnu,2*Nw-1,Nq), dtype=np.complex128)
    theta_nu_w_q = -2.0*np.einsum('ijk,jkm->ikm', F_updn, chi0_nu_w_q)
    theta_nu_w_q += 2.0 + gamma_d + U*np.einsum('ijk,jk->ijk',gamma_d, chi_d_w_q) \
                         -3.0*gamma_m + 3.0*U*np.einsum('ijk,jk->ijk',gamma_m, chi_m_w_q)
    self_energy = self_sum_U(self_old, theta_nu_w_q, U, beta, Nk, dim, mu)
    #Hartree term
    self_energy[:,:] += dens*U

    return self_energy

# Swinger-Dyson for the Hubbard-Holstein model
def Hubbard_Holstein_SDE(U:np.float64, g:np.float64, omega0:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d_w_q:np.ndarray, chi_m_w_q:np.ndarray, F_d_loc:np.array, F_m_loc:np.array, chi0_nu_w_q:np.ndarray, self_old:np.ndarray, dens:np.float64, Nk:int, mu:np.float64, dim:int=2):
    Nnu = self_old.shape[0]
    Nw,Nq = chi_d_w_q.shape
    Nw=(Nw+1)//2; Nnu//=2
    self_energy = np.zeros( (2*Nnu,Nk), dtype=np.complex128)
    wmats = 2*np.pi/beta*np.linspace(-Nw,Nw,2*Nw-1,endpoint=True)
    Uw = np.reshape(Udyn_arr(omegas=wmats, omega0=omega0, g=g, U=0.0),newshape=(1,2*Nw-1,1))
    #Uw = Udyn_arr(omegas=wmats, omega0=omega0, g=g, U=0.0)

    theta_nu_w_q = np.zeros( (2*Nnu,2*Nw-1,Nq), dtype=np.complex128)
    theta_nu_w_q = 2.0*np.einsum('ijk,jkm->ikm', F_d_loc, chi0_nu_w_q)
    theta_nu_w_q += 2.0 + gamma_d + np.einsum('ijk,ijk,jk',gamma_d, Uw, chi_d_w_q) \
                         -3.0*gamma_m +3.0*np.einsum('ijk,ijk,jk',gamma_m, Uw, chi_m_w_q)
   #thetha_nu_w_q += 2.0 + gamma_d*( 1.0+(Uw+U)*np.reshape(chi_d_w_q,newshape=(1,Nw,Nq)) ) \
#                      +3.0*gamma_m*(-1.0+(Uw+U)*np.reshape(chi_m_w_q[1],newshape=(1,Nw,Nq)))

   #Here we also sum Fock term
    self_energy = self_sum_Uw(self_old, theta_nu_w_q, Uw, U, beta, Nk, dim, mu)
    #Hartree term
    self_energy[:,:] += dens*( U-2.0*(g**2/omega0) )

    return self_energy# Swinger-Dyson for the Hubbard-Holstein model

def Hubbard_Holstein_SDE_loc(U:np.float64, g:np.float64, omega0:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d_w:np.ndarray, chi_m_w:np.ndarray, F_d_loc:np.ndarray, F_m_loc:np.ndarray, chi0_nu_w:np.ndarray, self_old:np.ndarray, dens:np.float64, mu:np.float64):
    Nnu = F_d_loc.shape[0]//2
    Nw = chi_d_w.shape[0]//2+1
    #Nw=(Nw+1)//2; Nnu//=2
    self_energy = np.zeros( (2*Nnu), dtype=np.complex128)
    wmats = 2*np.pi/beta*np.linspace(-Nw+1,Nw-1,2*Nw-1,endpoint=True)
    Uw = np.reshape(Udyn_arr(omegas=wmats, omega0=omega0, g=g, U=0.0), shape=(1,2*Nw-1))
    #Uw = Udyn_arr(omegas=wmats, omega0=omega0, g=g, U=0.0)

    theta_nu_w = np.zeros( (2*Nnu,2*Nw-1), dtype=np.complex128)
    theta_nu_w -= 2.0*np.einsum('ijk,jk->ik', 0.5*(F_d_loc-F_m_loc), chi0_nu_w/beta)
    theta_nu_w -= 2.0 + gamma_d + np.einsum('ij,ij,j->ij',gamma_d, (Uw+U), chi_d_w) \
                         -3.0*gamma_m + 3.0*np.einsum('ij,ij,j->ij',gamma_m, (Uw+U), chi_m_w)

    integral = 2.0*np.einsum('ijk,jk->ik', 0.5*(F_d_loc+F_m_loc), chi0_nu_w/beta)
    self_energy_2 = self_sum_Uw_loc(self_old, integral, Uw, U, beta, mu)

    print(np.sum(self_energy_2))

    #Here we also sum Fock term
    self_energy = self_sum_Uw_loc(self_old, theta_nu_w, Uw, U, beta,  mu)

    #Hartree term
    self_energy += dens*( U-2.0*(g**2/omega0) )

    return self_energy

#internal auxiliary routine
#@jit(nopython=True)
def self_sum_Uw(self_old:np.array, theta:np.ndarray, Uw:np.ndarray, U:np.float64, beta:np.float64, Nk:int, dim:int , mu:np.float64) -> np.ndarray:
    Nnu,Nw,Nqdim = theta.shape
    Nnu//=2; Nw=(Nw+1)//2; Nq=int(Nqdim**(1/dim))
    self_en = np.zeros((2*Nnu,Nk), dtype=np.complex128)
    for inu in range(-Nnu,Nnu):
        nu=(np.pi/beta)*(2*inu+1)
        for ik in range(Nk):
            k = ik2k(ik,dim,Nk)
            self_en[inu,ik] = -(0.5/beta**2)*np.sum( ( (Uw[0,Nw,0]-Uw[0,:,:]) + (Uw[0,:,:]+U)*theta[inu,:,:])  * G_wq_given_nuk(nu,k,self_old,Nw,Nq,beta,mu)) #vertex term
    return self_en/Nqdim

#internal auxiliary routine
@jit(nopython=True)
def self_sum_U(self_old:np.array, theta:np.ndarray, U:np.float64, beta:np.float64, Nk:int, dim:int, mu:np.float64 ) -> np.ndarray:
    Nnu,Nw,Nqdim = theta.shape
    Nnu//=2; Nw=(Nw+1)//2; Nq=int(Nqdim**(1/dim))
    self_en = np.zeros( (2*Nnu,Nk), dtype=np.complex128)
    for inu in range(-Nnu,Nnu):
        nu=(np.pi/beta)*(2*inu+1)
        for ik in range(Nk):
            k = ik2k(ik,dim,Nk)
            self_en[inu,ik] = -(0.5/beta**2)*np.sum( (U*theta[inu,:,:]) * G_wq_given_nuk(nu,k,self_old,Nw,Nq,beta,mu)  ) #vertex term
    return self_en/Nqdim

@jit(nopython=True)
def self_sum_Uw_loc(g_old:np.array, theta:np.ndarray, Uw:np.ndarray, U:np.float64, beta:np.float64, mu:np.float64) -> np.ndarray:
    Nnu,Nw = theta.shape
    Nnu//=2; Nw=(Nw+1)//2
    self_en = np.zeros((2*Nnu), dtype=np.complex128)
    for inu in range(-Nnu,Nnu):
        nu=(np.pi/beta)*(2*inu+1)
        self_en[inu+Nnu] = -(0.5/beta**2)*np.sum( ( (Uw[0,Nw]-Uw[0,:]) + (Uw[0,:]+U)*theta[inu+Nnu,:]) * G_w_given_nu(nu,g_old,Nw,beta,mu)) #vertex term
    return self_en