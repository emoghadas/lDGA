import numpy as np
from lDGA.utilities import ik2k, k2ik, G_wq_given_nuk, Udyn_arr, G_w_given_nu, U_trans, Udyn, build_nu_mats, build_w_mats
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
def Hubbard_Holstein_SDE(U:np.float64, g:np.float64, omega0:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray, A_d:np.ndarray, A_m:np.ndarray, chi_d_w_q:np.ndarray, chi_m_w_q:np.ndarray, F_d_loc:np.array, F_m_loc:np.array, chi0_nu_w_q:np.ndarray, self_old:np.ndarray, dens:np.float64, Nk:int, mu:np.float64, dim:int=2):
    Nnu = self_old.shape[0]
    Nw,Nq = chi_d_w_q.shape
    Nw=(Nw+1)//2; Nnu//=2
    self_energy = np.zeros( (2*Nnu,Nk), dtype=np.complex128)
    wmats = 2*np.pi/beta*np.linspace(-Nw,Nw,2*Nw-1,endpoint=True)
    numats= np.pi/beta*(np.linspace(-Nnu,Nnu,endpoint=False)+0.5)
    Uw = Udyn_arr(omegas=wmats, omega0=omega0, g=g, U=0.0)
    Ununup = 2*(U_trans(nu=numats,nup=numats, omega0=omega0, g=g, U=U))
    #Uw = Udyn_arr(omegas=wmats, omega0=omega0, g=g, U=0.0)

    theta_nu_w_q = np.zeros( (2*Nnu,2*Nw-1,Nq), dtype=np.complex128)

    theta_nu_w_q += 4.0*np.reshape(Ununup,newshape=(1,len(numats),len(numats)))-2.0*np.reshape(Uw,newshape=(1,len(Uw),1))

    theta_nu_w_q+= np.einsum('j,ijk->ijk',Uw,gamma_d) - A_d/chi0_nu_w_q #1.d
    theta_nu_w_q-= A_m/chi0_nu_w_q #1.m
    
    theta_nu_w_q+=np.einsum('j,ijk,mjk,mjk->ijk',2*Uw*(2*Uw-U),1-np.einsum('j,ijk->ijk',2*Uw-U,chi_d_w_q) ,gamma_d,chi0_nu_w_q) #2.d
    theta_nu_w_q-=np.einsum('j,ijk,im,mjk,mjk->ijk',(2*Uw-U),1-np.einsum('j,ijk->ijk',2*Uw-U,chi_d_w_q) ,Ununup,gamma_d,chi0_nu_w_q)

    theta_nu_w_q+=np.einsum('ijk,ijk,im,mjk,mjk',gamma_m,U*(1-U*chi_m_w_q),Ununup,gamma_m,chi0_nu_w_q) #2.m

    integral = 2.0*np.einsum('ijk,jk->ik', 0.5*(F_d_loc+F_m_loc), chi0_nu_w_q/beta)

   #Here we also sum Fock term
    self_energy = self_sum_Uw(self_old, theta_nu_w_q, Uw, U, beta, Nk, dim, mu)
    #Hartree term
    self_energy[:,:] += dens*( U-2.0*(g**2/omega0) )

    return self_energy# Swinger-Dyson for the Hubbard-Holstein model


#internal auxiliary routine
#@jit(nopython=True)
def self_sum_Uw(self_old:np.ndarray, g_old:np.ndarray, theta:np.ndarray,  omega0:np.float64, g:np.float64, beta:np.float64, Nk:int, dim:int , mu:np.float64) -> np.ndarray:
    n4iwf,n4iwb,Nqdim = theta.shape
    n4iwf//=2; n4iwb=(n4iwb+1)//2; Nq=int(Nqdim**(1/dim))
    nfiw = g_old.shape[0] //2
    self_en = np.zeros((2*n4iwf,Nk), dtype=np.complex128)

    for inu in range(-n4iwf,n4iwf):
        nu=(np.pi/beta)*(2*inu+1)
        for ik in range(Nk):
            k = ik2k(ik,dim,Nk)
            self_en[inu,ik] -=(0.5/beta**2)*np.sum( theta[inu,:,:] * G_wq_given_nuk(nu,k,self_old,n4iwf,Nq,beta,mu)) #vertex term

        for inup in range(-nfiw,nfiw):
            nup=(np.pi/beta)*(2*inup+1)
            self_en[inu+n4iwf] -= g_old[inup+nfiw]*Udyn(nu-nup,omega0,g,U=0.0)/beta
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


# LOCAL TEST

def Hubbard_Holstein_SDE_loc(u:np.float64, g0:np.float64, omega0:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray,  A_d:np.ndarray, A_m:np.ndarray, chi_d_w:np.ndarray, chi_m_w:np.ndarray, F_d_loc:np.ndarray, F_m_loc:np.ndarray, chi0_nu_w:np.ndarray, g_old:np.ndarray, dens:np.float64, mu:np.float64):
    n4iwf = F_d_loc.shape[0]//2
    n4iwb = chi_d_w.shape[0]//2
    self_energy = np.zeros( (2*n4iwf), dtype=np.complex128)

    wmats = build_w_mats(n4iwb,beta)
    numats= build_nu_mats(n4iwf,beta)

    uw = Udyn_arr(omega=wmats, omega0=omega0, g=g0, u=u)
    ununup = U_trans(nu=numats,nup=numats, omega0=omega0, g=g0, u=u)
    u_d = 2*uw-u
    u_m = -u

    theta_nu_w = np.zeros( (2*n4iwf,2*n4iwb+1), dtype=np.complex128)

    theta_nu_w += -4.0*ununup[0,0]+2.0*np.reshape(uw,newshape=(1,len(uw))) # U terms

    theta_nu_w += -2*np.einsum('j,ij->ij',uw,gamma_d) + (A_d + 3*A_m)/beta # 34.1
    
    theta_nu_w +=  np.einsum('ij,j,mj,mj->ij',gamma_d,2*uw*u_d*(1-u_d*chi_d_w) ,gamma_d,chi0_nu_w)/beta**2 # 34.2

    theta_nu_w -=  np.einsum('ij,j,im,mj,mj->ij',gamma_d,u_d*(1-u_d*chi_d_w),ununup,gamma_d,chi0_nu_w)/beta**2 # 34.3

    theta_nu_w -= 3*np.einsum('ij,j,im,mj,mj->ij',gamma_m,u_m*(1-u_m*chi_m_w),ununup,gamma_m,chi0_nu_w)/beta**2 # 34.4

    theta_nu_w -= 2*np.einsum('j,ikj,kj->ij',uw,F_d_loc,chi0_nu_w)/beta**2 #local part


    theta_nu_w += 2*ununup[0,0]*np.einsum('ijk,jk->ik', F_d_loc+F_m_loc,chi0_nu_w)/(beta**2) # should be zero, subtracting the antiadiabatic part

    
    #Here also Fock term
    self_energy = self_sum_Uw_loc(g_old, theta_nu_w, omega0,g0, beta)

    #Hartree term
    self_energy += dens*( u-4.0*(g0**2/omega0) )

    return self_energy

def Hubbard_SDE_loc(u:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d_w:np.ndarray, chi_m_w:np.ndarray, F_d_loc:np.ndarray, F_m_loc:np.ndarray, chi0_nu_w:np.ndarray, g_old:np.ndarray, dens:np.float64, mu:np.float64):
    n4iwf = F_d_loc.shape[0]//2
    n4iwb = chi_d_w.shape[0]//2

    self_energy = np.zeros( (2*n4iwf), dtype=np.complex128)
    theta_nu_w = np.zeros( (2*n4iwf,2*n4iwb+1), dtype=np.complex128)


    theta_nu_w += (2 + gamma_d  - 3*gamma_m)
    theta_nu_w += np.einsum('ijk,jk->ik', F_d_loc-F_m_loc,chi0_nu_w)/(beta**2)

    theta_nu_w -= np.einsum('ijk,jk->ik', F_d_loc+F_m_loc,chi0_nu_w)/(beta**2) #should be zero

    theta_nu_w -= u*( np.einsum('ij,j->ij',gamma_d,chi_d_w) + 3*np.einsum('ij,j->ij',gamma_m,chi_m_w) )

    theta_nu_w *= -u

    #Here also Fock term
    self_energy = self_sum_Uw_loc(g_old, theta_nu_w, 0.0, 0.0, beta)


    #Hartree term
    self_energy += dens*u

    self_energy2  = np.zeros( (2*n4iwf), dtype=np.complex128)    
    theta  = np.einsum('ijk,jk->ik', (F_d_loc-F_m_loc ),chi0_nu_w)
    #theta  = np.einsum('ijk,jk->ik', -2*F_m_loc ,chi0_nu_w)
    self_energy2 = self_sum_Uw_loc(g_old, theta, 0.0, 0.0, beta)*u/(beta**2)

    self_energy2 += dens*u

    return self_energy            ,self_energy2

@jit(nopython=True)
def self_sum_Uw_loc(g_old:np.ndarray, theta:np.ndarray, omega0:np.float64, g:np.float64, beta:np.float64) -> np.ndarray:
    n4iwf,n4iwb = theta.shape
    nfiw=g_old.shape[0]//2
    n4iwf//=2; n4iwb=(n4iwb-1)//2
    self_en = np.zeros((2*n4iwf), dtype=np.complex128)
    
    for inu in range(-n4iwf,n4iwf):
        nu=(np.pi/beta)*(2*inu+1)
        #Vertex term
        self_en[inu+n4iwf] +=  np.sum(theta[inu+n4iwf,:] * G_w_given_nu(nu,g_old,n4iwb,beta))*(0.5/beta)
        #Fock term
        if(g!=0.0):
            for inup in range(-nfiw,nfiw):
                nup=(np.pi/beta)*(2*inup+1)
                self_en[inu+n4iwf] -= g_old[inup+nfiw]*Udyn(nu-nup,omega0,g,u=0.0)/beta

    return self_en
