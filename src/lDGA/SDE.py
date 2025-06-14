import numpy as np
from lDGA.utilities import ik2k, k2ik, G_wq_given_nuk, Udyn_arr, G_w_given_nu, U_trans, Udyn, build_nu_mats, build_w_mats,Udyn_arr
from numba import jit
from mpi4py import MPI


# Lattice Swinger-Dyson for the Hubbard model
#TODO: CHECK FOR POSSIBLE BUGS AT LEAST ON SELF_SUM_U
def Hubbard_SDE(u:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d_w_q:np.ndarray, chi_m_w_q:np.ndarray, F_d_loc:np.array, F_m_loc:np.array, chi0_nu_w_q:np.ndarray, self_old:np.ndarray, dens:np.float64, qpoints:np.ndarray, Nk:int, mu:np.float64, dim:int=2, self_dga:np.ndarray=None):
    n4iwf=F_d_loc.shape[0]//2
    n4iwb,Nq = chi_d_w_q.shape
    n4iwb = n4iwb//2
    self_energy = np.zeros( (2*n4iwf,Nk), dtype=np.complex128)
    F_updn = F_d_loc - F_m_loc


    theta_nu_w_q = np.zeros( (2*n4iwf,2*n4iwb+1,Nq), dtype=np.complex128)
    theta_nu_w_q -= 2.0*np.einsum('ijk,jkm->ikm', F_updn, chi0_nu_w_q)
    theta_nu_w_q += 2.0 + gamma_d + u*np.einsum('ijk,jk->ijk',gamma_d, chi_d_w_q) \
                         -3.0*gamma_m + 3.0*u*np.einsum('ijk,jk->ijk',gamma_m, chi_m_w_q)
    self_energy = self_sum_U(self_old, theta_nu_w_q, u, beta, qpoints, Nk, dim, mu, self_dga)
    #Hartree term

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    #Hartree only on the master node
    if(mpi_rank==0):
        self_energy[:,:] += dens*u

    return self_energy

# Lattice Swinger-Dyson for the Hubbard-Holstein model
def Hubbard_Holstein_SDE(u:np.float64, g0:np.float64, omega0:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray, A_d:np.ndarray, A_m:np.ndarray, chi_d_w_q:np.ndarray, chi_m_w_q:np.ndarray, F_d_loc:np.array, F_m_loc:np.array, chi0_nu_w_q:np.ndarray, self_old:np.ndarray, g_old:np.ndarray, dens:np.float64, qpoints:np.ndarray, Nk:int, Nqtot:int, mu:np.float64, dim:int=2, self_dga:np.ndarray=None):

    #Here we also sum Fock term
    n4iwf = F_d_loc.shape[0]//2; n4iwb = chi_d_w_q.shape[0]//2
    Nq    = chi_d_w_q.shape[1]

    self_energy = np.zeros( (2*n4iwf,Nk), dtype=np.complex128)

    wmats = build_w_mats(n4iwb,beta)
    numats= build_nu_mats(n4iwf,beta)

    uw = Udyn_arr(omega=wmats, omega0=omega0, g=g0, u=u)
    ununup = U_trans(nu=numats,nup=numats, omega0=omega0, g=g0, u=u)
    u_d = np.reshape(2*uw-u,newshape=(2*n4iwb+1,1)) # will  be U_d(w,nu)
    u_m = np.reshape( -u ,newshape=(1,1) ) # will be U_m(w,nu)

    #USING OUR FORMULA
    theta_nu_wq = np.zeros( (2*n4iwf,2*n4iwb+1,Nq), dtype=np.complex128)

    theta_nu_wq += -4.0*ununup[0,0] +2.0*np.reshape(uw,newshape=(1,len(uw),1)) # U terms

    theta_nu_wq += -2*np.einsum('j,ijk->ijk',uw,gamma_d) + (A_d + 3*A_m)/beta # 34.1

    theta_nu_wq +=  np.einsum('ijk,j,jk,mjk,mjk->ijk',gamma_d, 2*uw, u_d*(1-u_d*chi_d_w_q), gamma_d, chi0_nu_w_q)/beta**2 # 34.2

    theta_nu_wq -=  np.einsum('ijk,jk,im,mjk,mjk->ijk',gamma_d,u_d*(1-u_d*chi_d_w_q),ununup,gamma_d,chi0_nu_w_q)/beta**2 # 34.3

    theta_nu_wq -= 3*np.einsum('ijk,jk,im,mjk,mjk->ijk',gamma_m,u_m*(1-u_m*chi_m_w_q),ununup,gamma_m,chi0_nu_w_q)/beta**2 # 34.4

    theta_nu_wq -= 2*np.einsum('j,imj,mjk->ijk',uw,F_d_loc,chi0_nu_w_q)/beta**2 #local part

    #SHOULD BE ZERO BUT IN HUBBARD HELPS CONVERGING
    #theta_nu_wq += 2*u*np.einsum('imj,mjk->ijk', F_d_loc+F_m_loc,chi0_nu_w_q)/(beta**2) # should be zero, subtracting the antiadiabatic part


    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    #Here also Fock term
    #N.B. now working only for a local self-energy, for SC-DGA to be corrected for a k-dependent one
    self_energy = self_sum_Uw(self_old, g_old, theta_nu_wq, omega0, g0, beta, qpoints, Nk, Nqtot, dim, mu, mpi_rank,self_dga)
    #Hartree term


    #Hartree only on the master node
    if(mpi_rank==0):
        self_energy += (dens*( u - (4.0*g0**2/omega0) ) + g0**2/omega0)

    return self_energy # Swinger-Dyson for the Hubbard-Holstein model


#internal auxiliary routine
@jit(nopython=True)
def self_sum_Uw(self_old:np.ndarray, g_old:np.ndarray, theta:np.ndarray,  omega0:np.float64, g0:np.float64, beta:np.float64, qpoints:np.ndarray,  Nk:int,  Nqtot:int, dim:int , mu:np.float64, mpi_rank:int, self_dga:np.ndarray=None,ts:np.ndarray=None) -> np.ndarray:
    n4iwf,n4iwb,Nqloc = theta.shape
    n4iwf//=2; n4iwb=n4iwb//2
    niwf = g_old.shape[0] //2

    self_en = np.zeros((2*n4iwf,Nk), dtype=np.complex128)

    for inu in range(-n4iwf,n4iwf):
        nu=(np.pi/beta)*(2*inu+1)
        for ik in range(Nk):
            k = ik2k(ik,dim,Nk)
            self_en[inu+n4iwf,ik] +=(0.5/beta)*np.sum( theta[inu+n4iwf,:,:] * G_wq_given_nuk(nu,k,self_old,n4iwb,qpoints,beta,mu,self_dga,ts=ts))/Nqtot #vertex term
            if( (g0!=0.0) and (not self_dga is None) ):
                self_en[inu+n4iwf,ik] -= np.sum(G_wq_given_nuk(nu,k,self_old,n4iwb,qpoints,beta,mu,self_dga,ts=ts)*Udyn_arr(build_w_mats(n4iwb,beta),omega0,g0).reshape(2*n4iwb+1,1))/beta/Nqtot
        if( (g0!=0.0 and mpi_rank==0) and ( self_dga is None) ):
            for inup in range(-niwf,niwf):
                nup=(np.pi/beta)*(2*inup+1)
                self_en[inu+n4iwf,:] -= g_old[inup+niwf]*Udyn(nu-nup,omega0,g0,u=0.0)*np.exp(1j*nup*1e-10)/beta
    return self_en

#internal auxiliary routine
@jit(nopython=True)
def self_sum_U(self_old:np.array, theta:np.ndarray, U:np.float64, beta:np.float64, qpoints:np.ndarray, Nk:int,  Nqtot:int, dim:int, mu:np.float64, self_dga:np.ndarray=None) -> np.ndarray:
    n4iwf,n4iwb,Nqloc = theta.shape
    n4i2f//=2; n2iwb//=2;
    self_en = np.zeros( (2*n4iwf,Nk), dtype=np.complex128)
    for inu in range(-n4iwf,n4iwf):
        nu=(np.pi/beta)*(2*inu+1)
        for ik in range(Nk):
            k = ik2k(ik,dim,Nk)
            self_en[inu,ik] = -(0.5/beta**2)*np.sum( (U*theta[inu,:,:]) * G_wq_given_nuk(nu,k,self_old,n4iwb,qpoints,beta,mu,self_dga,ts=ts)  ) #vertex term
    return self_en/Nqtot






###################################### ONLY FOR LOCAL TESTS PURPOSES ###########################################
#
# Local Swinger-Dyson for the Hubbard-Holstein model
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

    #USING OUR FORMULA
    theta_nu_w = np.zeros( (2*n4iwf,2*n4iwb+1), dtype=np.complex128)

    theta_nu_w += -4.0*ununup[0,0] + 2.0*np.reshape(uw,newshape=(1,len(uw))) # U terms

    theta_nu_w += -2*np.einsum('j,ij->ij', uw, gamma_d) + (A_d + 3*A_m)/beta # 34.1

    theta_nu_w +=  np.einsum('ij,j,mj,mj->ij', gamma_d, 2*uw*u_d*(1-u_d*chi_d_w), gamma_d, chi0_nu_w)/beta**2 # 34.2

    theta_nu_w -=  np.einsum('ij,j,im,mj,mj->ij', gamma_d, u_d*(1-u_d*chi_d_w), ununup, gamma_d, chi0_nu_w)/beta**2 # 34.3

    theta_nu_w -= 3*np.einsum('ij,j,im,mj,mj->ij', gamma_m, u_m*(1-u_m*chi_m_w), ununup, gamma_m, chi0_nu_w)/beta**2 # 34.4

    theta_nu_w -= 2*np.einsum('j,ikj,kj->ij', uw, F_d_loc, chi0_nu_w)/beta**2 #local part

    #TODO: i think this should be a minus sign, but with high enough statistics and large tails it should vanish
    theta_nu_w += 2*u*np.einsum('ijk,jk->ik', F_d_loc+F_m_loc, chi0_nu_w)/(beta**2) # should be zero, subtracting the antiadiabatic part

    #Here also Fock term
    self_energy = self_sum_Uw_loc(g_old, theta_nu_w, omega0,g0, beta)
    #Hartree term
    self_energy += dens*( u - (4.*g0**2/omega0) ) + g0**2/omega0 



    #USING DIRECTLY THE LOCAL SCHWINGER DYSON
    self_energy2  = np.zeros( (2*n4iwf), dtype=np.complex128)
    theta  = np.einsum('ijk,jk,k->ik', 2.0*F_d_loc ,chi0_nu_w,uw-u) #only ph
    theta  += np.einsum('ijk,jk->ik', (F_d_loc-F_m_loc) ,chi0_nu_w)*u #only hubb
    self_energy2 = self_sum_Uw_loc(g_old, theta, omega0, g0, beta)/(beta**2)
    self_energy2 += dens*( u - (4.*g0**2/omega0) ) + (g0**2/omega0) 

    return self_energy, self_energy2

# Local Swinger-Dyson for the Hubbard model
def Hubbard_SDE_loc(u:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d_w:np.ndarray, chi_m_w:np.ndarray, F_d_loc:np.ndarray, F_m_loc:np.ndarray, chi0_nu_w:np.ndarray, g_old:np.ndarray, dens:np.float64, mu:np.float64):
    n4iwf = F_d_loc.shape[0]//2
    n4iwb = chi_d_w.shape[0]//2

    self_energy = np.zeros( (2*n4iwf), dtype=np.complex128)
    theta_nu_w = np.zeros( (2*n4iwf,2*n4iwb+1), dtype=np.complex128)


    theta_nu_w += (2 + gamma_d  - 3*gamma_m)
    theta_nu_w += np.einsum('ijk,jk->ik', F_d_loc-F_m_loc,chi0_nu_w)/(beta**2)

    #ZERO
    theta_nu_w -= np.einsum('ijk,jk->ik', F_d_loc+F_m_loc,chi0_nu_w)/(beta**2) #should be zero

    theta_nu_w -= u*( np.einsum('ij,j->ij',gamma_d,chi_d_w) + 3*np.einsum('ij,j->ij',gamma_m,chi_m_w) )

    theta_nu_w *= -u

    #Here also Fock term
    self_energy = self_sum_Uw_loc(g_old, theta_nu_w, 0.0, 0.0, beta)


    #Hartree term
    self_energy += dens*u

    self_energy2  = np.zeros( (2*n4iwf), dtype=np.complex128)    
    theta  = np.einsum('ijk,jk->ik', (F_d_loc-F_m_loc ),chi0_nu_w)
    self_energy2 = self_sum_Uw_loc(g_old, theta, 0.0, 0.0, beta)*u/(beta**2)

    self_energy2 += dens*u 

    return self_energy, self_energy2

@jit(nopython=True)
def self_sum_Uw_loc(g_old:np.ndarray, theta:np.ndarray, omega0:np.float64, g0:np.float64, beta:np.float64) -> np.ndarray:
    n4iwf,n4iwb = theta.shape
    nfiw=g_old.shape[0]//2
    n4iwf//=2; n4iwb=(n4iwb-1)//2
    self_en = np.zeros((2*n4iwf), dtype=np.complex128)

    for inu in range(-n4iwf,n4iwf):
        nu=(np.pi/beta)*(2*inu+1)
        #Vertex term
        self_en[inu+n4iwf] +=  np.sum(theta[inu+n4iwf,:] * G_w_given_nu(nu,g_old,n4iwb,beta))*(0.5/beta)
        #Fock term
        if(g0!=0.0):
            for inup in range(-nfiw,nfiw):
                nup=(np.pi/beta)*(2*inup+1)
                self_en[inu+n4iwf] -= (g_old[inup+nfiw])*( Udyn(nu-nup,omega0,g0,u=0.0))/beta

    return self_en