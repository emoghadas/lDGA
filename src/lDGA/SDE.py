import numpy as np
from utilities import ik2k, k2ik

#Without Hartree, should we insert it?
def Hubbard_SDE(U:np.float64, beta:np.float64, gamma:np.ndarray, Chi_r_w_q:np.ndarray, F_r_loc:np.array, Chi_0gen:np.array, Self_old:np.array, dens:np.float64, dim:int=2 ):
    #assert shapes
    Nnu,Nk = Self_old.shape
    Nw,Nq = Chi_r_w_q.shape[1:]
    Self_Energy = np.zeros( Self_old.shape, dtype=np.complex128)
    F_updn = F_r_loc[1] - F_r_loc[0] #order should be ["density","magnetic"]
    theta_nu_w_q = np.sum( \
                           np.reshape(F_updn,newshape(Nnu,Nnu,Nw,1)) * \
                           np.reshape( np.diag(Chi_0gen,axis1=0,axis2=1),newshape(1,Nnu,Nw,Nq)), \
                           axis=1)
    thetha_nu_w_q += 2 + gamma[0]*( 1.0+U*np.reshape(Chi_r_w_q[0],newshape=(1,Nw,Nq)) ) \
        +3.0*gamma[1]*(-1.0+U*np.reshape(Chi_r_w_q[1],newshape=(1,Nw,Nq)))

    G_aux = np.zeros( (Nw,Nq), dtype=np.complex128)
    for inu in range(-Nnu/2,Nnu/2):
        for ik in range(Nk):
            k = k2ik(ik,dim,Nk)
            Self_Energy[inu,ik] = -0.5*U*(1/beta**2)*np.sum( thehta_nu_w_q[inu,:,:] * G_wq_given_nuk(nu,k,Self_old,Nw,Nq,beta)  )
            

    #Hartree term
    Self_Energy[:,:] += 0.5*U*dens
    return something


def Holstein_SDE(g:np.float64, omega0:np.float64, beta:np.float64, gamma:np.ndarray, Chi_r_w_q:np.ndarray, F_r_loc:np.array, Chi_0gen:np.array, Self_old:np.array, dens:np.float64, dim:int=2):
    #assert shapes
    Nnu,Nk = Self_old.shape
    Nw,Nq = Chi_r_w_q.shape[1:]
    Self_Energy = np.zeros( Self_old.shape, dtype=np.complex128)
    F_d = -F_r_loc[0] #order should be ["density","magnetic"]
    wmats = 2*np.pi/beta*np.linspace(0,Nw,Nw,endpoint=False)
    Uw = np.reshape(Udyn_arr(omegas=wmats, omega0=omega0, g=g),newshape=(1,Nw,1))
    theta_nu_w_q = 2.0*np.sum( \
                           np.reshape(F_d,newshape(Nnu,Nnu,Nw,1)) * \
                           np.reshape( np.diag(Chi_0gen,axis1=0,axis2=1),newshape(1,Nnu,Nw,Nq)), \
                           axis=1)
    thetha_nu_w_q += 2.0 + gamma[0]*( 1.0+Uw*np.reshape(Chi_r_w_q[0],newshape=(1,Nw,Nq)) ) \
        +3.0*gamma[1]*(-1.0+Uw*np.reshape(Chi_r_w_q[1],newshape=(1,Nw,Nq)))

    G_aux = np.zeros( (Nw,Nq), dtype=np.complex128)
    for inu in range(-Nnu/2,Nnu/2):
        for ik in range(Nk):
            k = k2ik(ik,dim,Nk)
            Self_Energy[inu,ik] = -0.5*(1/beta**2)*np.sum( Uw[0,:,:]*thehta_nu_w_q[inu,:,:] * G_wq_given_nuk(nu,k,Self_old,Nw,Nq,beta)  )

    #Hartree term
    Self_Energy[:,:] += -2.0*(g**2/omega0)*dens
    return something

def Hubbard_Holstein_SDE(U:np.float64, g:np.float64, omega0:np.float64, beta:np.float64, gamma:np.ndarray, Chi_r_w_q:np.ndarray, F_r_loc:np.array, Chi_0gen:np.array, Self_old:np.array, dens:np.float64, dim:int=2):
    #assert shapes
    Nnu,Nk = Self_old.shape
    Nw,Nq = Chi_r_w_q.shape[1:]
    Self_Energy = np.zeros( Self_old.shape, dtype=np.complex128)
    F_d = -F_r_loc[0] #order should be ["density","magnetic"]
    wmats = 2*np.pi/beta*np.linspace(0,Nw,Nw,endpoint=False)
    Uw = np.reshape(Udyn_arr(omegas=wmats, omega0=omega0, g=g, U=U),newshape=(1,Nw,1))
    theta_nu_w_q = 2.0*np.sum( \
                           np.reshape(F_d,newshape(Nnu,Nnu,Nw,1)) * \
                           np.reshape( np.diag(Chi_0gen,axis1=0,axis2=1),newshape(1,Nnu,Nw,Nq)), \
                           axis=1)
    thetha_nu_w_q += 2.0 + gamma[0]*( 1.0+Uw*np.reshape(Chi_r_w_q[0],newshape=(1,Nw,Nq)) ) \
        +3.0*gamma[1]*(-1.0+Uw*np.reshape(Chi_r_w_q[1],newshape=(1,Nw,Nq)))

    G_aux = np.zeros( (Nw,Nq), dtype=np.complex128)
    for inu in range(-Nnu/2,Nnu/2):
        for ik in range(Nk):
            k = k2ik(ik,dim,Nk)
            Self_Energy[inu,ik] = -0.5*(1/beta**2)*np.sum( Uw[0,:,:]*thehta_nu_w_q[inu,:,:] * G_wq_given_nuk(nu,k,Self_old,Nw,Nq,beta)  )

    #Hartree term
    Self_Energy[:,:] += -2.0*(g**2/omega0)*dens

    return something
