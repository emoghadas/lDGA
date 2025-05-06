import numpy as np
from scipy.optimize import root_scalar, root
from numba import jit
import lDGA.utilities as util

@jit(nopython=True)
def root_pauli_2( lambda_m:np.float64, beta:np.float64, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, dens:np.float64,  tail_flag:bool , lambda_maxpole:np.float64)->np.float64:
    n4iwb = chi_d_latt.shape[0]//2; tails = 2.0/util.build_w_mats(n4iwb,beta)**2; tails[n4iwb]=0; tail0=beta/12.0
    Nq = chi_m_latt.shape[1]
    if( not tail_flag):
        tails[:]=0.0; tail0=0.0
    latt_sum = 0.5*( np.sum( np.sum(chi_d_latt,axis=1)/Nq + np.sum( chi_m_latt/(1.0+(lambda_maxpole+np.exp(lambda_m))*chi_m_latt),axis=1)/Nq -tails) ).real/beta +tail0
    #latt_sum = 0.5*( np.sum( np.sum(chi_d_latt,axis=1)/Nq + np.sum( chi_m_latt/(1.0+((lambda_m))*chi_m_latt),axis=1)/Nq -tails) ).real/beta +tail0
    return dens*(1-dens)  - latt_sum

#PAULI PRINCIPLE
# sum_wq chi^L_up,up = sum_wq ( chi_d + chi^L_m) = (n/2-1)*(n/2)
@jit(nopython=True)
def root_pauli( lambda_m:np.float64, beta:np.float64, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, chi_d_loc:np.ndarray, chi_m_loc:np.ndarray,  tail_flag:bool , lambda_maxpole:np.float64)->np.float64:
    n4iwb = chi_d_latt.shape[0]//2; tails = 2.0/util.build_w_mats(n4iwb,beta)**2; tails[n4iwb]=0; tail0=beta/12.0
    Nq = chi_m_latt.shape[1]
    if( not tail_flag):
        tails[:]=0.0; tail0=0.0
    latt_sum = 0.5*( np.sum( np.sum(chi_d_latt,axis=1)/Nq + np.sum( chi_m_latt/(1.0+(lambda_maxpole+np.exp(lambda_m))*chi_m_latt),axis=1)/Nq -tails) ).real/beta +tail0
    #latt_sum = 0.5*( np.sum( np.sum(chi_d_latt,axis=1)/Nq + np.sum( chi_m_latt/(1.0+((lambda_m))*chi_m_latt),axis=1)/Nq -tails) ).real/beta +tail0
    return 0.5*np.sum(chi_d_loc+chi_m_loc).real/beta  - latt_sum

def get_lambda_pauli( beta:np.float64, Nq:int, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, dens:np.float64, tail_flag:bool=False ):
    n4iwb = chi_m_latt.shape[0]//2
    lambda_maxpole = -np.min(1/chi_m_latt[n4iwb,:].real)
    print("Lambda_MaxPole:",lambda_maxpole)
    root_sol = root(root_pauli,args=(beta,Nq,chi_d_latt,chi_m_latt,dens,tail_flag,lambda_maxpole),x0=np.log(0.2),method="lm",tol=1e-8)
    lambda_sol = lambda_maxpole+np.exp(root_sol.x)
    if(root_sol.success):
        print("After ",root_sol.nfev," function evaluations, the root is found to be ",lambda_sol)
    else:
        print("Root finding did not converge. The best estimate is ",lambda_sol)
    return lambda_sol

#UNIFORM CHI_LOC = CHI_LATT
@jit(nopython=True)
def root_function_uniform(lambda_r:np.float64, beta:np.float64, Chi_w_q:np.ndarray, Chi_imp_w:np.ndarray) -> np.float64:
    # may want to assert dimensions of Chi_w_q and Chi_imp_w
    Nqs = Chi_w_q.shape[1]
    return np.sum( Chi_imp_w - np.sum( Chi_w_q/(1.0+lambda_r*Chi_w_q),axis=1)/Nqs  ).real/beta

def get_lambda_uniform(Chi_w_q:np.ndarray, Chi_imp_w:np.ndarray, beta:np.float64, lambda_0=1e-2) -> np.float64:
    print("Finding the uniform lambda_r")
    root_sol = root_scalar(root_function_uniform,args=(beta,Chi_w_q,Chi_imp_w),x0=lambda_0)
    if(root_sol.converged): 
        print("After ",root_sol.iterations," iterations, the root is found to be ",root_sol.root)
    else:
        print("Root finding did not converge. The best estimate is ",root_sol.root)
    return root_sol.root

#W-DEPENDENT CHI_LOC = CHI_LATT
@jit(nopython=True)
def root_function_wdep(lambda_r_w:np.float64, Chi_w_q:np.ndarray,Chi_imp_w:np.float64,lambda_w_mp:np.float64) -> np.float64:
    # may want to assert dimensions of Chi_w_q and Chi_imp_w
    Nqs = Chi_w_q.shape
    Chi_lambda_w = np.sum( Chi_w_q/(1.0+(lambda_w_mp+np.exp(lambda_r_w))*Chi_w_q))/np.float64(Nqs)
    return (Chi_imp_w - Chi_lambda_w)

def get_lambda_wdep(Chi_w_q:np.ndarray, Chi_imp_w:np.ndarray, beta:np.float64) -> np.ndarray:
    print("Finding the w-dependent lambda_r")
    n4iwb = Chi_w_q.shape[0]//2
    lambda_w_mp = -np.min(1.0/Chi_w_q[n4iwb,:].real)
    lambda_sol = np.zeros(n4iwb*2+1)
    for iw in range(n4iwb*2+1):
        root_sol = root(root_function_wdep,args=(Chi_w_q[iw,:].real,Chi_imp_w[iw].real,lambda_w_mp),x0=np.log(0.1),method="lm")
        lambda_sol[iw] = lambda_w_mp+np.exp(root_sol.x)
    if(root_sol.success):
        print("After ",root_sol.nfev," function evaluations, the root is found to be ",lambda_sol)
    else:
        print("Root finding did not converge. The best estimate is ",lambda_sol)
    return lambda_sol, lambda_w_mp
