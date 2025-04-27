import numpy as np
from scipy.optimize import root_scalar, root
from numba import jit
import lDGA.utilities as util

#PAULI PRINCIPLE
# sum_wq chi^L_up,up = sum_wq ( chi_d + chi^L_m) = (n/2-1)*(n/2)
@jit(nopython=True)
def root_pauli( lambda_m:np.float64, beta:np.float64, Nq:int, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, dens:np.float64, tail_flag:bool )->np.float64:
    n4iwb = chi_d_latt.shape[0]//2; tails = 2.0/util.build_w_mats(n4iwb,beta)**2; tails[n4iwb]=0; tail0=beta/12.0
    if( not tail_flag):
        tails[:]=0.0; tail0=0.0
    latt_sum = 0.5*( np.sum( np.sum(chi_d_latt,axis=1)/Nq + np.sum( chi_m_latt/(1.0+np.exp(lambda_m)*chi_m_latt),axis=1)/Nq -tails) ).real/beta +tail0
    return  (dens)*(1.0 - dens) - latt_sum

def get_lambda_pauli( lambda_m:np.float64, beta:np.float64, Nq:int, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, dens:np.float64, tail_flag:bool=False ):
    root_sol = root(root_pauli,args=(beta,Nq,chi_d_latt,chi_m_latt,dens,tail_flag),x0=np.log(lambda_m),method="lm",tol=1e-8)
    lambda_sol = np.exp(root_sol.x)
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
    return np.sum( np.abs(Chi_imp_w - (Nqs)/np.sum( 1/Chi_w_q + lambda_r,axis=1)) )/beta

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
def root_function_wdep(lambda_r_w:np.ndarray, Chi_w_q:np.ndarray,Chi_imp_w:np.ndarray) -> np.ndarray:
    # may want to assert dimensions of Chi_w_q and Chi_imp_w
    Nqs = Chi_w_q.shape[1]
    Chi_lambda_w = np.zeros(Chi_imp_w.shape,dtype=type(Chi_imp_w[0]))
    Nw = int(lambda_r_w.shape[0]/2)
    for i in range(Chi_imp_w.shape[0]):
        Chi_lambda_w[i] = (Nqs)*np.sum(1/Chi_w_q[i,:] + lambda_r_w[i]+1j*lambda_r_w[i+Nw])**-1
    return np.append( (Chi_imp_w - Chi_lambda_w).real, (Chi_imp_w - Chi_lambda_w).imag )

def get_lambda_wdep(Chi_w_q:np.ndarray, Chi_imp_w:np.ndarray, beta:np.float64, lambda_0:np.ndarray) -> np.ndarray:
    print("Finding the w-dependent lambda_r")    
    root_sol = root(root_function_wdep,args=(Chi_w_q,Chi_imp_w),x0=np.append(lambda_0.real,lambda_0.imag),method="lm")
    lambda_sol = root_sol.x[:len(lambda_0)] + 1j*root_sol.x[len(lambda_0):]
    if(root_sol.success): 
        print("After ",root_sol.nfev," function evaluations, the root is found to be ",lambda_sol)
    else:
        print("Root finding did not converge. The best estimate is ",lambda_sol)
    return lambda_sol
