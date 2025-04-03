import numpy as np
from scipy.optimize import root_scalar, root
from numba import jit

#PAULI PRINCIPLE
# sum_wq chi^L_up,up = sum_wq ( chi_d + chi^L_m) = (n/2-1)*(n/2)
@jit(nopython=True)
def root_pauli( lambda_m:np.float64, beta:np.float64, Nk:int, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, dens:np.float64 )->np.float64:
    return  (dens/2.0)*(1.0 - dens/2.0) - 0.5*np.abs(np.sum( np.sum(chi_d_latt,axis=1) +     1.0/np.sum(1.0/chi_m_latt + lambda_m,axis=1) ))/(beta*Nk)

def get_lambda_pauli( lambda_m:np.float64, beta:np.float64, Nk:int, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, dens:np.float64 ):
    root_sol = root(root_pauli,args=(beta,Nk,chi_d_latt,chi_m_latt,dens),x0=lambda_m,method="lm")
    lambda_sol = root_sol.x
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
