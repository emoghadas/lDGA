import numpy as np
from scipy.optimize import root_scalar, root
from numba import jit

@jit(nopython=True)
def root_function_uniform(lambda_r:np.float64, beta:np.float64, Chi_w_q:np.ndarray, Chi_imp_w:np.ndarray) -> np.float64:
    # may want to assert dimensions of Chi_w_q and Chi_imp_w
    Nqs = Chi_w_q.shape[1:]; Dim = len(Nqs); dim2sum = tuple(range(1,Dim+1))
    return np.sum( Chi_imp_w - (Nq**Dim)/np.sum( 1/Chi_w_q + lambda_r,axis=dim2sum ) )/beta

@jit(nopython=True)
def root_function_wdep(lambda_r_w:np.ndarray, Chi_w_q:np.ndarray,Chi_imp_w:np.ndarray) -> np.ndarray:
    # may want to assert dimensions of Chi_w_q and Chi_imp_w
    Nqs = Chi_w_q.shape[1:]; Dim = len(Nqs); dim2sum = tuple(range(1,Dim+1))
    Chi_lambda_w = np.zeros(Chi_imp_w.shape)
    for i in range(Chi_imp_w.shape[0]):
        Chi_lambda_w[i] = (Nq**Dim)*np.sum(1/Chi_w_q[i] + lambda_r_w[i])
    return Chi_imp_w - Chi_lambda_w

def get_lambda_uniform(Chi_w_q:np.ndarray, Chi_imp_w:np.ndarray, beta:np.float64, lambda_0=1e-2) -> np.float64:
    print("Finding the uniform lambda_r")
    root_sol = root_scalar(root_function_uniform,args=(beta,Chi_w_q,Chi_imp_w),x0=lambda_0)
    if(root_sol.converged): 
        print("After ",root_sol.iterations," iterations, the root is found to be ",root_sol.root)
    else:
        print("Root finding did not converge. The best estimate is ",root_sol.root)
    return root_sol.root

def get_lambda_wdep(Chi_w_q:np.ndarray, Chi_imp_w:np.ndarray, beta:np.float64, lambda_0=1e-2) -> np.ndarray:
    print("Finding the w-dependent lambda_r")    
    root_sol = root(root_function_wdep,args=(beta,Chi_w_q,Chi_imp_w),x0=lambda_0*np.ones(len(Chi_imp_w)))
    if(root_sol.success): 
        print("After ",root_sol.nfev," function evaluations, the root is found to be ",root_sol.x)
    else:
        print("Root finding did not converge. The best estimate is ",root_sol.x)
    return root_sol.x
