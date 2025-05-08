import numpy as np
from scipy.optimize import root_scalar, root
from numba import jit
import lDGA.utilities as util


def lambda_correction(lambda_type:str, beta:np.float64, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray,chi_d_loc:np.ndarray,chi_m_loc:np.ndarray):
    
    match lambda_type:
        case "Pauli":
            lambda_m = get_lambda_m_pauli( beta, chi_d_latt, chi_m_latt,chi_d_loc, chi_m_loc )
            lambda_d = 0.0
        case "Uniform":
            lambda_m = get_lambda_uniform(beta, chi_m_latt, chi_m_loc)
            lambda_d = get_lambda_uniform(beta, chi_d_latt, chi_d_loc)
        case "Dynamical":
            lambda_m = get_lambda_wdep(beta, chi_m_latt, chi_m_loc)
            lambda_d = get_lambda_wdep(beta, chi_d_latt, chi_d_loc)
        case default:
            raise ValueError(f"Wrong lambda_type={lambda_type}")
    return lambda_d, lambda_m

#PAULI PRINCIPLE
# sum_wq chi^L_up,up = sum_wq ( chi_d + chi^L_m) = (n/2-1)*(n/2)
@jit(nopython=True)
def root_pauli( lambda_m:np.float64, beta:np.float64, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, chi_d_loc:np.ndarray, chi_m_loc:np.ndarray, lambda_maxpole:np.float64)->np.float64:
    n4iwb = chi_d_latt.shape[0]//2;  Nq = chi_m_latt.shape[1]
    latt_sum = 0.5*( np.sum( np.sum(chi_d_latt,axis=1)/Nq + np.sum( chi_m_latt/(1.0+(lambda_maxpole+np.exp(lambda_m))*chi_m_latt),axis=1)/Nq ) ).real/beta
    return 0.5*np.sum(chi_d_loc+chi_m_loc).real/beta  - latt_sum

def get_lambda_m_pauli( beta:np.float64, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, chi_d_loc:np.ndarray, chi_m_loc:np.ndarray):
    n4iwb = chi_m_latt.shape[0]//2
    lambda_maxpole = -np.min(1/chi_m_latt[n4iwb,:].real)
    root_sol = root(root_pauli,args=(beta,chi_d_latt,chi_m_latt,chi_d_loc,chi_m_loc,lambda_maxpole),x0=np.log(0.2),method="lm",tol=1e-8)
    lambda_sol = lambda_maxpole+np.exp(root_sol.x)
    if(root_sol.success):
        print("After ",root_sol.nfev," function evaluations, the root is found to be ",lambda_sol)
    else:
        print("Root finding did not converge. The best estimate is ",lambda_sol)
    return lambda_sol

#UNIFORM CHI_LOC = CHI_LATT
@jit(nopython=True)
def root_function_uniform(lambda_r:np.float64, beta:np.float64, chi_r_latt:np.ndarray, chi_r_loc:np.ndarray, lambda_maxpole:np.float) -> np.float64:
    # may want to assert dimensions of Chi_w_q and Chi_imp_w
    Nqs = chi_r_latt.shape[1]
    return np.sum( chi_r_loc - np.sum( chi_r_latt/(1.0+lambda_r*chi_r_latt),axis=1)/Nqs  ).real/beta

def get_lambda_uniform(beta:np.float64, chi_r_latt:np.ndarray, chi_r_loc:np.ndarray) -> np.float64:
    n4iwb = chi_r_latt.shape[0]//2
    lambda_maxpole = -np.min(1/chi_r_latt[n4iwb,:].real)
    print("Finding the uniform lambda_r")
    root_sol = root(root_function_uniform,args=(beta,chi_r_latt,chi_r_loc,lambda_maxpole),x0=np.log(0.2))
    lambda_sol = lambda_maxpole+np.exp(root_sol.x)
    if(root_sol.success): 
        print("After ",root_sol.nfev," iterations, the root is found to be ",lambda_sol)
    else:
        print("Root finding did not converge. The best estimate is ",lambda_sol)
    return lambda_sol

#W-DEPENDENT CHI_LOC = CHI_LATT
@jit(nopython=True)
def root_function_wdep(lambda_r_w:np.float64, chi_r_latt:np.ndarray,chi_r_loc:np.float64,lambda_w_mp:np.float64) -> np.float64:
    # may want to assert dimensions of Chi_w_q and Chi_imp_w
    Nqs = chi_r_latt.shape
    Chi_lambda_w = np.sum( chi_r_latt/(1.0+(lambda_w_mp+np.exp(lambda_r_w))*chi_r_latt))/np.float64(Nqs)
    return (chi_r_loc - Chi_lambda_w)

def get_lambda_wdep(chi_r_latt:np.ndarray, chi_r_loc:np.ndarray, beta:np.float64) -> np.ndarray:
    print("Finding the w-dependent lambda_r")
    n4iwb = chi_r_latt.shape[0]//2
    lambda_w_mp = -np.min(1.0/chi_r_latt[n4iwb,:].real)
    lambda_sol = np.zeros(n4iwb*2+1)
    for iw in range(n4iwb*2+1):
        root_sol = root(root_function_wdep,args=(chi_r_latt[iw,:].real,chi_r_loc[iw].real,lambda_w_mp),x0=np.log(0.1),method="lm")
        lambda_sol[iw] = lambda_w_mp+np.exp(root_sol.x)
    if(root_sol.success):
        print("After ",root_sol.nfev," function evaluations, the root is found to be ",lambda_sol)
    else:
        print("Root finding did not converge. The best estimate is ",lambda_sol)
    return lambda_sol, lambda_w_mp
