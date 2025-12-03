import numpy as np
from scipy.optimize import root_scalar, root
from numba import jit
from mpi4py import MPI
import lDGA.utilities as util
import lDGA.bse as bse
import lDGA.SDE as sde
from lDGA.config import DGA_ConfigType
from typing import Tuple, Union

@jit(nopython=True)
def get_q_slice_indices(cfg):
    s = cfg.q_range
    return s.start, s.stop

def lambda_correction_epot(dga_cfg, chi_d_latt, chi_m_latt, v_d_w_q, v_m_w_q, A_d,A_m, chi_d_w_q, chi_m_w_q, chi0_w_q, mu, G_nu_k, s_nuk_loc, comm):
    lambda_type = dga_cfg.lambda_type
    u = dga_cfg.U
    omega0=dga_cfg.w0
    g0=dga_cfg.g0
    lam = 2*g0**2/omega0
    n=dga_cfg.occ_imp
    beta = dga_cfg.beta
    chi_d_loc = dga_cfg.chi_d_loc    
    chi_m_loc = dga_cfg.chi_m_loc
    weights = dga_cfg.weights
    n4iwb = dga_cfg.n4iwb
    n4iwf = dga_cfg.n4iwf
    niwf = dga_cfg.niwf
    ntail = s_nuk_loc.shape[0]//2
    Nq = dga_cfg.n_qpoints_fullbz
    n_qpoints = dga_cfg.n_qpoints
    s, e = get_q_slice_indices(dga_cfg)
    q_range = slice(s, e)
    wmats = util.build_w_mats(n4iwb,beta)
    uw = util.Udyn_arr(omega=wmats, omega0=omega0, g=g0, u=0.)

    g_imp = dga_cfg.g_imp
    s_imp = dga_cfg.s_imp

    ht = n*( u  - (4.*g0**2/omega0) )

    rank = comm.Get_rank()

    #if rank==0:
    #    epot_1p = np.sum(G_nu_k*s_nuk_loc)/(beta*Nq) + 0.5*ht - 1.2*0.075
    #
    #    epot_2p = 0.5*u*np.sum(chi_d_loc - chi_m_loc)/beta + u*n**2
    #    epot_2p += np.sum((chi_d_loc)*uw)/beta + 2*(-2*g0**2/omega0)*n**2
    #    #epot_2p -= g0/omega0 * 2*g0**2*np.sum((chi_d_loc)*(-wmats**2 + omega0**2)/(wmats**2 + omega0**2)**2)/beta
    #    #epot -= ((chi_d_loc - np.sum(chi0_loc_w, axis=0)/beta)*uw)[n4iwb]/beta
    #    #epot += (0.5*(chi_d_loc - chi_m_loc)*uw + uw*n**2)[n4iwb]/beta
    #    print(f"DMFT Epot residual: {(epot_2p)} ")

    # initial guess only on root
    if rank==0:
        lambda_maxpole_d = -np.min(1/chi_d_latt[n4iwb,:].real)
        lambda_maxpole_m = -np.min(1/chi_m_latt[n4iwb,:].real)
    else:
        lambda_maxpole_d = lambda_maxpole_m = None

    # bcast initial guess to all ranks
    lambda_maxpole_d, lambda_maxpole_m = comm.bcast((lambda_maxpole_d, lambda_maxpole_m), root=0)
    
    # write solution for lam_m here
    lam_m_sol = None

    if rank==0:
        lam_ds = np.linspace(lambda_maxpole_d+0.5, 20, 50)
        lam_ms = []
        for lam_di in lam_ds:
            lam_d = np.log(lam_di - lambda_maxpole_d)
            root_sol = root(root_pauli_epot, args=(beta, chi_d_latt, chi_m_latt, chi_d_loc, chi_m_loc, lam_d, lambda_maxpole_d, lambda_maxpole_m, weights), x0=np.log(0.2), method="lm", tol=1e-8)
            lambda_sol_m = lambda_maxpole_m + np.exp(root_sol.x)
            lam_ms.append(lambda_sol_m)

        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(lam_ds, lam_ms, ".-")
        plt.savefig("lam_d_v_lam_m.pdf", bbox_inches="tight")

        lam_ms = np.array(lam_ms)
    else:
        lam_ds = None
        lam_ms = None
    
    lam_ds, lam_ms = comm.bcast((lam_ds, lam_ms), root=0)

    epot_res = []
    for lam_di, lam_mi in zip(lam_ds, lam_ms):
        # correct susceptibilities on each process
        chi_d_w_q_lam = chi_d_w_q / (1 + lam_di*chi_d_w_q)
        chi_m_w_q_lam = chi_m_w_q / (1 + lam_mi*chi_m_w_q)

        # sde for selfenergy
        sigma_dga_q = sde.Hubbard_Holstein_SDE(dga_cfg, v_d_w_q, v_m_w_q, A_d, A_m, chi_d_w_q_lam, chi_m_w_q_lam, chi0_w_q, mu, G_nu_k, s_nuk_loc)

        sigma_dga = np.zeros_like(sigma_dga_q, dtype=np.complex128) if rank==0 else None
        comm.Reduce(sigma_dga_q, sigma_dga, op=MPI.SUM, root=0)
        
        s_nuk_dga = np.copy(s_nuk_loc)
        s_nuk_dga[ntail-n4iwf:ntail+n4iwf,:] = sigma_dga

        new_mu=0.0
        if(rank==0):
            new_mu = util.get_mu(dga_cfg, s_nuk_dga, verbose=False)
        new_mu = comm.bcast(new_mu, root=0)

        G_nu_k_dga = bse.G_nu_k(dga_cfg, new_mu, s_nuk_dga)

        # get full lattice functions
        chi_d_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
        chi_d_q_full[:,q_range] = chi_d_w_q_lam
        chi_m_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
        chi_m_q_full[:,q_range] = chi_m_w_q_lam

        chi_d_lam = np.zeros_like(chi_d_q_full) if rank==0 else None
        chi_m_lam = np.zeros_like(chi_m_q_full) if rank==0 else None

        comm.Reduce(chi_d_q_full, chi_d_lam, op=MPI.SUM, root=0)
        comm.Reduce(chi_m_q_full, chi_m_lam, op=MPI.SUM, root=0)

        if rank==0:
            s_nuk_tail = np.broadcast_to(s_imp[niwf-1000:niwf+1000, None], (2000, Nq)).copy()
            s_nuk_tail[1000-ntail:1000+ntail,:] = s_nuk_dga
            g_nuk_tail = np.broadcast_to(g_imp[niwf-1000:niwf+1000, None], (2000, Nq)).copy()
            g_nuk_tail[1000-ntail:1000+ntail,:] = G_nu_k_dga

            norm = beta*Nq
            epot1 = np.sum(g_nuk_tail*(s_nuk_tail-lam/2))/norm + ht*0.5
            epot2 = 0.5*u/beta * np.sum((chi_d_lam - chi_m_lam)@weights/np.sum(weights)) + u*n**2
            epot2 += np.sum(((chi_d_lam)*uw.reshape(2*n4iwb+1,1))@weights)/(beta*np.sum(weights)) - 2*lam*n**2
            #epot2 -= g0/omega0 * 2*g0**2 * np.sum(((chi_d_lam)*((-wmats**2 + omega0**2)/(wmats**2 + omega0**2)**2).reshape(2*n4iwb+1,1))@weights/np.sum(weights))/beta
            res = (epot2-epot1).real
        else:
            res = None
        
        res = comm.bcast(res, root=0)
        epot_res.append(res)
    
    if rank==0:
        plt.figure()
        plt.plot(lam_ds, epot_res, ".-")
        plt.axhline(0, c="k", ls="--")
        plt.savefig("lam_d_v_epot.pdf", bbox_inches="tight")


    # call epot rootfinder
    args = (lam_m_sol, dga_cfg, beta, chi_d_latt, chi_m_latt, chi_d_loc, chi_m_loc, lambda_maxpole_d, lambda_maxpole_m, weights, v_d_w_q, v_m_w_q, A_d, A_m, chi_d_w_q, chi_m_w_q, chi0_w_q, mu, G_nu_k, s_nuk_loc, comm)
    root_sol = root_scalar(root_epot, args=args, x0=np.log(0.2), method="bisect", bracket=(lambda_maxpole_d+0.5, 20), rtol=1e-8)
    lambda_sol_d = lambda_maxpole_d + np.exp(root_sol.root)

    if rank==0:
        if(root_sol.converged):
            print("After ",root_sol.function_calls," function evaluations, the root is found to be ",lambda_sol_d)
        else:
            print("Root finding did not converge. The best estimate is ",lambda_sol_d)

    # get final lam_m for given lam_d only on root
    if rank==0:
        root_sol = root(root_pauli_epot, args=(beta, chi_d_latt, chi_m_latt, chi_d_loc, chi_m_loc, root_sol.root, lambda_maxpole_d, lambda_maxpole_m, weights), x0=np.log(0.2), method="lm", tol=1e-8)
        lambda_sol_m = lambda_maxpole_m + np.exp(root_sol.x)

    if rank==0:
        if(root_sol.success):
            print("After ",root_sol.nfev," function evaluations, the root is found to be ",lambda_sol_m)
        else:
            print("Root finding did not converge. The best estimate is ",lambda_sol_m)

    return (lambda_sol_d, lambda_sol_m) if rank==0 else (None, None)


# function to determine the correct lambda_d(lambda_m) to fullfill E_pot consistency
def root_epot(lam_d, lam_m_sol, dga_cfg, beta, chi_d_latt, chi_m_latt, chi_d_loc, chi_m_loc, lambda_maxpole_d, lambda_maxpole_m, weights, v_d_w_q, v_m_w_q, A_d, A_m, chi_d_w_q, chi_m_w_q, chi0_w_q, mu, G_nu_k, s_nuk_loc, comm):
    u = dga_cfg.U
    omega0=dga_cfg.w0
    g0=dga_cfg.g0
    n=dga_cfg.occ_imp
    n4iwf = dga_cfg.n4iwf
    n4iwb = dga_cfg.n4iwb
    niwf = dga_cfg.niwf
    ntail = s_nuk_loc.shape[0]//2
    Nq = dga_cfg.n_qpoints_fullbz
    n_qpoints = dga_cfg.n_qpoints
    s, e = get_q_slice_indices(dga_cfg)
    q_range = slice(s, e)

    g_imp = dga_cfg.g_imp
    s_imp = dga_cfg.s_imp

    lam = 2*g0**2/omega0
    ht = n*( u  - (4.*g0**2/omega0) )

    wmats = util.build_w_mats(n4iwb,beta)
    numats= util.build_nu_mats(n4iwf,beta)
    uw = util.Udyn_arr(omega=wmats, omega0=omega0, g=g0, u=0.)

    rank = comm.Get_rank()

    lam_d = float(lam_d)

    # get lam_m for given lam_d only on root
    if rank==0:
        root_sol = root(root_pauli_epot, args=(beta, chi_d_latt, chi_m_latt, chi_d_loc, chi_m_loc, lam_d, lambda_maxpole_d, lambda_maxpole_m, weights), x0=np.log(0.2), method="lm", tol=1e-8, options={'xtol':1e-3})
        lam_m_sol = lambda_maxpole_m + np.exp(root_sol.x)
        lam_d_sol = lambda_maxpole_d + np.exp(lam_d)
    else:
        lam_d_sol = None

    # bcast new lam_r
    lam_d_sol = comm.bcast(lam_d_sol, root=0)
    lam_m_sol = comm.bcast(lam_m_sol, root=0)

    # correct susceptibilities on each process
    chi_d_w_q_lam = chi_d_w_q / (1 + lam_d_sol*chi_d_w_q)
    chi_m_w_q_lam = chi_m_w_q / (1 + lam_m_sol*chi_m_w_q)

    # sde for selfenergy
    sigma_dga_q = sde.Hubbard_Holstein_SDE(dga_cfg, v_d_w_q, v_m_w_q, A_d, A_m, chi_d_w_q_lam, chi_m_w_q_lam, chi0_w_q, mu, G_nu_k, s_nuk_loc)

    sigma_dga = np.zeros_like(sigma_dga_q, dtype=np.complex128) if rank==0 else None
    comm.Reduce(sigma_dga_q, sigma_dga, op=MPI.SUM, root=0)
    
    s_nuk_dga = np.copy(s_nuk_loc)
    s_nuk_dga[ntail-n4iwf:ntail+n4iwf,:] = sigma_dga

    new_mu=0.0
    if(rank==0):
        new_mu = util.get_mu(dga_cfg, s_nuk_dga, verbose=False)
    new_mu = comm.bcast(new_mu, root=0)
    
    G_nu_k_dga = bse.G_nu_k(dga_cfg, new_mu, s_nuk_dga)

    # get full lattice functions
    chi_d_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
    chi_d_q_full[:,q_range] = chi_d_w_q_lam
    chi_m_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
    chi_m_q_full[:,q_range] = chi_m_w_q_lam

    chi_d_lam = np.zeros_like(chi_d_q_full) if rank==0 else None
    chi_m_lam = np.zeros_like(chi_m_q_full) if rank==0 else None

    comm.Reduce(chi_d_q_full, chi_d_lam, op=MPI.SUM, root=0)
    comm.Reduce(chi_m_q_full, chi_m_lam, op=MPI.SUM, root=0)

    if rank==0:
        s_nuk_tail = np.broadcast_to(s_imp[niwf-1000:niwf+1000, None], (2000, Nq)).copy()
        s_nuk_tail[1000-ntail:1000+ntail,:] = s_nuk_dga
        g_nuk_tail = np.broadcast_to(g_imp[niwf-1000:niwf+1000, None], (2000, Nq)).copy()
        g_nuk_tail[1000-ntail:1000+ntail,:] = G_nu_k_dga

        norm = beta*Nq
        epot1 = np.sum(g_nuk_tail*(s_nuk_tail-lam/2))/norm + ht*0.5
        epot2 = 0.5*u/beta * np.sum((chi_d_lam - chi_m_lam)@weights/np.sum(weights)) + u*n**2
        epot2 += np.sum(((chi_d_lam)*uw.reshape(2*n4iwb+1,1))@weights)/(beta*np.sum(weights)) - 2*lam*n**2
        #epot2 -= g0/omega0 * 2*g0**2 * np.sum(((chi_d_lam)*((-wmats**2 + omega0**2)/(wmats**2 + omega0**2)**2).reshape(2*n4iwb+1,1))@weights/np.sum(weights))/beta
        res = (epot2-epot1).real
    else:
        res = None
    
    res = comm.bcast(res, root=0)

    return res


#PAULI PRINCIPLE
# sum_wq chi^L_up,up = sum_wq ( chi_d + chi^L_m) = (n/2-1)*(n/2)
@jit(nopython=True)
def root_pauli_epot( lambda_m:np.float64, beta:np.float64, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, chi_d_loc:np.ndarray, chi_m_loc:np.ndarray, lambda_d:np.float64, lambda_maxpole_d: np.float64, lambda_maxpole_m:np.float64, weights:np.ndarray) -> np.float64:
    n4iwb = chi_d_latt.shape[0]//2;  Nq = chi_m_latt.shape[1]
    latt_sum = 0.5*( np.sum( (chi_d_latt/(1+(lambda_maxpole_d+np.exp(lambda_d))*chi_d_latt)) @weights/np.sum(weights) + (chi_m_latt/(1.0+(lambda_maxpole_m+np.exp(lambda_m))*chi_m_latt))@weights/np.sum(weights) ) ).real/beta
    return 0.5*np.sum(chi_d_loc+chi_m_loc).real/beta  - latt_sum



def lambda_correction(dga_cfg:DGA_ConfigType, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray) -> Tuple[Union[np.float64, np.ndarray], Union[np.float64, np.ndarray]]:
    lambda_type = dga_cfg.lambda_type
    beta = dga_cfg.beta
    chi_d_loc = dga_cfg.chi_d_loc    
    chi_m_loc = dga_cfg.chi_m_loc
    weights = dga_cfg.weights


    match lambda_type:
        case "Pauli":
            lambda_m = get_lambda_m_pauli(beta, chi_d_latt, chi_m_latt,chi_d_loc, chi_m_loc, weights)
            lambda_d = 0.0
        case "Uniform":
            lambda_m = get_lambda_uniform(beta, chi_m_latt, chi_m_loc, weights)
            lambda_d = get_lambda_uniform(beta, chi_d_latt, chi_d_loc, weights)
        case "Dynamical":
            lambda_m = get_lambda_wdep(beta, chi_m_latt, chi_m_loc, weights)
            lambda_d = get_lambda_wdep(beta, chi_d_latt, chi_d_loc, weights)
        case "None":
            lambda_m = 0
            lambda_d = 0
        case default:
            raise ValueError(f"Wrong lambda_type={lambda_type}")
    return lambda_d, lambda_m

#PAULI PRINCIPLE
# sum_wq chi^L_up,up = sum_wq ( chi_d + chi^L_m) = (n/2-1)*(n/2)
@jit(nopython=True)
def root_pauli( lambda_m:np.float64, beta:np.float64, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, chi_d_loc:np.ndarray, chi_m_loc:np.ndarray, lambda_maxpole:np.float64, weights:np.ndarray) -> np.float64:
    n4iwb = chi_d_latt.shape[0]//2;  Nq = chi_m_latt.shape[1]
    latt_sum = 0.5*( np.sum( chi_d_latt@weights/np.sum(weights) + (chi_m_latt/(1.0+(lambda_maxpole+np.exp(lambda_m))*chi_m_latt))@weights/np.sum(weights) ) ).real/beta
    return 0.5*np.sum(chi_d_loc+chi_m_loc).real/beta  - latt_sum

def get_lambda_m_pauli( beta:np.float64, chi_d_latt:np.ndarray, chi_m_latt:np.ndarray, chi_d_loc:np.ndarray, chi_m_loc:np.ndarray, weights:np.ndarray) -> np.float64:
    n4iwb = chi_m_latt.shape[0]//2
    lambda_maxpole = -np.min(1/chi_m_latt[n4iwb,:].real)
    root_sol = root(root_pauli,args=(beta,chi_d_latt,chi_m_latt,chi_d_loc,chi_m_loc,lambda_maxpole,weights),x0=np.log(0.2),method="lm",tol=1e-8)
    lambda_sol = lambda_maxpole+np.exp(root_sol.x)
    if(root_sol.success):
        print("After ",root_sol.nfev," function evaluations, the root is found to be ",lambda_sol)
    else:
        print("Root finding did not converge. The best estimate is ",lambda_sol)
    return lambda_sol

#UNIFORM CHI_LOC = CHI_LATT
@jit(nopython=True)
def root_function_uniform(lambda_r:np.float64, beta:np.float64, chi_r_latt:np.ndarray, chi_r_loc:np.ndarray, lambda_maxpole:np.float64, weights:np.ndarray) -> np.float64:
    # may want to assert dimensions of Chi_w_q and Chi_imp_w
    Nqs = chi_r_latt.shape[1]
    return np.sum( chi_r_loc - ( chi_r_latt/(1.0+(lambda_maxpole+np.exp(lambda_r))*chi_r_latt))@weights/np.sum(weights)  ).real/beta

def get_lambda_uniform(beta:np.float64, chi_r_latt:np.ndarray, chi_r_loc:np.ndarray, weights:np.ndarray) -> np.float64:
    n4iwb = chi_r_latt.shape[0]//2
    lambda_maxpole = -np.min(1/chi_r_latt[n4iwb,:].real)
    print("Finding the uniform lambda_r")
    root_sol = root(root_function_uniform,args=(beta,chi_r_latt,chi_r_loc,lambda_maxpole,weights),x0=np.log(0.2),method="lm",tol=1e-8)
    lambda_sol = lambda_maxpole+np.exp(root_sol.x)
    if(root_sol.success): 
        print("After ",root_sol.nfev," iterations, the root is found to be ",lambda_sol)
    else:
        print("Root finding did not converge. The best estimate is ",lambda_sol)
    return lambda_sol

#W-DEPENDENT CHI_LOC = CHI_LATT
@jit(nopython=True)
def root_function_wdep(lambda_r_w:np.float64, chi_r_latt:np.ndarray,chi_r_loc:np.float64,lambda_w_mp:np.float64, weights:np.ndarray) -> np.float64:
    # may want to assert dimensions of Chi_w_q and Chi_imp_w
    Nqs = chi_r_latt.shape
    Chi_lambda_w = ( chi_r_latt/(1.0+(lambda_w_mp+np.exp(lambda_r_w))*chi_r_latt))@weights/np.sum(weights)
    return (chi_r_loc - Chi_lambda_w)

def get_lambda_wdep(beta:np.float64, chi_r_latt:np.ndarray, chi_r_loc:np.ndarray, weights:np.ndarray) -> np.ndarray:
    print("Finding the w-dependent lambda_r")
    n4iwb = chi_r_latt.shape[0]//2
    lambda_w_mp = -np.min(1.0/chi_r_latt[n4iwb,:].real)
    lambda_sol = np.zeros(n4iwb*2+1)
    for iw in range(n4iwb*2+1):
        root_sol = root(root_function_wdep,args=(chi_r_latt[iw,:].real,chi_r_loc[iw].real,lambda_w_mp,weights),x0=np.log(0.1),method="lm")
        lambda_sol[iw] = lambda_w_mp+np.exp(root_sol.x)
    if(root_sol.success):
        print("After ",root_sol.nfev," function evaluations, the root is found to be ",lambda_sol)
    else:
        print("Root finding did not converge. The best estimate is ",lambda_sol)
    return lambda_sol, lambda_w_mp
