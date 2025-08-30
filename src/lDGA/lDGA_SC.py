# basic strucutre of ladder-DGA workflow
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import h5py
from datetime import datetime
import lDGA.config as cfg
from lDGA.config import DGA_ConfigType
from lDGA.dmft_reader import read_dmft_config
import lDGA.bse as bse
import lDGA.utilities as util
import lDGA.lambda_corr as lamb
import lDGA.SDE as sde
from mpi4py import MPI
import sys
import scipy.optimize as scop
import matplotlib.pyplot as plt

def load_config() -> str:
    # Use first argument if provided, else default to ./dga.toml
    config_path = sys.argv[1] if len(sys.argv) > 1 else "dga.toml"
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return config_path


def distribute_qpoints(dga_cfg:DGA_ConfigType, size, rank) -> None:
    # assign q points to each process
    nq_per_process = dga_cfg.n_qpoints // size
    remainder = dga_cfg.n_qpoints % size
    start_idx = rank * nq_per_process + min(rank, remainder)
    end_idx = start_idx + nq_per_process + (1 if rank < remainder else 0)
    q_range = slice(int(start_idx), int(end_idx))
    q_grid_loc = dga_cfg.q_grid[q_range,:]
    dga_cfg.q_grid_loc = q_grid_loc
    return q_range
    
def main():
    # read config file
    toml_file = load_config()
    dga_cfg = read_dmft_config(toml_file)

    # get impurity quantities
    g = dga_cfg.g_imp
    s = dga_cfg.s_imp
    chi_loc_w = dga_cfg.chi_loc_w
    p3ph = dga_cfg.p3ph
    chi = dga_cfg.chi_ph
    beta = dga_cfg.beta
    mu = dga_cfg.mu_imp
    u = dga_cfg.U
    n = dga_cfg.occ_imp
    w0 = dga_cfg.w0
    g0 = dga_cfg.g0
    niwf = dga_cfg.niwf
    n2iwb = dga_cfg.n2iwb
    n3iwf = dga_cfg.n3iwf
    n3iwb = dga_cfg.n3iwb
    n4iwf = dga_cfg.n4iwf
    n4iwb = dga_cfg.n4iwb

    # asymptotics
    asymp = dga_cfg.asymp
    nouter = dga_cfg.nouter

    # get DGA configs
    max_iter = dga_cfg.max_iter
    lambda_type = dga_cfg.lambda_type
    file_name = dga_cfg.file_name

    # init mpi communicator and get number of processes and ranks
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank==0:
        print("\n")
        print("**************************************")
        print(f"Starting Hubbard-Holstein DGA calculation ...")
        print(f" Here U={u} - g0={g0} - w0={w0}")
        print(f"Size of frequencies:")
        print(f"niwf: {niwf} - n4iwf: {n4iwf} - n4iwb: {n4iwb}")
        lambda_ph = 2*g0**2/w0
        print("lambda_ph:", lambda_ph)
        print(f"n={n} - mu={mu} - beta={beta}")
        print("Hopping amplitudes:")
        print("ts:", dga_cfg.ts)
        print("**************************************")
        sys.stdout.flush()

    
    if rank==0:
        print("\n")
        print("Initializing momentum grids ...")
        sys.stdout.flush()

    # initialize lattice
    dga_cfg.init_lattice()
    util.create_kgrid(dga_cfg)
    util.create_qgrid(dga_cfg)

    # get lattice information
    kdim = dga_cfg.kdim
    irrbz = dga_cfg.irrbz
    nq = dga_cfg.nq
    nk = dga_cfg.nk
    n_kpoints = dga_cfg.n_kpoints
    n_qpoints = dga_cfg.n_qpoints
    n_qpoints_fullbz = dga_cfg.n_qpoints_fullbz

    k_grid = dga_cfg.k_grid
    q_grid = dga_cfg.q_grid
    weights = dga_cfg.weights

    # get current time
    now_obj = datetime.now()
    now = now_obj.strftime("%Y-%m-%d_%H:%M:%S")

    # distribute q points across different processes
    q_range = distribute_qpoints(dga_cfg, size, rank)
    q_grid_loc = dga_cfg.q_grid_loc

    comm.Barrier()
    if rank==0:
        print("Calculate local quantities ... ")
        sys.stdout.flush()

    # local bubble, reducible vertex F and physical susceptibility on each process
    if asymp=='bare-u':
        dga_cfg.chi0_w_full = bse.chi0_loc_w_full(dga_cfg)
        dga_cfg.chi0_w = dga_cfg.chi0_w_full[nouter-n4iwf:nouter+n4iwf,:]
    else:
        dga_cfg.chi0_w = bse.chi0_loc_w(dga_cfg)

    if asymp=='bare-u':
        if rank==0:
            print("Computing irreducible vertex gamma_r from local BSE ...")
        gamma_d, gamma_m = bse.gamma_w(dga_cfg)
        dga_cfg.gamma_d = gamma_d
        dga_cfg.gamma_m = gamma_m

    F_d_loc, F_m_loc = bse.F_r_loc(dga_cfg)
    dga_cfg.F_d_loc = F_d_loc
    dga_cfg.F_m_loc = F_m_loc

    if asymp=='dual':
        chi_loc = chi_loc_w[:,n2iwb-n4iwb:n2iwb+n4iwb+1]
        dga_cfg.chi_d_loc = chi_loc[0,:]
        dga_cfg.chi_m_loc = chi_loc[1,:]
    else:
        chi_d_loc, chi_m_loc = bse.chi_r_loc(dga_cfg)
        dga_cfg.chi_d_loc = chi_d_loc
        dga_cfg.chi_m_loc = chi_m_loc

    if rank==0:
        print("Calculate lattice bubble ...")
        sys.stdout.flush()

    # lattice bubble for each processes' q-points
    chi0_w_q = bse.chi0_w_q(dga_cfg, mu)

    if rank==0:
        print("Calculate lattice susceptibility and hedin vertex ...")
        sys.stdout.flush()

    # compute chi and v of lattice
    match asymp:
        case 'bubble':
            chi_d_w_q, v_d_w_q, A_d, chi_m_w_q, v_m_w_q, A_m = bse.chi_v_r_w_q(dga_cfg, chi0_w_q)
        case 'bare-u':
            chi_d_w_q, v_d_w_q, A_d, chi_m_w_q, v_m_w_q, A_m = bse.bse_asymp(dga_cfg, chi0_w_q)        
        case 'dual':
            chi_d_w_q, v_d_w_q, A_d, chi_m_w_q, v_m_w_q, A_m = bse.dual_bse(dga_cfg, chi0_w_q)

    if rank==0:
        print("Calculate chi_r_latt for lambda corrections ...")
        sys.stdout.flush()

    # lambda corrections
    chi_d_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
    chi_d_q_full[:,q_range] = chi_d_w_q
    chi_m_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
    chi_m_q_full[:,q_range] = chi_m_w_q

    chi_d_latt = np.zeros_like(chi_d_q_full) if rank==0 else None
    chi_m_latt = np.zeros_like(chi_d_q_full) if rank==0 else None

    comm.Reduce(chi_d_q_full, chi_d_latt, op=MPI.SUM, root=0)
    comm.Reduce(chi_m_q_full, chi_m_latt, op=MPI.SUM, root=0)

    lambda_m = np.zeros((n4iwb*2+1,1))
    lambda_d = np.zeros((n4iwb*2+1,1))

    if rank==0:
        print("Doing lambda Correction of type:", lambda_type)
        sys.stdout.flush()
        lambda_d0, lambda_m0 = lamb.lambda_correction(dga_cfg, chi_d_latt, chi_m_latt)
        lambda_d[:,0] = lambda_d0
        lambda_m[:,0] = lambda_m0

    lambda_d = comm.bcast(lambda_d, root=0)
    lambda_m = comm.bcast(lambda_m, root=0)

    # correct susceptibilities on each process
    chi_d_w_q = chi_d_w_q / (1 + lambda_d*chi_d_w_q)
    chi_m_w_q = chi_m_w_q / (1 + lambda_m*chi_m_w_q)

    if rank==0:
        print("Calculate SDE for selfenergy")
        sys.stdout.flush()

    # sde for selfenergy
    sigma_dga_q = sde.Hubbard_Holstein_SDE(dga_cfg, v_d_w_q, v_m_w_q, A_d,A_m, chi_d_w_q, chi_m_w_q, chi0_w_q, mu)

    if(max_iter==1):
        sigma_dga = np.zeros_like(sigma_dga_q,dtype=np.complex128) if rank==0 else None
        comm.Reduce(sigma_dga_q, sigma_dga, op=MPI.SUM, root=0)
    else:
        sigma_dga = np.zeros_like(sigma_dga_q,dtype=np.complex128)
        comm.Allreduce(sigma_dga_q,sigma_dga,op=MPI.SUM)

    #Computing new mu
    new_mu=0.0
    if(rank==0):
        new_mu = util.get_mu(dga_cfg, sigma_dga)
    new_mu = comm.bcast(new_mu, root=0)

    if(rank==0):
        print("Saving data of lambda-corrected DGA ...")
        sys.stdout.flush()
        fsave = h5py.File(f'{file_name}_{now}.h5','a')
        group = fsave.create_group('config')
        group.create_dataset('max_iter',data=max_iter)
        group.create_dataset('beta',data=beta)
        group.create_dataset('dens',data=n)
        group.create_dataset('u',data=u)
        group.create_dataset('w0',data=w0)
        group.create_dataset('g0',data=g0)
        group = fsave.create_group('iter_0')
        group.create_dataset('sigma',data=sigma_dga)
        group.create_dataset('lambda_d',data=lambda_d)
        group.create_dataset('lambda_m',data=lambda_m)
        group.create_dataset('chi_d_latt',data=chi_d_latt)
        group.create_dataset('chi_m_latt',data=chi_m_latt)
        group.create_dataset('mu',data=new_mu)


    # Loop for Self-Consistent lDGA
    for iter in range(1,max_iter):
        if rank==0:
            print("")
            print("")
            print(f"***** Doing iter={iter} *****")
            sys.stdout.flush()


        chi0_w_q = bse.chi0_w_q(dga_cfg, new_mu, s_dga=sigma_dga)

        # compute chi and v
        if asymp=='bubble':
            chi_d_w_q, v_d_w_q, A_d, chi_m_w_q, v_m_w_q, A_m = bse.chi_v_r_w_q(dga_cfg, chi0_w_q)
        else:
            chi_d_w_q, v_d_w_q, A_d, chi_m_w_q, v_m_w_q, A_m = bse.bse_asymp(dga_cfg, chi0_w_q)

        # store new chis in chi_r_latt
        chi_d_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
        chi_d_q_full[:,q_range] = chi_d_w_q
        chi_m_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
        chi_m_q_full[:,q_range] = chi_m_w_q

        chi_d_latt = np.zeros_like(chi_d_q_full) if rank==0 else None
        chi_m_latt = np.zeros_like(chi_d_q_full) if rank==0 else None
        #print(f"Process {rank}: ready to pass chi_lattice")
        comm.Reduce(chi_d_q_full, chi_d_latt, op=MPI.SUM, root=0)
        comm.Reduce(chi_m_q_full, chi_m_latt, op=MPI.SUM, root=0)

        #TODO: LAMBDA DECAY PROCEDURE
        lambda_d *= np.exp(-iter)
        lambda_m *= np.exp(-iter)

        chi_d_w_q = chi_d_w_q / (1.0 + lambda_d*chi_d_w_q)
        chi_m_w_q = chi_m_w_q / (1.0 + lambda_m*chi_m_w_q)

        sigma_dga_q = sde.Hubbard_Holstein_SDE(dga_cfg, v_d_w_q, v_m_w_q, A_d,A_m, chi_d_w_q, chi_m_w_q, chi0_w_q, new_mu, sigma_dga)

        if(rank==0):
            old_mu=new_mu*1
            old_sigma_dga = sigma_dga*1

        sigma_dga = np.zeros_like(sigma_dga_q)
        comm.Allreduce(sigma_dga_q, sigma_dga, op=MPI.SUM)

        new_mu=0.0
        if(rank==0):
            new_mu = util.get_mu(dga_cfg, sigma_dga)
            print("New mu found:", new_mu)
            sys.stdout.flush()
        new_mu = comm.bcast(new_mu, root=0)

        convg=False

        if(rank==0):
            error_sigma = np.linalg.norm( sigma_dga-old_sigma_dga )/np.linalg.norm( sigma_dga )
            if error_sigma < 1e-3 :
                convg=True
        convg = comm.bcast(convg, root=0)

        if(rank==0):
            print(f"Saving data of iteration {iter}")
            sys.stdout.flush()
            fsave = h5py.File(f'{file_name}_{now}.h5','a')
            group = fsave.create_group(f'iter_{iter}')
            group.create_dataset('sigma',data=sigma_dga)
            group.create_dataset('lambda_d',data=lambda_d)
            group.create_dataset('lambda_m',data=lambda_m)
            group.create_dataset('chi_d_latt',data=chi_d_latt)
            group.create_dataset('chi_m_latt',data=chi_m_latt)
            group.create_dataset('mu',data=new_mu)
            group.create_dataset('convg',data=convg)
        if(convg): break
        
    if(rank==0):
        fsave.close()
        if max_iter>1:
            print("After exiting convg=",convg)
            print("error_sigma:",error_sigma)
            sys.stdout.flush()
        print("Finished calculation")
        sys.stdout.flush()

if __name__ == "__main__":
    main()