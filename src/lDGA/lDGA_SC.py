# basic strucutre of ladder-DGA workflow
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import h5py
from datetime import datetime
import lDGA.config as cfg
from lDGA.dmft_reader import read_dmft_config
import lDGA.bse as bse
import lDGA.utilities as util
import lDGA.lambda_corr as lamb
import lDGA.SDE as sde
from mpi4py import MPI
import sys
import scipy.optimize as scop
import matplotlib.pyplot as plt

# TODO: check whether this can be done in parallel
filenum=1 # 0 is Hubb only - 1 is Hubb-Hol -2 is Hubb-Hol many freq
match filenum:
    case 0:
        dmft_file = "../../example/Hubb/g0_n0_95_2p-2025-04-18-Fri-01-08-44.hdf5"
    case 1:
        dmft_file = "../../example/gsq0_0_w1_n0_95_2p-2025-03-22-Sat-17-25-49.hdf5"
    case 2:
        dmft_file = "../../example/gsq0_0_w1_n0_95_2p-2025-04-14-Mon-16-49-08.hdf5"
    case default:
        raise ValueError(f"Wrong filenum={filenum}")


toml_file="dga.toml"

dga_cfg = read_dmft_config(dmft_file, toml_file )
 #dga_cfg = cfg.DGA_Config(dmft_file)
 #reader = dmft_reader.DMFT_Reader(dga_cfg)

beta = np.float64(dga_cfg.beta)

g = dga_cfg.g_imp
s = dga_cfg.s_imp
chi = dga_cfg.chi_ph

mu = dga_cfg.mu_imp
u = dga_cfg.U
n = dga_cfg.occ_imp
niwf = dga_cfg.niwf
n4iwf = dga_cfg.n4iwf
n4iwb = dga_cfg.n4iwb
kdim = dga_cfg.kdim
irrbz = dga_cfg.irrbz
nq = dga_cfg.nq
if irrbz:
    n_qpoints = int(nq*(nq+1)/2)
    nk = 2*nq-2
    dga_cfg.nk
else:
    n_qpoints = nq**kdim
    nk = dga_cfg.nk
max_iter = dga_cfg.max_iter
w0 = dga_cfg.w0
g0 = dga_cfg.g0
lambda_type = dga_cfg.lambda_type
file_name = dga_cfg.file_name


print(f" Here U={u} - g0={g0} - w0={w0}")


now_obj = datetime.now()
now = now_obj.strftime("%Y-%m-%d_%H:%M:%S")

match filenum:
    case 0:
        w0 = 1.0
        g0 = 0.0
    case 1:
        w0 = 1.0
        g0 = 0.1**0.5 
    case 2:
        w0 = 1.0
        g0 = 0.1**0.5
    case default:
        raise ValueError(f"Wrong filenum={filenum}")



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if(rank==0):
    print("**************************************")
    print("Doing Hubbard-Holstein calculation...")
    print(f" Here U={u} - g0={g0} - w0={w0}")
    print(f"Size of frequencies:")
    print(f"niwf: {niwf} - n4iwf: {n4iwf} - n4iwb: {n4iwb}")
    lambda_ph = 2*g0**2/w0
    print("lambda_ph:",lambda_ph)
    print(f"n={n} - mu={mu} - beta={beta}")
    print("**************************************")

print("ts:",dga_cfg.ts)


q = np.linspace(0, np.pi, nq, endpoint=True)
if irrbz:
    if kdim!=2:
        raise ValueError("Irreducible BZ summation only implemented for d=2")
    q_grid, weights = util.irr_q_grid(q)
else:
    if kdim==2:
        q_grid = np.meshgrid(q,q)
        q_grid = np.array(q_grid).reshape(2,-1).T
    elif kdim==3:
        q_grid = np.meshgrid(q,q,q)
        q_grid = np.array(q_grid).reshape(3,-1).T
    else:
        raise ValueError("Number of dimension cannot exceed 3")
    weights = np.ones(q_grid.shape[0], dtype=np.complex128)

# assign q points to each process
nq_per_process = n_qpoints // size
remainder = n_qpoints % size
start_idx = rank * nq_per_process + min(rank, remainder)
end_idx = start_idx + nq_per_process + (1 if rank < remainder else 0)
q_range = slice(int(start_idx), int(end_idx))
q_grid_loc = q_grid[q_range,:]

print("Calculate local bubble - rank:",rank)
sys.stdout.flush()

# local bubble on each process
chi0_w = bse.chi0_loc_w(dga_cfg)

# kgrid has to be initialized beforehand
kpoints = np.linspace(-np.pi, np.pi, nk, endpoint=False)
k_grid = np.meshgrid(kpoints, kpoints)
k_grid = np.array(k_grid).reshape(2,-1).T
n_kpoints = nk**kdim

print("Calculate lattice bubble - rank:",rank)
sys.stdout.flush()

# lattice bubble for each processes' q-points
chi0_w_q = bse.chi0_w_q(dga_cfg, mu, k_grid, q_grid_loc)

print("Calculate lattice susceptibility and hedin vertex - rank:",rank)
sys.stdout.flush()

# compute chi and v
chi_d_w_q, v_d_w_q, A_d, chi_m_w_q, v_m_w_q ,A_m = bse.chi_v_r_w_q(dga_cfg, chi0_w, chi0_w_q, q_grid_loc)

print("Calculate chi_d/m_latt for lambda corrections - rank:",rank)
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
    print("Lambda Correction of type:", lambda_type)
    chi_d_loc = chi[0,...]+chi[1,...]
    chi_d_loc = np.sum(chi_d_loc, axis=(0,1))/beta**2 + bse.asymp_chi(2*n4iwf, beta) #Tails correction are important to have Re[chi_loc]>0
    chi_m_loc = chi[0,...]-chi[1,...]
    chi_m_loc = np.sum(chi_m_loc, axis=(0,1))/beta**2 + bse.asymp_chi(2*n4iwf, beta) #Tails correction are important to have Re[chi_loc]>0
    lambda_d0, lambda_m0 = lamb.lambda_correction(lambda_type,beta,chi_d_latt,chi_m_latt,chi_d_loc,chi_m_loc, weights)
    lambda_d[:,0] = lambda_d0
    lambda_m[:,0] = lambda_m0



lambda_d = comm.bcast(lambda_d, root=0)
lambda_m = comm.bcast(lambda_m, root=0)


# correct susceptibilities on each process
chi_d_w_q = chi_d_w_q / (1 + lambda_d*chi_d_w_q)
chi_m_w_q = chi_m_w_q / (1 + lambda_m*chi_m_w_q)


print("Calculate SDE for selfenergy")
sys.stdout.flush()

# sde for selfenergy
F_d_loc, F_m_loc = bse.F_r_loc(beta, chi0_w, chi, n4iwf, n4iwb)
sigma_dga_q = sde.Hubbard_Holstein_SDE(dga_cfg, v_d_w_q, v_m_w_q, A_d,A_m, chi_d_w_q, chi_m_w_q, F_d_loc, F_m_loc, chi0_w_q, q_grid_loc, n_kpoints, n_kpoints, mu, irrbz, kdim)

if(max_iter==1):
    sigma_dga = np.zeros_like(sigma_dga_q,dtype=np.complex128) if rank==0 else None
    comm.Reduce(sigma_dga_q, sigma_dga, op=MPI.SUM, root=0)
else:
    sigma_dga = np.zeros_like(sigma_dga_q,dtype=np.complex128)
    comm.Allreduce(sigma_dga_q,sigma_dga,op=MPI.SUM)

#Computing new mu
new_mu=0.0
if(rank==0):
    new_mu = util.get_mu( mu, n, sigma_dga, k_grid, beta )
new_mu = comm.bcast(new_mu, root=0)


if(rank==0):
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
    print("")
    print("")
    print(f"***** Doing iter={iter} *****")


    chi0_w_q = bse.chi0_w_q(dga_cfg, new_mu, k_grid, q_grid_loc, s_dga=sigma_dga)

    chi_d_w_q, v_d_w_q, A_d, chi_m_w_q, v_m_w_q ,A_m = bse.chi_v_r_w_q(dga_cfg, chi0_w, chi0_w_q, q_grid_loc)

    # store new chis in chi_r_latt
    chi_d_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
    chi_d_q_full[:,q_range] = chi_d_w_q
    chi_m_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
    chi_m_q_full[:,q_range] = chi_m_w_q

    chi_d_latt = np.zeros_like(chi_d_q_full) if rank==0 else None
    chi_m_latt = np.zeros_like(chi_d_q_full) if rank==0 else None
    print(f"Process {rank}: ready to pass chi_lattice")
    comm.Reduce(chi_d_q_full, chi_d_latt, op=MPI.SUM, root=0)
    comm.Reduce(chi_m_q_full, chi_m_latt, op=MPI.SUM, root=0)

    #TODO: LAMBDA DECAY PROCEDURE
    lambda_d *= np.exp(-iter)
    lambda_m *= np.exp(-iter)

    chi_d_w_q = chi_d_w_q / (1.0 + lambda_d*chi_d_w_q)
    chi_m_w_q = chi_m_w_q / (1.0 + lambda_m*chi_m_w_q)

    sigma_dga_q = sde.Hubbard_Holstein_SDE(dga_cfg, v_d_w_q, v_m_w_q, A_d,A_m, chi_d_w_q, chi_m_w_q, F_d_loc, F_m_loc, chi0_w_q,  q_grid_loc, n_kpoints, n_kpoints, new_mu, irrbz, kdim, sigma_dga)

    if(rank==0):
        old_mu=new_mu*1
        old_sigma_dga = sigma_dga*1

    sigma_dga = np.zeros_like(sigma_dga_q)
    comm.Allreduce(sigma_dga_q, sigma_dga, op=MPI.SUM)

    new_mu=0.0
    if(rank==0):
        new_mu = util.get_mu( mu, n, sigma_dga, k_grid, beta )
        print("new_mu found:",new_mu)
    new_mu = comm.bcast(new_mu, root=0)

    convg=False

    if(rank==0):
        error_sigma = np.linalg.norm( sigma_dga-old_sigma_dga )/np.linalg.norm( sigma_dga )
        if error_sigma < 1e-3 :
            convg=True
    convg = comm.bcast(convg, root=0)


    if(rank==0):
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
    print("Finished calculation")
