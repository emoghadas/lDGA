# basic strucutre of ladder-DGA workflow
import os
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
import lDGA.config as cfg
import lDGA.dmft_reader as dmft_reader
import lDGA.bse as bse
import lDGA.utilities as util
import lDGA.lambda_corr as lamb
import lDGA.SDE as sde
from mpi4py import MPI
import sys

# TODO: check whether this can be done in parallel
dmft_file = "../../example/b53_u2_4_2part-2022-11-19-Sat-07-10-47.hdf5"

dga_cfg = cfg.DGA_Config(dmft_file)
reader = dmft_reader.DMFT_Reader(dga_cfg)

beta = np.float64(dga_cfg.dmft_dict['beta'])
mu = dga_cfg.dmft_dict['mu']
u = dga_cfg.dmft_dict['U']
n = dga_cfg.dmft_dict['occ']
g = dga_cfg.dmft_dict['giw']
s = dga_cfg.dmft_dict['siw']
chi = dga_cfg.dmft_dict['chi_ph']
niwf = dga_cfg.niwf
n4iwf = dga_cfg.n4iwf
n4iwb = dga_cfg.n4iwb
kdim = dga_cfg.kdim
nk = dga_cfg.nk
nq = dga_cfg.nq
#TODO: has to be written manually
w0 = 1 #dga_cfg.dmft_dict['w0']
g0 = 0 #dga_cfg.dmft_dict['g0']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_qpoints = nq**kdim
if n_qpoints%size!=0:
    raise ValueError("Number of q-points has to be multiple of number of processors")
nq_local = n_qpoints/size

# q-grid
q = np.linspace(-np.pi,np.pi,nq,endpoint=False)
if kdim==2:
    q_grid = np.meshgrid(q,q)
    q_grid = np.array(q_grid).reshape(2,-1).T
elif kdim==3:
    q_grid = np.meshgrid(q,q,q)
    q_grid = np.array(q_grid).reshape(3,-1).T
else:
    raise ValueError("Number of dimension cannot exceed 3")

# slice for each process
q_range = slice(int(rank*nq_local), int((rank+1)*nq_local))
print(q_range)
q_grid = q_grid[q_range,:]

print("Calculate local bubble")
sys.stdout.flush()

# local bubble on each process
chi0_w = bse.chi0_loc_w(beta, g, n4iwf, n4iwb)

# kgrid has to be initialized beforehand
kpoints = np.linspace(-np.pi, np.pi, nk, endpoint=False)
k_grid = np.meshgrid(kpoints, kpoints)
k_grid = np.array(k_grid).reshape(2,-1).T

print("Calculate lattice bubble")
sys.stdout.flush()

# lattice bubble for each processes' q-points
chi0_q_w = bse.chi0_q_w(beta, mu, s, k_grid, kdim, nk, q_grid, niwf, n4iwf, n4iwb)

print("Calculate lattice susceptibility and hedin vertex")
sys.stdout.flush()

# compute chi and v
w_n = 1j*2*np.arange(-n4iwb,n4iwb+1)*np.pi/beta
#u_w = util.Udyn_arr(w_n, w0, g0, u)
u_w = u*np.ones([2*n4iwb+1], dtype=np.float64)
chi_d_q_w, v_d_q_w, chi_m_q_w, v_m_q_w = bse.chi_v_r_q_w(beta, u_w, chi0_w, chi0_q_w, chi, n4iwf, n4iwb, q_grid, nk)

print("Calculate lambda corrections")
sys.stdout.flush()

# lambda corrections
chi_d_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
chi_d_q_full[:,q_range] = chi_d_q_w
chi_m_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
chi_m_q_full[:,q_range] = chi_m_q_w


chi_d_q_full_master = np.zeros_like(chi_d_q_full) if rank==0 else None
chi_m_q_full_master = np.zeros_like(chi_d_q_full) if rank==0 else None

comm.Reduce(chi_d_q_full, chi_d_q_full_master, op=MPI.SUM, root=0)
comm.Reduce(chi_m_q_full, chi_m_q_full_master, op=MPI.SUM, root=0)

if rank==0:
    chi_d_loc = chi[0,...]+chi[1,...]
    chi_d_loc = np.sum(chi_d_loc, axis=(0,1))/beta**2
    chi_m_loc = chi[0,...]-chi[1,...]
    chi_m_loc = np.sum(chi_m_loc, axis=(0,1))/beta**2
    lambda_d = lamb.get_lambda_uniform(chi_d_q_full_master.real, chi_d_loc.real, beta)
    lambda_m = lamb.get_lambda_uniform(chi_m_q_full_master.real, chi_m_loc.real, beta)
else:
    lambda_d = 0
    lambda_m = 0

lambda_d = comm.bcast(lambda_d, root=0)
lambda_m = comm.bcast(lambda_m, root=0)

# correct susceptibilities on each process
chi_d_q_w = chi_d_q_w / (1 + lambda_d*chi_d_q_w)
chi_m_q_w = chi_m_q_w / (1 + lambda_m*chi_m_q_w)

print("Calculate SDE for selfenergy")
sys.stdout.flush()

# sde for selfenergy
F_d_loc, F_m_loc = bse.F_r_loc(beta, chi0_w, chi, n4iwf, n4iwb)
sigma = sde.Hubbard_Holstein_SDE(u, g0, w0, beta, v_d_q_w, v_m_q_w, chi_d_q_w, chi_m_q_w, F_d_loc, F_m_loc, chi0_q_w, s, n, nk, kdim)


sigma_full = np.zeros_like(sigma) if rank==0 else None

comm.Reduce(sigma, sigma_full, op=MPI.SUM, root=0)

# save sigma, chi_q
