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
dmft_file = "../../example/gsq0_0_w1_n0_95_2p-2025-03-22-Sat-17-25-49.hdf5"

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
nu_range = slice(chi.shape[1]//2-n4iwf, chi.shape[1]//2+n4iwf)
n4iwb = dga_cfg.n4iwb
w_range = slice(chi.shape[-1]//2-n4iwb, chi.shape[-1]//2+n4iwb+1)
chi = chi[:,nu_range,nu_range,w_range]
kdim = dga_cfg.kdim
nk = 2
nq = 2
dim=2
#TODO: has to be written manually
w0 = dga_cfg.dmft_dict['w0']
g0 = dga_cfg.dmft_dict['g0']

print("Doing Hubbard-Holstein calculation...")
print(f" Here U={u} - g0={g0} - w0={w0}")
print(f"Size of frequencies:")
print(f"niwf: {niwf} - n4iwf: {n4iwf} - n4iwb: {n4iwb}")

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

kpoints = np.linspace(-np.pi, np.pi, 48, endpoint=False)
k_grid_bub = np.meshgrid(kpoints, kpoints)
k_grid_bub = np.array(k_grid_bub).reshape(2,-1).T

print("Calculate lattice bubble")
sys.stdout.flush()

# lattice bubble for each processes' q-points
chi0_w_q = bse.chi0_w_q(beta, mu, s, k_grid_bub, 48**2, q_grid, niwf, n4iwf, n4iwb)

print("Calculate lattice susceptibility and hedin vertex")
sys.stdout.flush()

# compute chi and v
w_n = 1j*2*np.arange(-n4iwb,n4iwb+1)*np.pi/beta
#u_w = util.Udyn_arr(w_n, w0, g0, u)
#u_w = u*np.ones([2*n4iwb+1], dtype=np.float64)
chi_d_w_q, v_d_w, vR_d_w, uphi_d_w, chi_m_w_q, v_m_w, vR_m_w, uphi_m_w = bse.chi_v_r_w_q(beta, u, w0, g0, chi0_w, chi0_w_q, chi, n4iwf, n4iwb, q_grid)

print("Calculate SDE for selfenergy")
sys.stdout.flush()

# sde for selfenergy
F_d_loc, F_m_loc = bse.F_r_loc(beta, chi0_w, chi, n4iwf, n4iwb)
sigma = sde.Hubbard_Holstein_SDE(u, g0, w0, beta, v_d_w, v_m_w, uphi_d_w, uphi_m_w, chi_d_w_q, chi_m_w_q, F_d_loc, F_m_loc, chi0_w_q, s, g, n, q_grid, nk**2, mu, dim)


sigma_full = np.zeros_like(sigma) if rank==0 else None

comm.Reduce(sigma, sigma_full, op=MPI.SUM, root=0)

if rank==0:
    import matplotlib.pyplot as plt
    sigma_loc = np.sum(sigma_full, axis=1)/nk**2
    plt.plot(s.imag[niwf-n4iwf:niwf+n4iwf], label="DMFT")
    plt.plot(sigma_loc.imag, "--", label=r"D$\Gamma$A")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\Im\Sigma(\nu)$")
    plt.legend()
    plt.savefig("../../example/test_q_imag.pdf")
    plt.show()

    plt.plot(s.real[niwf-n4iwf:niwf+n4iwf], label="DMFT")
    plt.plot(sigma_loc.real, "--", label=r"D$\Gamma$A")
    plt.xlabel(r"$\nu$")
    plt.ylabel(r"$\Re\Sigma(\nu)$")
    plt.legend()
    plt.savefig("../../test_q_real.pdf")
    plt.show()
    

# save sigma, chi_q
