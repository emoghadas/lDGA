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
import matplotlib.pyplot as plt

# TODO: check whether this can be done in parallel
dmft_file = "../../example/b53_u2_4_2part-2022-11-19-Sat-07-10-47.hdf5"
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
n4iwb = dga_cfg.n4iwb
kdim = dga_cfg.kdim
nk = dga_cfg.nk
nq = dga_cfg.nq
#TODO: has to be written manually
w0 = dga_cfg.dmft_dict['w0']
g0 = dga_cfg.dmft_dict['g0']

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print("**************************************")
print("Doing Hubbard-Holstein calculation...")
print(f" Here U={u} - g0={g0} - w0={w0}")
print(f"Size of frequencies:")
print(f"niwf: {niwf} - n4iwf: {n4iwf} - n4iwb: {n4iwb}")
lambda_ph = 2*g0**2/w0
print("lambda_ph:",lambda_ph)
print("**************************************")
print()

n_qpoints = nq**kdim
if n_qpoints%size!=0:
    raise ValueError(f"Number of q-points ({n_qpoints}) has to be multiple of number of processors ({size})")
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
q_grid_loc = q_grid[q_range,:]

print("Calculate local bubble")
sys.stdout.flush()

# local bubble on each process
chi0_w = bse.chi0_loc_w(beta, g, n4iwf, n4iwb)

# kgrid has to be initialized beforehand
kpoints = np.linspace(-np.pi, np.pi, nk, endpoint=False)
k_grid = np.meshgrid(kpoints, kpoints)
k_grid = np.array(k_grid).reshape(2,-1).T
n_kpoints = nk**kdim

print("Calculate lattice bubble")
sys.stdout.flush()

# lattice bubble for each processes' q-points
chi0_w_q = bse.chi0_w_q(beta, mu, s, k_grid, q_grid_loc, niwf, n4iwf, n4iwb)

plt.figure()
plt.plot(chi0_w_q.imag[0,:,0],label=f"q={q_grid_loc[0,:]}")
plt.savefig(f"chi0_q{q_grid_loc[0,:]}.pdf")

print("Calculate lattice susceptibility and hedin vertex")
sys.stdout.flush()

# compute chi and v
w_n = 1j*2*np.arange(-n4iwb,n4iwb+1)*np.pi/beta
#u_w = util.Udyn_arr(w_n, w0, g0, u)
u_w = u*np.ones([2*n4iwb+1], dtype=np.float64)
chi_d_w_q, v_d_w_q, _,A_d, chi_m_w_q, v_m_w_q ,_,A_m = bse.chi_v_r_w_q(beta, u, w0, g0, chi0_w, chi0_w_q, chi, n4iwf, n4iwb, q_grid_loc)

for iq,q in enumerate(q_grid_loc):
    plt.figure()    
    plt.plot(A_d[0,:,iq].imag,label=f"q={q}")
    plt.legend()
    plt.savefig(f"A_D_with(qx,qy)_{q[0]:.2f}_{q[1]:.2f}.pdf")


print("Calculate lambda corrections")
sys.stdout.flush()

# lambda corrections
chi_d_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
chi_d_q_full[:,q_range] = chi_d_w_q
chi_m_q_full = np.zeros([2*n4iwb+1, n_qpoints], dtype=np.complex128)
chi_m_q_full[:,q_range] = chi_m_w_q


chi_d_q_full_master = np.zeros_like(chi_d_q_full) if rank==0 else None
chi_m_q_full_master = np.zeros_like(chi_d_q_full) if rank==0 else None

comm.Reduce(chi_d_q_full, chi_d_q_full_master, op=MPI.SUM, root=0)
comm.Reduce(chi_m_q_full, chi_m_q_full_master, op=MPI.SUM, root=0)

if(False):
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
    chi_d_w_q = chi_d_w_q / (1 + lambda_d*chi_d_w_q)
    chi_m_w_q = chi_m_w_q / (1 + lambda_m*chi_m_w_q)

print("Calculate SDE for selfenergy")
sys.stdout.flush()

# sde for selfenergy
F_d_loc, F_m_loc = bse.F_r_loc(beta, chi0_w, chi, n4iwf, n4iwb)
sigma = sde.Hubbard_Holstein_SDE(u, g0, w0, beta, v_d_w_q, v_m_w_q, A_d,A_m, chi_d_w_q, chi_m_w_q, F_d_loc, F_m_loc, chi0_w_q, s, g, n, q_grid_loc,n_kpoints, n_qpoints, mu, kdim)


sigma_full = np.zeros_like(sigma) if rank==0 else None

comm.Reduce(sigma, sigma_full, op=MPI.SUM, root=0)

if(rank==0):
    nu = util.build_nu_mats(n4iwf,beta)
    Nnuloc=s.shape[0]//2
    nuloc = util.build_nu_mats(Nnuloc,beta)
    locslice=slice(Nnuloc-n4iwf,Nnuloc+n4iwf)
                                   


    plt.figure()
    plt.plot(nu,np.sum(sigma_full.imag,axis=-1)/sigma_full.shape[-1],label=f"q-sum")
    plt.plot(nuloc[locslice],s[locslice].imag,":",label="local")
    plt.legend()
    plt.xlim(-20,20); plt.ylim(-0.3,0.3)
    plt.savefig("mpi_sde_q_ImSigma.pdf")
    plt.figure()
    plt.plot(nu,np.sum(sigma_full.real,axis=-1)/sigma_full.shape[-1],label=f"q-sum")
    plt.plot(nuloc[locslice],s[locslice].real,":",label="local")
    plt.legend()
    #plt.xlim(-20,20)
    plt.savefig("mpi_sde_q_ReSigma.pdf")


# save sigma, chi_q
