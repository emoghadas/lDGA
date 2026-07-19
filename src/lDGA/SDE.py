import numpy as np
from lDGA.utilities import ik2k, k2ik, G_wq_given_nuk, G_wq_given_nuk_irr, Udyn_arr, G_w_given_nu, U_trans, Udyn, build_nu_mats, build_w_mats,Udyn_arr
from lDGA.bse import asymp_chi
import lDGA.bse as bse 
import numba as nb
from numba import jit
from numba.experimental import jitclass
from lDGA.config import DGA_ConfigType
from mpi4py import MPI


# Lattice Swinger-Dyson for the Hubbard model
#TODO: CHECK FOR POSSIBLE BUGS AT LEAST ON SELF_SUM_U
def Hubbard_SDE(dga_cfg:DGA_ConfigType, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d_w_q:np.ndarray, chi_m_w_q:np.ndarray, F_d_loc:np.array, F_m_loc:np.array, chi0_nu_w_q:np.ndarray, self_old:np.ndarray, dens:np.float64, qpoints:np.ndarray, Nk:int, mu:np.float64, dim:int=2, self_dga:np.ndarray=None) -> np.ndarray:
    u=dga_cfg.U; beta=dga_cfg.beta
    n4iwf=dga_cfg.n4iwf; n4iwb=dga_cfg.n4iwb
    _ , Nq = chi_d_w_q.shape

    self_energy = np.zeros( (2*n4iwf,Nk), dtype=np.complex128)
    F_updn = F_d_loc - F_m_loc


    theta_nu_w_q = np.zeros( (2*n4iwf,2*n4iwb+1,Nq), dtype=np.complex128)
    theta_nu_w_q -= 2.0*np.einsum('ijk,jkm->ikm', F_updn, chi0_nu_w_q)
    theta_nu_w_q += 2.0 + gamma_d + u*np.einsum('ijk,jk->ijk',gamma_d, chi_d_w_q) \
                         -3.0*gamma_m + 3.0*u*np.einsum('ijk,jk->ijk',gamma_m, chi_m_w_q)
    self_energy = self_sum_U(self_old, theta_nu_w_q, u, beta, qpoints, Nk, dim, mu, self_dga)
    #Hartree term

    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()
    #Hartree only on the master node
    if(mpi_rank==0):
        self_energy[:,:] += dens*u

    return self_energy

# Lattice Swinger-Dyson for the Hubbard-Holstein model
def Hubbard_Holstein_SDE(dga_cfg:DGA_ConfigType, gamma_d:np.ndarray, gamma_m:np.ndarray, A_d:np.ndarray, A_m:np.ndarray, chi_d_w_q:np.ndarray, chi_m_w_q:np.ndarray, chi0_nu_w_q:np.ndarray, mu:np.float64, G_nu_k:np.ndarray, self_dga:np.ndarray=None) -> np.ndarray:
    #Here we also sum Fock term

    u=dga_cfg.U; beta=dga_cfg.beta; dens=dga_cfg.occ_imp
    g0=dga_cfg.g0; omega0=dga_cfg.w0
    ts=dga_cfg.ts
    n4iwf=dga_cfg.n4iwf; n4iwb=dga_cfg.n4iwb
    self_old=dga_cfg.s_imp
    g_old=dga_cfg.g_imp
    F_d_loc = dga_cfg.F_d_loc
    F_m_loc = dga_cfg.F_m_loc
    dim = dga_cfg.kdim
    irrbz = dga_cfg.irrbz
    all_q_sym = dga_cfg.all_q_sym
    symq_weights = dga_cfg.symq_weights
    qpoints = dga_cfg.q_grid_loc
    Nk = dga_cfg.n_kpoints
    Nqtot = dga_cfg.n_qpoints_fullbz
    Nq = chi_d_w_q.shape[1]
    asymp = dga_cfg.asymp

    self_energy = np.zeros( (2*n4iwf,Nk), dtype=np.complex128)

    wmats = build_w_mats(n4iwb,beta)
    numats= build_nu_mats(n4iwf,beta)

    uw = Udyn_arr(omega=wmats, omega0=omega0, g=g0, u=u)
    udyn_arr = Udyn_arr(build_w_mats(n4iwb,beta),omega0,g0).reshape(2*n4iwb+1,1)
    ununup = U_trans(nu=numats, nup=numats, omega0=omega0, g=g0, u=u)
    u_d = np.reshape(2*uw-u, shape=(2*n4iwb+1,1)) # will  be U_d(w,nu)
    u_m = np.reshape(-u, shape=(1,1) ) # will be U_m(w,nu)

    #USING OUR FORMULA
    theta_nu_wq = np.zeros( (2*n4iwf,2*n4iwb+1,Nq), dtype=np.complex128)

    theta_nu_wq += -4.0*ununup[0,0] + 2.0*np.reshape(uw, shape=(1,len(uw),1)) # U terms

    if asymp=='bubble' or asymp=='dual':
        theta_nu_wq += -2*np.einsum('j,ijk->ijk',uw,gamma_d) + (A_d + 3*A_m)/beta # 34.1
    else:
        theta_nu_wq += -2*np.einsum('j,ijk->ijk',uw,gamma_d) + u*(gamma_d + 3*gamma_m) + (A_d + 3*A_m)/beta # here electronic and phononic contributions in A_r are separated

    theta_nu_wq +=  np.einsum('ijk,j,jk,mjk,mjk->ijk',gamma_d, 2*uw, u_d*(1-u_d*chi_d_w_q), gamma_d, chi0_nu_w_q)/beta**2 # 34.2
    theta_nu_wq +=  asymp_chi(2*n4iwf, beta) * np.einsum('ijk,j,jk->ijk',gamma_d, 2*uw, u_d*(1-u_d*chi_d_w_q))/beta**2 # asymptotic contribution

    theta_nu_wq -=  np.einsum('ijk,jk,im,mjk,mjk->ijk',gamma_d,u_d*(1-u_d*chi_d_w_q),ununup,gamma_d,chi0_nu_w_q)/beta**2 # 34.3
    theta_nu_wq -=  u*asymp_chi(2*n4iwf, beta)*np.einsum('ijk,jk->ijk',gamma_d,u_d*(1-u_d*chi_d_w_q))/beta**2 # asymptotic contribution

    theta_nu_wq -= 3*np.einsum('ijk,jk,im,mjk,mjk->ijk',gamma_m,u_m*(1-u_m*chi_m_w_q),ununup,gamma_m,chi0_nu_w_q)/beta**2 # 34.4
    theta_nu_wq -= u*asymp_chi(2*n4iwf, beta)*3*np.einsum('ijk,jk->ijk',gamma_m,u_m*(1-u_m*chi_m_w_q))/beta**2 # asymptotic contribution

    theta_nu_wq -= 2*np.einsum('j,imj,mjk->ijk',uw,F_d_loc,chi0_nu_w_q)/beta**2 #local part

    #SHOULD BE ZERO BUT IN HUBBARD HELPS CONVERGING
    theta_nu_wq += 2*u*np.einsum('imj,mjk->ijk', F_d_loc+F_m_loc,chi0_nu_w_q)/(beta**2) # should be zero, subtracting the antiadiabatic part


    comm = MPI.COMM_WORLD
    mpi_rank = comm.Get_rank()

    #Here also Fock term
    #N.B. now working only for a local self-energy, for SC-DGA to be corrected for a k-dependent one
    if dga_cfg.fft:
        if irrbz:
            fft_q_sym = all_q_sym
            fft_q_weights = symq_weights
        else:
            fft_q_sym = qpoints[:, None, :]
            fft_q_weights = np.ones(Nq, dtype=np.float64)
        self_energy = self_sum_Uw_fft(
            theta_nu_wq-2*udyn_arr.reshape(1,2*n4iwb+1,1),
            beta, fft_q_sym, fft_q_weights, Nk, Nqtot, dim, G_nu_k
        )
    elif irrbz:
        #self_energy = self_sum_Uw_irr(self_old, g_old, theta_nu_wq, omega0, g0, udyn_arr, beta, qpoints, all_q_sym, symq_weights, Nk, Nqtot, dim, mu, mpi_rank, ts, self_dga)
        self_energy = self_sum_Uw_test(self_old, g_old, theta_nu_wq-2*udyn_arr.reshape(1,2*n4iwb+1,1), omega0, g0, udyn_arr, beta, qpoints, all_q_sym, symq_weights, Nk, Nqtot, dim, mu, mpi_rank, ts, G_nu_k, self_dga)
    else:
        self_energy = self_sum_Uw(self_old, g_old, theta_nu_wq, omega0, g0, udyn_arr, beta, qpoints, Nk, Nqtot, dim, mu, mpi_rank, ts, self_dga)
    
    #Hartree term
    #Hartree only on the master node
    if(mpi_rank==0):
        self_energy += (dens*( u - (4.0*g0**2/omega0) ) + g0**2/omega0)

    return self_energy # Swinger-Dyson for the Hubbard-Holstein model


def _fft_plus_convolution(theta_q:np.ndarray, green_k:np.ndarray, dim:int) -> np.ndarray:
    """Return sum_q theta(q) * green(k+q) for one or more leading batches."""
    nk = int(round(theta_q.shape[-1] ** (1.0 / dim)))
    grid_shape = theta_q.shape[:-1] + (nk,) * dim
    momentum_axes = tuple(range(theta_q.ndim - 1, theta_q.ndim - 1 + dim))
    theta_grid = theta_q.reshape(grid_shape)
    green_grid = green_k.reshape(grid_shape)

    # ik2k/k2ik store the periodic grid in [-pi, pi), so index zero denotes
    # -pi rather than zero momentum. Shift q to zero-based cyclic indices
    # before applying the convolution theorem.
    theta_grid = np.roll(theta_grid, shift=-(nk//2), axis=momentum_axes)

    # C(k)=sum_q T(q)G(k+q).  The complex conjugations reverse only the
    # momentum argument; unlike a usual correlation, neither input itself is
    # conjugated in the SDE.
    theta_minus_q_fft = np.conj(np.fft.fftn(np.conj(theta_grid), axes=momentum_axes))
    green_fft = np.fft.fftn(green_grid, axes=momentum_axes)
    return np.fft.ifftn(theta_minus_q_fft * green_fft, axes=momentum_axes).reshape(theta_q.shape)


def self_sum_Uw_fft(theta:np.ndarray, beta:np.float64, all_q_sym:np.ndarray,
                    symq_weights:np.ndarray, Nk:int, Nqtot:int, dim:int,
                    G_nu_k:np.ndarray, batch_size:int=8) -> np.ndarray:
    """Momentum-FFT implementation of the irreducible-BZ SDE convolution.

    Each MPI rank expands only its local irreducible q wedge.  Consequently
    this routine returns the same partial q sum as ``self_sum_Uw_test`` and the
    existing Reduce/Allreduce in the caller remains unchanged.  Expansion and
    FFTs are done in small fermionic-frequency batches, so no (nu,w,q_full)
    array is ever allocated.
    """
    nnu, nw, nqloc = theta.shape
    ntail = G_nu_k.shape[0] // 2
    n4iwf = nnu // 2
    n4iwb = nw // 2
    nk_lin = int(round(Nk ** (1.0 / dim)))
    if nk_lin**dim != Nk or Nk != Nqtot:
        raise ValueError("FFT SDE requires identical regular full-BZ k and q grids")
    if nk_lin % 2:
        raise ValueError("FFT SDE requires an even linear momentum-grid size")
    if all_q_sym.shape[0] != nqloc:
        raise ValueError("Symmetry-expanded q grid is inconsistent with theta")

    # Map the local irreducible wedge and all its (possibly repeated) symmetry
    # images to the flattened full BZ once. Repetitions are intentional: the
    # direct path visits all symmetry images and symq_weights compensates them.
    q_indices = np.empty((all_q_sym.shape[1], nqloc), dtype=np.int64)
    for isym in range(all_q_sym.shape[1]):
        for iq in range(nqloc):
            q_indices[isym, iq] = k2ik(all_q_sym[iq, isym], nk_lin)

    self_en = np.zeros((nnu, Nk), dtype=np.complex128)
    batch_size = max(1, min(int(batch_size), nnu))
    prefactor = 0.5 / (beta * Nqtot)

    for iw_idx, iw in enumerate(range(-n4iwb, n4iwb + 1)):
        for start in range(0, nnu, batch_size):
            stop = min(start + batch_size, nnu)
            theta_full = np.zeros((stop - start, Nk), dtype=np.complex128)
            weighted_theta = theta[start:stop, iw_idx, :] * symq_weights[None, :]
            for isym in range(q_indices.shape[0]):
                # A fixed symmetry operation maps distinct irreducible points
                # one-to-one. Coincident images occur between operations and
                # are therefore accumulated by these successive additions.
                theta_full[:, q_indices[isym]] += weighted_theta

            green = G_nu_k[
                ntail - n4iwf + iw + start:
                ntail - n4iwf + iw + stop, :
            ]
            self_en[start:stop] += prefactor * _fft_plus_convolution(
                theta_full, green, dim
            )

    return self_en

#internal auxiliary routine
@jit(nopython=True)
def self_sum_Uw(self_old:np.ndarray, g_old:np.ndarray, theta:np.ndarray,  omega0:np.float64, g0:np.float64, udyn_arr:np.ndarray, beta:np.float64, qpoints:np.ndarray,  Nk:int,  Nqtot:int, dim:int , mu:np.float64, mpi_rank:int, ts:np.ndarray, self_dga:np.ndarray=None) -> np.ndarray:
    n4iwf,n4iwb,Nqloc = theta.shape
    n4iwf//=2; n4iwb=n4iwb//2
    niwf = g_old.shape[0] //2
    Nk_lin = int(np.round(Nk**(1/dim)))

    self_en = np.zeros((2*n4iwf,Nk), dtype=np.complex128)

    for inu in range(-n4iwf,n4iwf):
        nu=(np.pi/beta)*(2*inu+1)
        for ik in range(Nk):
            k = ik2k(ik,dim,Nk_lin)
            G_wq = G_wq_given_nuk(nu,k,self_old,n4iwf,n4iwb,qpoints,Nk_lin,beta,mu,ts,self_dga)
            self_en[inu+n4iwf,ik] +=(0.5/beta)*np.sum( theta[inu+n4iwf,:,:] * G_wq)/Nqtot #vertex term
            if( (g0!=0.0) and (not self_dga is None) ):
                self_en[inu+n4iwf,ik] -= np.sum(G_wq*udyn_arr)/beta/Nqtot
        if( (g0!=0.0 and mpi_rank==0) and ( self_dga is None) ):
            for inup in range(-niwf,niwf):
                nup=(np.pi/beta)*(2*inup+1)
                self_en[inu+n4iwf,:] -= g_old[inup+niwf]*Udyn(nu-nup,omega0,g0,u=0.0)*np.exp(1j*nup*1e-10)/beta
    return self_en


#internal auxiliary routine
@jit(nopython=True)
def self_sum_Uw_irr(self_old:np.ndarray, g_old:np.ndarray, theta:np.ndarray,  omega0:np.float64, g0:np.float64, udyn_arr:np.ndarray, beta:np.float64, qpoints:np.ndarray,  all_q_sym:np.ndarray, symq_weights:np.ndarray, Nk:int,  Nqtot:int, dim:int , mu:np.float64, mpi_rank:int, ts:np.ndarray, self_dga:np.ndarray=None) -> np.ndarray:
    n4iwf,n4iwb,Nqloc = theta.shape
    n4iwf//=2; n4iwb=n4iwb//2
    niwf = g_old.shape[0] //2
    n_sym = 8 if dim==2 else 48
    Nk_lin = int(np.round(Nk**(1/dim)))
    self_en = np.zeros((2*n4iwf,Nk), dtype=np.complex128)

    for inu in range(-n4iwf,n4iwf):
        nu=(np.pi/beta)*(2*inu+1)
        th = np.ascontiguousarray(theta[inu + n4iwf, :, :])
        th1d = th.reshape(-1)
        for ik in range(Nk):
            k = ik2k(ik,dim,Nk_lin)
            Gwq = G_wq_given_nuk_irr(nu, k, self_old, n4iwf, n4iwb, qpoints, all_q_sym, symq_weights, Nk_lin, beta, mu, ts, self_dga)
            # loop over symmetry equivalent q-points
            for i_sym in range(n_sym):
                G1d  = Gwq[i_sym].reshape(-1)
                val = np.dot(th1d, G1d)
                self_en[inu + n4iwf, ik] += (0.5 / beta) * val / Nqtot
                if( (g0!=0.0) and (not self_dga is None) ):
                    self_en[inu+n4iwf,ik] -= np.sum(Gwq[i_sym]*udyn_arr) / beta / Nqtot
        if( (g0!=0.0 and mpi_rank==0) and (self_dga is None)):
            for inup in range(-niwf,niwf):
                nup=(np.pi/beta)*(2*inup+1)
                self_en[inu+n4iwf,:] -= g_old[inup+niwf]*Udyn(nu-nup,omega0,g0,u=0.0)*np.exp(1j*nup*1e-10)/beta
    return self_en


@jit(nopython=True)
def self_sum_Uw_test(self_old:np.ndarray, g_old:np.ndarray, theta:np.ndarray,  omega0:np.float64, g0:np.float64, udyn_arr:np.ndarray, beta:np.float64, qpoints:np.ndarray,  all_q_sym:np.ndarray, symq_weights:np.ndarray, Nk:int,  Nqtot:int, dim:int , mu:np.float64, mpi_rank:int, ts:np.ndarray, G_nu_k:np.ndarray, self_dga:np.ndarray=None) -> np.ndarray:
    n4iwf,n4iwb,Nqloc = theta.shape
    n4iwf//=2; n4iwb=n4iwb//2
    ntail = G_nu_k.shape[0]//2
    niwf = g_old.shape[0] //2
    n_sym = 8 if dim==2 else 48
    Nk_lin = int(np.round(Nk**(1/dim)))
    self_en = np.zeros((2*n4iwf,Nk), dtype=np.complex128)

    for ik in range(Nk):
        k = ik2k(ik,dim,Nk_lin)
        for iw in range(-n4iwb,n4iwb+1):
            G_nupw = G_nu_k[ntail - n4iwf + iw:ntail + n4iwf + iw, :]
            for iq, all_qs in enumerate(all_q_sym):
                for iq_sym,q_sym in enumerate(all_qs):
                    i_kq = k2ik(k+q_sym, Nk_lin)
                    G_nupw_kpq = symq_weights[iq] * G_nupw[:, i_kq]
                    self_en[:,ik] += 0.5 / (beta*Nqtot) * G_nupw_kpq*theta[:,iw+n4iwb,iq]
                    #if (g0!=0):
                    #self_en[:,ik] -= 1.0 / (beta*Nqtot) * G_nupw_kpq*udyn_arr[iw+n4iwb,0]


    return self_en

#internal auxiliary routine
@jit(nopython=True)
def self_sum_U(self_old:np.array, theta:np.ndarray, U:np.float64, beta:np.float64, qpoints:np.ndarray, Nk:int,  Nqtot:int, dim:int, mu:np.float64, self_dga:np.ndarray=None, ts:np.ndarray=None) -> np.ndarray:
    n4iwf,n4iwb,Nqloc = theta.shape
    n4i2f//=2; n2iwb//=2;
    self_en = np.zeros( (2*n4iwf,Nk), dtype=np.complex128)
    for inu in range(-n4iwf,n4iwf):
        nu=(np.pi/beta)*(2*inu+1)
        for ik in range(Nk):
            k = ik2k(ik,dim,int(Nk**(1/dim)))
            self_en[inu,ik] = -(0.5/beta**2)*np.sum( (U*theta[inu,:,:]) * G_wq_given_nuk(nu,k,self_old,n4iwb,qpoints,beta,mu,self_dga,ts=ts)  ) #vertex term
    return self_en/Nqtot






###################################### ONLY FOR LOCAL TESTS PURPOSES ###########################################

def local_sde_check(dga_cfg:DGA_ConfigType):
    beta = dga_cfg.beta
    U = dga_cfg.U
    w0 = dga_cfg.w0
    g0 = dga_cfg.g0
    mu = dga_cfg.mu_imp
    n = dga_cfg.occ_imp
    g = dga_cfg.g_imp
    n4iwf = dga_cfg.n4iwf
    n4iwb = dga_cfg.n4iwb
    g2 = dga_cfg.chi_ph
    chi0_w_full = dga_cfg.chi0_w_full
    chi0_w = dga_cfg.chi0_w
    gamma_d = dga_cfg.gamma_d
    gamma_m = dga_cfg.gamma_m
    F_d_loc = dga_cfg.F_d_loc
    F_m_loc = dga_cfg.F_m_loc
    nouter = dga_cfg.nouter
    asymp = dga_cfg.asymp

    chi0_w_q = chi0_w

    if asymp=='bare-u':
        data = bse.bse_asymp(dga_cfg, chi0_w_q.reshape(*chi0_w_q.shape,1), local=True)
    elif asymp=='dual':
        data = bse.dual_bse(dga_cfg, chi0_w_q.reshape(*chi0_w_q.shape,1), local=True)
    else:
        data = bse.chi_v_r_w_q(dga_cfg, chi0_w_q.reshape(*chi0_w_q.shape,1), local=True)

    chi_d_w_q, v_d_w_q, A_d, chi_m_w_q, v_m_w_q, A_m = data

    sigma_sde_loc = Hubbard_Holstein_SDE_loc(U, g0, w0, beta, v_d_w_q[...,0], v_m_w_q[...,0], A_d[...,0], A_m[...,0], chi_d_w_q[...,0], chi_m_w_q[...,0], F_d_loc, F_m_loc, chi0_w_q, g, n, mu, asymp)
    
    return sigma_sde_loc

# Local Swinger-Dyson for the Hubbard-Holstein model
def Hubbard_Holstein_SDE_loc(u:np.float64, g0:np.float64, omega0:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray,  A_d:np.ndarray, A_m:np.ndarray, chi_d_w:np.ndarray, chi_m_w:np.ndarray, F_d_loc:np.ndarray, F_m_loc:np.ndarray, chi0_nu_w:np.ndarray, g_old:np.ndarray, dens:np.float64, mu:np.float64, asymp:nb.types.string) -> np.ndarray:
    n4iwf = F_d_loc.shape[0]//2
    n4iwb = chi_d_w.shape[0]//2
    self_energy = np.zeros( (2*n4iwf), dtype=np.complex128)

    wmats = build_w_mats(n4iwb,beta)
    numats= build_nu_mats(n4iwf,beta)

    uw = Udyn_arr(omega=wmats, omega0=omega0, g=g0, u=u)
    ununup = U_trans(nu=numats,nup=numats, omega0=omega0, g=g0, u=u)
    u_d = 2*uw-u
    u_m = -u

    #USING OUR FORMULA
    theta_nu_w = np.zeros( (2*n4iwf,2*n4iwb+1), dtype=np.complex128)

    theta_nu_w += -4.0*ununup[0,0] + 2.0*np.reshape(uw, shape=(1,len(uw))) # U terms

    if asymp=='bubble' or asymp=='dual':
        theta_nu_w += -2*np.einsum('j,ij->ij',uw,gamma_d) + (A_d + 3*A_m)/beta # 34.1
    else:
        theta_nu_w += -2*np.einsum('j,ij->ij',uw,gamma_d) + u*(gamma_d + 3*gamma_m) + (A_d + 3*A_m)/beta # here electronic and phononic contributions in A_r are separated

    theta_nu_w +=  np.einsum('ij,j,mj,mj->ij', gamma_d, 2*uw*u_d*(1-u_d*chi_d_w), gamma_d, chi0_nu_w)/beta**2 # 34.2
    theta_nu_w +=  asymp_chi(2*n4iwf, beta) * np.einsum('ij,j,j->ij',gamma_d, 2*uw, u_d*(1-u_d*chi_d_w))/beta**2 # asymptotic contribution

    theta_nu_w -=  np.einsum('ij,j,im,mj,mj->ij', gamma_d, u_d*(1-u_d*chi_d_w), ununup, gamma_d, chi0_nu_w)/beta**2 # 34.3
    theta_nu_w -=  u*asymp_chi(2*n4iwf, beta)*np.einsum('ij,j->ij',gamma_d,u_d*(1-u_d*chi_d_w))/beta**2 # asymptotic contribution

    theta_nu_w -= 3*np.einsum('ij,j,im,mj,mj->ij', gamma_m, u_m*(1-u_m*chi_m_w), ununup, gamma_m, chi0_nu_w)/beta**2 # 34.4
    theta_nu_w -= u*asymp_chi(2*n4iwf, beta)*3*np.einsum('ij,j->ij',gamma_m,u_m*(1-u_m*chi_m_w))/beta**2 # asymptotic contribution

    theta_nu_w -= 2*np.einsum('j,ikj,kj->ij', uw, F_d_loc, chi0_nu_w)/beta**2 #local part

    #TODO: i think this should be a minus sign, but with high enough statistics and large tails it should vanish
    theta_nu_w += 2*u*np.einsum('ijk,jk->ik', F_d_loc+F_m_loc, chi0_nu_w)/(beta**2) # should be zero, subtracting the antiadiabatic part

    #Here also Fock term
    self_energy = self_sum_Uw_loc(g_old, theta_nu_w, omega0,g0, beta)
    #Hartree term
    self_energy += dens*( u - (4.*g0**2/omega0) ) + g0**2/omega0 



    #USING DIRECTLY THE LOCAL SCHWINGER DYSON
    self_energy2  = np.zeros( (2*n4iwf), dtype=np.complex128)
    theta  = np.einsum('ijk,jk,k->ik', 2.0*F_d_loc ,chi0_nu_w,uw-u) #only ph
    theta  += np.einsum('ijk,jk->ik', (F_d_loc-F_m_loc) ,chi0_nu_w)*u #only hubb
    self_energy2 = self_sum_Uw_loc(g_old, theta, omega0, g0, beta)/(beta**2)
    self_energy2 += dens*( u - (4.*g0**2/omega0) ) + (g0**2/omega0) 

    return self_energy #, self_energy2

# Local Swinger-Dyson for the Hubbard model
def Hubbard_SDE_loc(u:np.float64, beta:np.float64, gamma_d:np.ndarray, gamma_m:np.ndarray, chi_d_w:np.ndarray, chi_m_w:np.ndarray, F_d_loc:np.ndarray, F_m_loc:np.ndarray, chi0_nu_w:np.ndarray, g_old:np.ndarray, dens:np.float64, mu:np.float64) -> np.ndarray:
    n4iwf = F_d_loc.shape[0]//2
    n4iwb = chi_d_w.shape[0]//2

    self_energy = np.zeros( (2*n4iwf), dtype=np.complex128)
    theta_nu_w = np.zeros( (2*n4iwf,2*n4iwb+1), dtype=np.complex128)


    theta_nu_w += (2 + gamma_d  - 3*gamma_m)
    theta_nu_w += np.einsum('ijk,jk->ik', F_d_loc-F_m_loc,chi0_nu_w)/(beta**2)

    #ZERO
    theta_nu_w -= np.einsum('ijk,jk->ik', F_d_loc+F_m_loc,chi0_nu_w)/(beta**2) #should be zero

    theta_nu_w -= u*( np.einsum('ij,j->ij',gamma_d,chi_d_w) + 3*np.einsum('ij,j->ij',gamma_m,chi_m_w) )

    theta_nu_w *= -u

    #Here also Fock term
    self_energy = self_sum_Uw_loc(g_old, theta_nu_w, 0.0, 0.0, beta)


    #Hartree term
    self_energy += dens*u

    self_energy2  = np.zeros( (2*n4iwf), dtype=np.complex128)    
    theta  = np.einsum('ijk,jk->ik', (F_d_loc-F_m_loc ),chi0_nu_w)
    self_energy2 = self_sum_Uw_loc(g_old, theta, 0.0, 0.0, beta)*u/(beta**2)

    self_energy2 += dens*u 

    return self_energy, self_energy2

@jit(nopython=True)
def self_sum_Uw_loc(g_old:np.ndarray, theta:np.ndarray, omega0:np.float64, g0:np.float64, beta:np.float64) -> np.ndarray:
    n4iwf,n4iwb = theta.shape
    nfiw=g_old.shape[0]//2
    n4iwf//=2; n4iwb=(n4iwb-1)//2
    self_en = np.zeros((2*n4iwf), dtype=np.complex128)

    for inu in range(-n4iwf,n4iwf):
        nu=(np.pi/beta)*(2*inu+1)
        #Vertex term
        self_en[inu+n4iwf] +=  np.sum(theta[inu+n4iwf,:] * G_w_given_nu(nu,g_old,n4iwb,beta))*(0.5/beta)
        #Fock term
        if(g0!=0.0):
            for inup in range(-nfiw,nfiw):
                nup=(np.pi/beta)*(2*inup+1)
                self_en[inu+n4iwf] -= (g_old[inup+nfiw])*( Udyn(nu-nup,omega0,g0,u=0.0))/beta

    return self_en
