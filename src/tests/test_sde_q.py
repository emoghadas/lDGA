import os
os.environ["OMP_NUM_THREADS"] = "1" 
import numpy as np
import lDGA.config as cfg
import lDGA.dmft_reader as dmft_reader
import lDGA.bse as bse
import lDGA.utilities as util
import lDGA.lambda_corr as lamb
import lDGA.SDE as sde
import sys
import unittest
import matplotlib.pyplot as plt

class TestQSDE(unittest.TestCase):
    def test_U_dyn_q(self):
        #dmft_file = "../example/b55_75_u2_4_2part-2024-05-02-Thu-18-57-28.hdf5"
        #dmft_file = "../example/b53_u2_4_2part-2022-11-19-Sat-07-10-47.hdf5"
        #dmft_file = "../example/b55_75_u2_4_2part-2024-11-21-Thu-11-44-14.hdf5" # n4iwb=20
        dmft_file = "../example/gsq0_0_w1_n0_95_2p-2025-03-22-Sat-17-25-49.hdf5"

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
        nk = 48
        nq = 4
        dim=2
        #TODO: has to be written manually
        w0 = dga_cfg.dmft_dict['w0']
        g0 = dga_cfg.dmft_dict['g0']

        print("Doing Hubbard-Holstein calculation...")
        print(f" Here U={u} - g0={g0} - w0={w0}")
        print(f"Size of frequencies:")
        print(f"niwf: {niwf} - n4iwf: {n4iwf} - n4iwb: {n4iwb}")

        lambda_ph = 2*g0**2/w0
        print("lambda_ph:",lambda_ph)

        print("Calculate local bubble")
        sys.stdout.flush()


        #k_grid = util.build_k_grid(nk,dim)
        #q_grid = util.build_k_grid(nq,dim)

        # kgrid has to be initialized beforehand
        kpoints = np.linspace(-np.pi, np.pi, nk, endpoint=False)
        k_grid = np.meshgrid(kpoints, kpoints)
        k_grid = np.array(k_grid).reshape(2,-1).T
        nk *= nk

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
        nq*=nq

        # local bubble on each process
        chi0_w = bse.chi0_loc_w(beta, g, n4iwf, n4iwb)

        print("Calculate q bubble")
        sys.stdout.flush()
        # wq bubble on each process
        chi0_w_q = bse.chi0_w_q(beta, mu, s, k_grid, kdim, nk, q_grid, niwf, n4iwf, n4iwb)

        print("Calculate local susceptibilities")
        sys.stdout.flush()

        #for iw in range(chi.shape[-1]):
        #    chi[0,:,:,iw] = 0.5*(chi[0,:,:,iw] + chi[0,:,:,iw].T )
        #    chi[1,:,:,iw] = 0.5*(chi[1,:,:,iw] + chi[1,:,:,iw].T )
        chi_d_loc = chi[0,...]+chi[1,...]                        
        chi_d_phys = np.sum(chi_d_loc, axis=(0,1))/beta**2 + bse.asymp_chi(2*n4iwf, beta)
        chi_m_loc = chi[0,...]-chi[1,...]
        chi_m_phys = np.sum(chi_m_loc, axis=(0,1))/beta**2 + bse.asymp_chi(2*n4iwf, beta)

        print("Calculate edin vertex")
        sys.stdout.flush()

        chi_d_w_q, v_d_w, vR_d_w, uphi_d_w, chi_m_w_q, v_m_w, vR_m_w, uphi_m_w = bse.chi_v_r_w_q(beta, u, w0, g0, chi0_w, chi0_w_q, chi, n4iwf, n4iwb, q_grid, nk)
        


        print("Calculate SDE for selfenergy")
        sys.stdout.flush()

        nu_range_1 = slice(niwf-n4iwf, niwf+n4iwf)
        nu_range_1 = slice(niwf-n4iwf, niwf+n4iwf)

        # sde for selfenergy
        F_d_loc, F_m_loc = bse.F_r_loc(beta, chi0_w, chi, n4iwf, n4iwb)

        #Doing DGA for dynamic U
        sigma_hh = sde.Hubbard_Holstein_SDE(u, g0, w0, beta, v_d_w, v_m_w, uphi_d_w, uphi_m_w, chi_d_w_q, chi_m_w_q, F_d_loc, F_m_loc, chi0_w_q, s, g, n, q_grid, nk, mu, dim)


        nu = util.build_nu_mats(n4iwf,beta)
        Nnuloc=s.shape[0]//2
        nuloc = util.build_nu_mats(Nnuloc,beta)
        locslice=slice(Nnuloc-n4iwf,Nnuloc+n4iwf)
                                   
        plt.figure()
        plt.title("Re[Sigma]")
        #for ik,k in enumerate(k_grid):
            #plt.plot(nu,sigma_hh[:,ik].real,"-",label=f"(kx,ky)=({k[0]:.3f},{k[1]:.3f})")
        plt.plot(nu, np.sum(sigma_hh.real, axis=-1)/nk, "-")
        plt.plot(nuloc[locslice],s[locslice].real,":",label="local")
#        plt.xlim(100,150)
        plt.ylim(0.35,0.4)
        plt.xlim(-20,20)
        plt.legend()
        plt.savefig("sde_q_check_real.pdf")

        plt.figure()
        plt.title("Im[Sigma]")
        #for ik,k in enumerate(k_grid):
        #    plt.plot(nu,sigma_hh[:,ik].imag,"-",label=f"(kx,ky)=({k[0]:.3f},{k[1]:.3f})")
        plt.plot(nu, np.sum(sigma_hh.imag, axis=-1)/nk, "-")
        plt.plot(nuloc[locslice],s[locslice].imag,":",label="local")
        plt.xlim(-20,20)
        plt.ylim(-0.1,0.1)
        plt.legend()
        plt.savefig("sde_q_check_imag.pdf")


        #assert np.allclose(s[nu_range], sigma, rtol=1e-3, atol=1e-3)

if __name__ == '__main__':
    unittest.main()