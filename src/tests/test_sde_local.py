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

class TestLocSDE(unittest.TestCase):
    def test_U_const(self):
        #dmft_file = "../example/b55_75_u2_4_2part-2024-05-02-Thu-18-57-28.hdf5"
        dmft_file = "../example/b53_u2_4_2part-2022-11-19-Sat-07-10-47.hdf5"

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
        n4iwf = 50 #dga_cfg.n4iwf
        nu_range = slice(chi.shape[1]//2-n4iwf, chi.shape[1]//2+n4iwf)
        chi = chi[:,nu_range,nu_range,:]
        n4iwb = dga_cfg.n4iwb
        kdim = dga_cfg.kdim
        nk = 1 
        nq = 1 
        #TODO: has to be written manually
        w0 = 1 #dga_cfg.dmft_dict['w0']
        g0 = 0 #dga_cfg.dmft_dict['g0']

        print("Calculate local bubble")
        sys.stdout.flush()

        # local bubble on each process
        chi0_w = bse.chi0_loc_w(beta, g, n4iwf, n4iwb)

        print("Calculate local susceptibility and hedin vertex")
        sys.stdout.flush()

        chi_d_loc = chi[0,...]+chi[1,...]
        chi_d_phys = np.sum(chi_d_loc, axis=(0,1))/beta**2 + bse.asymp_chi(2*n4iwf, beta)
        chi_m_loc = chi[0,...]-chi[1,...]
        chi_m_phys = np.sum(chi_m_loc, axis=(0,1))/beta**2 + bse.asymp_chi(2*n4iwf, beta)

        u_w = u*np.ones([2*n4iwb+1], dtype=np.float64)
        _, v_d_w, _, v_m_w = bse.chi_v_r_q_w(beta, u_w, chi0_w, chi0_w.reshape(*chi0_w.shape,1), chi, n4iwf, n4iwb, np.array([[0.,0.]]), 1)
        v_d_w = v_d_w[...,0]
        v_m_w = v_m_w[...,0]

        print("Calculate SDE for selfenergy")
        sys.stdout.flush()

        nu_range = slice(niwf-n4iwf, niwf+n4iwf)

        # sde for selfenergy
        F_d_loc, F_m_loc = bse.F_r_loc(beta, chi0_w, chi, n4iwf, n4iwb)
        sigma = sde.Hubbard_Holstein_SDE_loc(u, g0, w0, beta, v_d_w, v_m_w, chi_d_phys, chi_m_phys, F_d_loc, F_m_loc, chi0_w, g, n, mu)

        plt.plot(sigma.real)
        plt.plot(s[nu_range].real, "--")
        plt.savefig("sde_local_check_real.pdf")

        plt.figure()
        plt.plot(sigma.imag)
        plt.plot(s[nu_range].imag, "--")
        plt.savefig("sde_local_check_imag.pdf")

        assert np.allclose(s[nu_range], sigma, rtol=1e-1, atol=1e-1)

if __name__ == '__main__':
    unittest.main()