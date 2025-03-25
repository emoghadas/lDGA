import h5py as h5
import numpy as np
from .config import DGA_Config

class DMFT_Reader:
    """
    class to read dmft data from hdf5 file
    """
    def __init__(self, config: DGA_Config):
        self.config = config
        #self.dmft_dict: dict = {}
        self._load_dmft_data()

    def _load_dmft_data(self) -> None:
        with h5.File(self.config.hdf5_file, 'r') as f:
            # read general info
            self.config.dmft_dict['beta'] = f['.config'].attrs['general.beta']
            self.config.dmft_dict['U'] = f['.config'].attrs['atoms.1.udd']
            self.config.dmft_dict['mu'] = f['.config'].attrs['general.mu']
            occ = f['stat-last/ineq-001/occ/value'][()]
            self.config.dmft_dict['occ'] = (occ[:,0,:,0] + occ[:,1,:,1])[0,0]/2

            # read frequency infos
            niwf = f['.axes/iw'][()].shape[0]//2
            n4iwf = f['.axes/iwf-g4'][()].shape[0]//2
            n4iwb = f['.axes/iwb-g4'][()].shape[0]//2
            self.config.niwf = niwf
            self.config.n4iwf = n4iwf
            self.config.n4iwb = n4iwb

            #Phononic parameters
            self.config.dmft_dict['w0']=1.0
            self.config.dmft_dict['g0']=0.1**0.5

            # read Green's function
            giw = f['stat-last/ineq-001/giw/value'][0,0,:] * 0.5
            giw += f['stat-last/ineq-001/giw/value'][0,1,:] * 0.5
            self.config.dmft_dict['giw'] = giw

            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(f['stat-last/ineq-001/giw/value'][0,0,:].imag,"-")
            plt.plot(f['stat-last/ineq-001/giw/value'][0,1,:].imag,":")
            plt.savefig("Gupdw_im.pdf")
            plt.figure()
            plt.plot(f['stat-last/ineq-001/giw/value'][0,0,:].real,"-")
            plt.plot(f['stat-last/ineq-001/giw/value'][0,1,:].real,":")
            plt.savefig("Gupdw_re.pdf")


            # read Selfenergy
            siw = f['stat-last/ineq-001/siw/value'][0,0,:] * 0.5
            siw += f['stat-last/ineq-001/siw/value'][0,1,:] * 0.5
            self.config.dmft_dict['siw'] = siw

            # read 2p-GF
            g4iw = f['stat-last/ineq-001/g4iw/value'][()]
            g4iw_uu = 0.5*(g4iw[0,0,0,0,...] + g4iw[0,1,0,1,...])
            g4iw_ud = 0.5*(g4iw[0,0,0,1,...] + g4iw[0,1,0,0,...])
            g4iw_sym = np.empty([2,*g4iw_uu.shape], dtype=complex)
            g4iw_sym[0,...] = g4iw_uu
            g4iw_sym[1,...] = g4iw_ud
            
            # trim g according to n4iwf
            nu_range = slice(niwf-n4iwf, niwf+n4iwf)
            g_tr = giw[nu_range]

            # construct local chi
            gg_straight = np.tensordot(g_tr,g_tr,((),()))
            g4iw_sym[...,n4iwb] = g4iw_sym[...,n4iwb] - gg_straight.reshape(1,*gg_straight.shape) # delta(w,0)
            chi_ph = g4iw_sym * self.config.dmft_dict['beta']
            self.config.dmft_dict['chi_ph'] = chi_ph