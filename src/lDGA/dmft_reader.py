# dmft_reader.py
import h5py as h5
import numpy as np
import os # For path handling and dummy file creation

# Assuming config.py is in the same package/directory.
# If it's in a different module, adjust the import accordingly.
from .config import DGA_Config 

class DMFT_Reader:
    """
    Class to read DMFT data from an HDF5 file and populate a DGA_Config jitclass object.
    """
    def __init__(self, config: DGA_Config):
        # The key is to store the DGA_Config object as an attribute
        # and then populate *its* fields.
        self.config = config
        self._load_dmft_data()

    def _load_dmft_data(self) -> None:
        try:
            with h5.File(self.config.hdf5_file, 'r') as f:
                # Read general info and set to self.config attributes
                # Ensure type compatibility for Numba: h5py usually returns Python natives
                # for attrs, which Numba converts.
                self.config.beta = float(f['.config'].attrs['general.beta'])
                self.config.U = float(f['.config'].attrs['atoms.1.udd'])
                self.config.mu_imp = float(f['.config'].attrs['general.mu'])
                
                # Reading and calculating occ_imp
                # Ensure the slicing and calculation result in a scalar for occ_imp
                occ = f['stat-last/ineq-001/occ/value'][()]
                self.config.occ_imp = float((occ[:,0,:,0] + occ[:,1,:,1])[0,0]/2)

                # Read frequency infos and set to self.config attributes
                niwf = f['.axes/iw'][()].shape[0] // 2
                n4iwf = f['.axes/iwf-g4'][()].shape[0] // 2
                n4iwb = f['.axes/iwb-g4'][()].shape[0] // 2
                self.config.niwf = niwf
                self.config.n4iwf = n4iwf
                self.config.n4iwb = n4iwb

                # Phononic parameters (HARD CODED AT THE MOMENT)
                self.config.w0 = 1.0
                self.config.g0 = 0.1**0.5

                # Read Green's function and Selfenergy
                # Assign directly to self.config.g_imp and self.config.s_imp
                giw = f['stat-last/ineq-001/giw/value'][0,0,:] * 0.5
                giw += f['stat-last/ineq-001/giw/value'][0,1,:] * 0.5
                self.config.g_imp = giw.astype(np.complex128) # Ensure correct dtype for Numba

                siw = f['stat-last/ineq-001/siw/value'][0,0,:] * 0.5
                siw += f['stat-last/ineq-001/siw/value'][0,1,:] * 0.5
                self.config.s_imp = siw.astype(np.complex128) # Ensure correct dtype for Numba

                # Read 2p-GF and construct chi_ph
                g4iw = f['stat-last/ineq-001/g4iw/value'][()]
                g4iw_uu = 0.5*(g4iw[0,0,0,0,...] + g4iw[0,1,0,1,...])
                g4iw_ud = 0.5*(g4iw[0,0,0,1,...] + g4iw[0,1,0,0,...])
                
                # Check dimensions for g4iw_sym to be safe
                # If g4iw_uu.shape is (a, b, c), then [2, a, b, c]
                g4iw_sym_shape = (2,) + g4iw_uu.shape
                g4iw_sym = np.empty(g4iw_sym_shape, dtype=np.complex128) # Ensure complex128
                g4iw_sym[0,...] = g4iw_uu
                g4iw_sym[1,...] = g4iw_ud
                
                # Trim g according to n4iwf
                nu_range = slice(niwf-n4iwf, niwf+n4iwf)
                g_tr = giw[nu_range]

                # Construct local chi (assuming gg_straight.shape matches expectations for broadcast)
                # Ensure the tensordot result has compatible shape for subtraction
                # Reshape (1,*gg_straight.shape) is for broadcasting.
                gg_straight = np.tensordot(g_tr,g_tr,((),()))

                # Ensure dimensions match for subtraction or broadcasting.
                # Assuming g4iw_sym has a dimension that matches the reshape target
                # and n4iwb is a valid index into that dimension.
                # This line might need further review if `n4iwb` is not intended as an index,
                # or if `g4iw_sym` doesn't have the expected dimension for this indexing.
                # For example, if g4iw_sym is (2, N1, N2, N3) and gg_straight is (N1, N2),
                # you might want to subtract across the last two dims of g4iw_sym.
                # For now, keeping your original line, but be aware of potential shape mismatches if not careful.
                # This complex indexing `[...,n4iwb]` on `g4iw_sym` seems to target a specific
                # "bosonic frequency" slice for modification.
                g4iw_sym[...,n4iwb] = g4iw_sym[...,n4iwb] - gg_straight.reshape(1,*gg_straight.shape) # delta(w,0)
                
                chi_ph = g4iw_sym * self.config.beta
                self.config.chi_ph = chi_ph.astype(np.complex128) # Ensure complex128
        except KeyError as e:
            print(f"Error reading HDF5 file: Missing expected key or attribute: {e}")
            raise
        except Exception as e:
            print(f"An unexpected error occurred during HDF5 loading: {e}")
            raise

if(False):

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
                self.config.beta = f['.config'].attrs['general.beta']
                self.config.U = f['.config'].attrs['atoms.1.udd']
                self.config.mu_imp = f['.config'].attrs['general.mu']
                occ = f['stat-last/ineq-001/occ/value'][()]
                self.config.occ_imp = (occ[:,0,:,0] + occ[:,1,:,1])[0,0]/2

                # read frequency infos
                niwf = f['.axes/iw'][()].shape[0]//2
                n4iwf = f['.axes/iwf-g4'][()].shape[0]//2
                n4iwb = f['.axes/iwb-g4'][()].shape[0]//2
                self.config.niwf = niwf
                self.config.n4iwf = n4iwf
                self.config.n4iwb = n4iwb

                #Phononic parameters (HARD CODED AT THE MOMENT)
                self.config.w0=1.0
                self.config.g0=0.1**0.5

                # read Green's function
                giw = f['stat-last/ineq-001/giw/value'][0,0,:] * 0.5
                giw += f['stat-last/ineq-001/giw/value'][0,1,:] * 0.5
                self.g_imp = giw

                # read Selfenergy
                siw = f['stat-last/ineq-001/siw/value'][0,0,:] * 0.5
                siw += f['stat-last/ineq-001/siw/value'][0,1,:] * 0.5
                self.s_imp = siw

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
                chi_ph = g4iw_sym * self.config.beta
                self.config.chi_ph = chi_ph