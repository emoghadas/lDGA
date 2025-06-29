# dmft_reader.py
# dmft_reader.py
import h5py as h5
import numpy as np
from typing import Tuple # For type hinting if needed, though not strictly used in return
import os # For path handling

# Assuming config.py is in the same package/directory.
from .config import DGA_Config 

def read_dmft_config(hdf5_file_path: str, ts=np.array([1.0,0.0],dtype=np.complex128)) -> DGA_Config:
    """
    Reads DMFT data from an HDF5 file and creates a fully populated
    DGA_Config jitclass object.
    """
    # Initialize variables that will be passed to DGA_Config
    # Set sensible defaults or ensure they will be overwritten by HDF5 data
    beta = 0.0
    U = 0.0
    mu_imp = 0.0
    occ_imp = 0.0
    niwf = 0
    n4iwf = 0
    n4iwb = 0
    w0 = 1.0 # Default from DGA_Config, can be overridden if in HDF5
    g0 = 0.1**0.5 # Default from DGA_Config, can be overridden if in HDF5
    
    # Placeholder for mandatory arrays (will be filled from HDF5)
    g_imp_data = None
    s_imp_data = None
    chi_ph_data = None

    try:
        with h5.File(hdf5_file_path, 'r') as f:
            # Read general info
            beta = float(f['.config'].attrs['general.beta'])
            U = float(f['.config'].attrs['atoms.1.udd'])
            mu_imp = float(f['.config'].attrs['general.mu'])
            
            occ = f['stat-last/ineq-001/occ/value'][()]
            occ_imp = float((occ[:,0,:,0] + occ[:,1,:,1])[0,0]/2)

            # Read frequency infos
            niwf = f['.axes/iw'][()].shape[0] // 2
            n4iwf = f['.axes/iwf-g4'][()].shape[0] // 2
            n4iwb = f['.axes/iwb-g4'][()].shape[0] // 2
            
            # Phononic parameters (HARD CODED AT THE MOMENT)
            # These will overwrite DGA_Config's defaults if you wanted them to come from HDF5
            # For now, leaving as hardcoded in the function, consistent with your original code
            # w0 = 1.0 
            # g0 = 0.1**0.5 

            # Read Green's function
            giw = f['stat-last/ineq-001/giw/value'][0,0,:] * 0.5
            giw += f['stat-last/ineq-001/giw/value'][0,1,:] * 0.5
            g_imp_data = giw.astype(np.complex128)

            # Read Selfenergy
            siw = f['stat-last/ineq-001/siw/value'][0,0,:] * 0.5
            siw += f['stat-last/ineq-001/siw/value'][0,1,:] * 0.5
            s_imp_data = siw.astype(np.complex128)

            # Read 2p-GF
            g4iw = f['stat-last/ineq-001/g4iw/value'][()]
            g4iw_uu = 0.5*(g4iw[0,0,0,0,...] + g4iw[0,1,0,1,...])
            g4iw_ud = 0.5*(g4iw[0,0,0,1,...] + g4iw[0,1,0,0,...])
            
            g4iw_sym_shape = (2,) + g4iw_uu.shape
            g4iw_sym = np.empty(g4iw_sym_shape, dtype=np.complex128)
            g4iw_sym[0,...] = g4iw_uu
            g4iw_sym[1,...] = g4iw_ud
            
            nu_range = slice(niwf-n4iwf, niwf+n4iwf)
            g_tr = giw[nu_range]

            gg_straight = np.tensordot(g_tr,g_tr,((),()))
            g4iw_sym[...,n4iwb] = g4iw_sym[...,n4iwb] - gg_straight.reshape(1,*gg_straight.shape)
            
            chi_ph_data = g4iw_sym * beta # Use the loaded beta
            chi_ph_data = chi_ph_data.astype(np.complex128)

    except KeyError as e:
        print(f"Error reading HDF5 file '{hdf5_file_path}': Missing expected key or attribute: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during HDF5 loading from '{hdf5_file_path}': {e}")
        raise

    # Construct and return the DGA_Config instance with all loaded data
    # All mandatory arguments (hdf5_file_path, g_imp_data, s_imp_data, chi_ph_data) are provided.
    # Other arguments can use their defaults from DGA_Config's __init__
    return DGA_Config(
        hdf5_file=hdf5_file_path,
        g_imp=g_imp_data,
        s_imp=s_imp_data,
        chi_ph=chi_ph_data,
        ts=ts,
        
        # Parameters read from file, overriding DGA_Config defaults
        beta=beta,
        U=U,
        mu_imp=mu_imp,
        occ_imp=occ_imp,
        niwf=niwf,
        n4iwf=n4iwf,
        n4iwb=n4iwb,
        
        # Parameters from DGA_Config defaults (if not in HDF5)
        # w0=w0, # You can uncomment this if w0/g0 are read from file or differ from DGA_Config default
        # g0=g0,
    )