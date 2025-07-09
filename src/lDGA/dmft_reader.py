# dmft_reader.py
# dmft_reader.py
import h5py as h5
import tomllib
import numpy as np
from lDGA.config import DGA_Config 

def read_dmft_config(toml_dgafile_path:str) -> DGA_Config:
    """
    Reads DMFT data from an HDF5 file and creates a fully populated
    DGA_Config jitclass object.
    """

    # Reading the TOML config file of DGA
    with open(toml_dgafile_path, 'rb') as f:
        toml_config = tomllib.load(f)
    
    g02 = get_config_value(toml_config,"phonons.g02", default=0.1)
    w0 = get_config_value(toml_config,"phonons.w0", default=1.0)
    max_iter = get_config_value(toml_config, "dga.max_iter", default=1)
    file_name = get_config_value(toml_config, "dga.file_name", default="result")
    lambda_type = get_config_value(toml_config, "dga.lambda_type", default="Pauli")
    lambda_decay = get_config_value(toml_config, "dga.lambda_decay", default=1)
    asymp = get_config_value(toml_config, "dga.asymp", default='bubble')
    # check if chosen asymptotics is supported
    asymp_types = ['bubble', 'bare-u']
    if asymp not in asymp_types:
        raise ValueError(f"No implementation for specified asymptotics method: {asymp}")
    nouter = get_config_value(toml_config, "dga.nouter", default=300)
    ts = np.float64(get_config_value(toml_config, "lattice.ts", default=np.array([1.0,0.0])))
    irrbz = get_config_value(toml_config, "lattice.irrbz", default=False)
    nk = get_config_value(toml_config, "lattice.nk", default=4)
    nq = get_config_value(toml_config, "lattice.nq", default=4)

    hdf5_dmftfile_path = get_config_value(toml_config, "dga.dmft_input", default="input.hdf5")
    worm = get_config_value(toml_config, "dga.dmft_worm", default=False)

    # Initialize variables that will be passed to DGA_Config
    # Set sensible defaults or ensure they will be overwritten by HDF5 data
    beta = 0.0
    U = 0.0
    mu_imp = 0.0
    occ_imp = 0.0
    niwf = 0
    n4iwf = 0
    n4iwb = 0
    g0 = g02**0.5 # Default from DGA_Config, can be overridden if in HDF5
    
    # Placeholder for mandatory arrays (will be filled from HDF5)
    g_imp_data = None
    s_imp_data = None
    chi_ph_data = None

    try:
        with h5.File(hdf5_dmftfile_path, 'r') as f:
            # Read general info
            beta = np.float64(f['.config'].attrs['general.beta'])
            U = np.float64(f['.config'].attrs['atoms.1.udd'])
            mu_imp = np.float64(f['.config'].attrs['general.mu'])
            
            if worm:
                occ = f['worm-last/ineq-001/occ/value'][()]    
            else:
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
            if worm:
                giw = f['worm-last/ineq-001/glocnew/value'][0,0,:] * 0.5
                giw += f['worm-last/ineq-001/glocnew/value'][0,1,:] * 0.5
            else:
                giw = f['stat-last/ineq-001/giw/value'][0,0,:] * 0.5
                giw += f['stat-last/ineq-001/giw/value'][0,1,:] * 0.5
            g_imp_data = giw.astype(np.complex128)

            # Read Selfenergy
            if worm:
                g0iw = f['worm-last/ineq-001/g0iw/value'][()]
                giw_full = f['worm-last/ineq-001/glocnew/value'][()]
                shape = g0iw.shape
                siw_full = np.empty(shape, dtype=complex)
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        siw_full[i,j,:] = 1/g0iw[i,j,:] - 1/giw_full[i,j,:]
                siw = 0.5*(siw_full[0,0,:] + siw_full[0,1,:]) 
            else:
                siw = f['stat-last/ineq-001/siw/value'][0,0,:] * 0.5
                siw += f['stat-last/ineq-001/siw/value'][0,1,:] * 0.5
            s_imp_data = siw.astype(np.complex128)

            # Read 2p-GF
            if worm:
                g4iw_uu = 0.5 * f['worm-last/ineq-001/g4iw-worm/00001/value'][()]
                g4iw_uu += 0.5 * f['worm-last/ineq-001/g4iw-worm/00016/value'][()]
                g4iw_ud = 0.5 * f['worm-last/ineq-001/g4iw-worm/00004/value'][()]
                g4iw_ud += 0.5 * f['worm-last/ineq-001/g4iw-worm/00013/value'][()]
            else:
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
            g4iw_sym[...,n4iwb] = g4iw_sym[...,n4iwb] - gg_straight.reshape(1, *gg_straight.shape)
            
            chi_ph_data = g4iw_sym * beta # Use the loaded beta
            chi_ph_data = chi_ph_data.astype(np.complex128)

    except KeyError as e:
        print(f"Error reading HDF5 file '{hdf5_dmftfile_path}': Missing expected key or attribute: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during HDF5 loading from '{hdf5_dmftfile_path}': {e}")
        raise

    # Construct and return the DGA_Config instance with all loaded data
    # All mandatory arguments (hdf5_file_path, g_imp_data, s_imp_data, chi_ph_data) are provided.
    # Other arguments can use their defaults from DGA_Config's __init__
    return DGA_Config(
        hdf5_file = hdf5_dmftfile_path,
        toml_file = toml_dgafile_path,
        g_imp = g_imp_data,
        s_imp = s_imp_data,
        chi_ph = chi_ph_data,
        ts = ts,
        
        # Parameters read from  DMFT file, overriding DGA_Config defaults
        beta = beta,
        U = U,
        mu_imp = mu_imp,
        occ_imp = occ_imp,
        niwf = niwf,
        n4iwf = n4iwf,
        n4iwb = n4iwb,
        
        # Parameters from toml_config
        w0 = w0,
        g0 = g0,
        irrbz = irrbz,
        nk = nk,
        nq = nq,
        max_iter = max_iter,
        file_name = file_name,
        lambda_type = lambda_type,
        lambda_decay = lambda_decay,
        asymp = asymp,
        nouter = nouter
    )


def get_config_value(config_data, key_path, default):
    """
    Safely extracts a value from nested dictionaries (like TOML data)
    and returns a default if the key path does not exist.

    Args:
        config_data (dict): The loaded TOML data.
        key_path (str): A dot-separated string representing the key path (e.g., "database.port").
        default: The value to return if the key is not found.

    Returns:
        The value from the config or the default.
    """
    keys = key_path.split('.')
    current_data = config_data
    for i, key in enumerate(keys):
        if isinstance(current_data, dict) and key in current_data:
            current_data = current_data[key]
        else:
            # If a key in the path is not found, return the default value
            return default
    return current_data