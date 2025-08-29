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
    asymp_types = ['bubble', 'bare-u', 'dual']
    if asymp not in asymp_types:
        raise ValueError(f"No implementation for specified asymptotics method: {asymp}")
    nouter = get_config_value(toml_config, "dga.nouter", default=300)
    ts = np.float64(get_config_value(toml_config, "lattice.ts", default=np.array([1.0,0.0])))
    irrbz = get_config_value(toml_config, "lattice.irrbz", default=False)
    nk = get_config_value(toml_config, "lattice.nk", default=4)
    nq = get_config_value(toml_config, "lattice.nq", default=4)

    hdf5_dmftfile_path = get_config_value(toml_config, "dga.dmft_input", default="input.hdf5")
    worm = get_config_value(toml_config, "dga.dmft_worm", default=False)
    dmft_solver = get_config_value(toml_config, "dga.dmft_solver", default="w2dyn")
    if asymp=='dual' and dmft_solver=='w2dyn':
        raise ValueError("Asymptotics based on dual-BSE are currently not implemented for the W2DYNAMICS impurity-solver")

    # Initialize variables that will be passed to DGA_Config
    # Set sensible defaults or ensure they will be overwritten by HDF5 data
    beta = 0.0
    U = 0.0
    mu_imp = 0.0
    occ_imp = 0.0
    niwf = 0
    n2iwb = 0
    n3iwf = 0
    n3iwb = 0
    n4iwf = 0
    n4iwb = 0
    g0 = g02**0.5 # Default from DGA_Config, can be overridden if in HDF5
    
    # Placeholder for mandatory arrays (will be filled from HDF5)
    g_imp_data = None
    s_imp_data = None
    chi_loc_w_data = None
    p3ph_data = None
    chi_ph_data = None

    try:
        with h5.File(hdf5_dmftfile_path, 'r') as f:

            match dmft_solver:
                case 'w2dyn':
            
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

                    chi_ph_data = beta*g4iw_conn(g4iw, giw)
                    chi_ph_data = chi_ph_data.astype(np.complex128)

                case default:
                    U = f['config/U'][()]
                    beta = f['config/beta'][()]
                    occ_imp = f['config/occ'][()]
                    mu_imp = f['config/mu_imp'][()]
                    w0 = f['config/w0'][()]
                    g02 = f['config/g02'][()]
                    g0 = g02**0.5

                    giw = f['data/giw'][()]
                    g_imp_data = giw.astype(np.complex128)
                    siw = f['data/siw'][()]
                    s_imp_data = siw.astype(np.complex128)
                    niwf = g_imp_data.shape[0]//2

                    chi_loc_w = f['data/chi_loc_w'][()]
                    chi_loc_w_data = chi_loc_w.astype(np.complex128)
                    n2iwb = chi_loc_w_data.shape[-1]//2
                    
                    g3iw = f['data/g3iw'][()]
                    n3iwf = g3iw.shape[-2]//2
                    n3iwb = g3iw.shape[-1]//2
                    p3ph_data = g3iw_conn(g3iw, giw, occ_imp, beta)
                    p3ph_data = p3ph_data.astype(np.complex128)
                    
                    g4iw = f['data/g4iw'][()]
                    n4iwf = g4iw.shape[-2]//2
                    n4iwb = g4iw.shape[-1]//2
                    chi_ph_data = g4iw_conn(g4iw, giw)
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
        chi_loc_w = chi_loc_w_data,
        p3ph = p3ph_data,
        chi_ph = chi_ph_data,
        ts = ts,
        
        # Parameters read from  DMFT file, overriding DGA_Config defaults
        beta = beta,
        U = U,
        mu_imp = mu_imp,
        occ_imp = occ_imp,
        niwf = niwf,
        n2iwb = n2iwb,
        n3iwf = n3iwf,
        n3iwb = n3iwb,
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


def g4iw_conn(g4iw, giw):
    niwf = giw.shape[0]//2
    n4iwf = g4iw.shape[-2]//2
    n4iwb = g4iw.shape[-1]//2
    nu_range = slice(niwf-n4iwf, niwf+n4iwf)
    g_tr = giw[nu_range]
    gg_straight = np.tensordot(g_tr,g_tr,((),()))
    g4iw[...,n4iwb] = g4iw[...,n4iwb] - gg_straight.reshape(1, *gg_straight.shape)
    return g4iw 


def g3iw_conn(g3iw, giw, n, beta):
    g = giw
    niwb = g3iw.shape[-1]//2
    niwf = g3iw.shape[-2]//2
    n1iwf = g.shape[-1]//2

    # 1p GF has to be shifted by omega, niwf+omega <= n1iwf 
    if niwf+niwb >n1iwf:
        raise ValueError(
            f"Parameter niwb_max is too large, cannot exceed 1-particle fermionic frequency box size n1iwf = {n1iwf}")

    disc = beta*n*g[n1iwf-niwf:n1iwf+niwf]
    g3iw[:,:,niwb] -= disc

    for iw in range(-niwb, niwb):
        gw0 = g[n1iwf-niwf:n1iwf+niwf]
        gw = g[n1iwf-niwf+iw:n1iwf+niwf+iw]
        norm = -beta * gw0 * gw
        iw_idx = niwb+iw
        g3iw[:,:,iw_idx] /= norm

    return g3iw