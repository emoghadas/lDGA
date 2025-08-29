from dataclasses import dataclass
from numba.experimental import jitclass
import numba as nb
import numpy as np

dga_config_spec = [
    ('hdf5_file', nb.types.string),
    ('toml_file', nb.types.string),
    ('file_name', nb.types.string),
    ('kdim', nb.int64),
    ('nk', nb.int64),
    ('nq', nb.int64),
    ('n_kpoints', nb.int64),
    ('n_qpoints', nb.int64),
    ('n_qpoints_fullbz', nb.int64),
    ('k_grid', nb.float64[:,:]),
    ('q_grid', nb.float64[:,:]),
    ('q_grid_loc', nb.float64[:,:]),
    ('weights', nb.complex128[:]),
    ('irrbz', nb.boolean),
    ('niwf', nb.int64),
    ('n2iwb', nb.int64),
    ('n3iwf', nb.int64),
    ('n3iwb', nb.int64),
    ('n4iwf', nb.int64),
    ('n4iwb', nb.int64),
    ('max_iter', nb.int64),
    ('lambda_decay', nb.int64),
    ('lambda_type', nb.types.string),
    ('use_mpi', nb.boolean),
    ('beta', nb.float64),
    ('g0', nb.float64),
    ('w0', nb.float64),
    ('U', nb.float64),
    ('mu_imp', nb.float64),
    ('occ_imp', nb.float64),
    ('asymp', nb.types.string),
    ('nouter', nb.int64),
    # --- Handling np.ndarray ---
    # For a NumPy array, you need to specify:
    # 1. Element type (e.g., nb.complex128 for complex numbers, nb.float64 for floats)
    # 2. Number of dimensions (e.g., 1 for 1D, 2 for 2D)
    # 3. Memory layout (e.g., 'C' for C-contiguous, 'F' for Fortran-contiguous)
    # If the array can be None, Numba will typically infer an optional type,
    # but for explicit safety, you might see nb.optional(nb.complex128[:]) etc.
    # Numba arrays are often represented as a combination of type, dims, and layout.
    # Here, we'll assume g_imp is a 1D complex array (e.g., for Green's function data)
    # and use nb.types.Array to represent it.
    # If it's truly optional, assigning None initially works.
    ('g_imp', nb.complex128[:]), # Example: 1D array of complex128.
                               # Numba will infer that it can be None if assigned None in __init__
                               # or if you use nb.optional(nb.complex128[:])
    ('s_imp', nb.complex128[:]),
    ('chi_loc_w', nb.complex128[:,:]),
    ('p3ph', nb.complex128[:,:,:]),
    ('chi_ph', nb.complex128[:,:,:,:]),
    ('chi0_w', nb.complex128[:,:]),
    ('chi0_w_full', nb.complex128[:,:]),
    ('F_d_loc', nb.complex128[:,:,:]),
    ('F_m_loc', nb.complex128[:,:,:]),
    ('gamma_d', nb.complex128[:,:,:]),
    ('gamma_m', nb.complex128[:,:,:]),
    ('chi_d_loc', nb.complex128[:]),
    ('chi_m_loc', nb.complex128[:]),
    ('ts', nb.float64[:]),
]


@jitclass(dga_config_spec)
class DGA_Config:
    """
    Configuration settings for the DGammaA calculation, optimized with Numba.
    Here we should put all terms that do not change during the DGA calculation.
    """
    def __init__(self,
                hdf5_file: str,
                toml_file: str,
                # Make these mandatory by removing default = None
                g_imp: np.ndarray,
                s_imp: np.ndarray,
                chi_loc_w: np.ndarray,
                p3ph: np.ndarray,
                chi_ph: np.ndarray,
                ts:np.ndarray,
                # Other parameters (can still have defaults)
                file_name: str = "results",
                kdim: int = 2, nk: int = 4, nq: int = 4, irrbz: bool = True,
                niwf: int = 1000, n2iwb: int = 100, 
                n3iwf: int = 100, n3iwb: int = 100,
                n4iwf: int = 100, n4iwb: int = 100,
                asymp: str = 'bubble', nouter: int = 300,
                max_iter: int = 1, lambda_decay: int = 1, lambda_type: str = "Pauli",
                use_mpi: bool = True,
                beta: float = 1000.0, g0: float = 0.0, w0: float = 1.0, U: float = 0.0,
                mu_imp: float = 0.0, occ_imp: float = 0.0):

        self.hdf5_file = hdf5_file
        self.toml_file = toml_file
        self.file_name = file_name
        # Lattice parameters    
        self.kdim = kdim
        self.nk = nk
        self.nq = nq
        self.irrbz = irrbz
        # Frequencies
        self.niwf = niwf
        self.n2iwb = n2iwb
        self.n3iwf = n3iwf
        self.n3iwb = n3iwb
        self.n4iwf = n4iwf
        self.n4iwb = n4iwb
        # asymptotics
        self.asymp = asymp
        self.nouter = nouter
        # Self-Consistency and Lambda correction parameters
        self.max_iter = max_iter
        self.lambda_decay = lambda_decay
        self.lambda_type = lambda_type
        self.use_mpi = use_mpi
        # Physical parameters
        self.beta = beta
        self.g0 = g0
        self.w0 = w0
        self.U = U
        # DMFT Impurity
        self.mu_imp = mu_imp
        self.occ_imp = occ_imp
        self.g_imp = g_imp
        self.s_imp = s_imp
        self.chi_loc_w = chi_loc_w
        self.p3ph = p3ph
        self.chi_ph = chi_ph
        self.ts = ts

    def init_lattice(self):
        # get correct number of kpoints for q and k grid
        if self.irrbz:
            self.n_qpoints = np.int64(self.nq * (self.nq + 1) / 2)
            self.nk = 2 * self.nq - 2
            self.n_qpoints_fullbz = self.nk**self.kdim
        else:
            self.n_qpoints_fullbz = self.nq**self.kdim
            self.n_qpoints = self.n_qpoints_fullbz
        self.n_kpoints = self.nk**self.kdim


#Instance important to pass to function
DGA_ConfigType = DGA_Config.class_type.instance_type