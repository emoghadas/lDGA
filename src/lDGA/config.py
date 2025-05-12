from dataclasses import dataclass

@dataclass
class DGA_Config:
    """
    Configuration settings for the DGammaA calculation. 
    We should make sure that this can be also generated from a config.json file or sth similar 
    """
    hdf5_file: str  # Path to the DMFT HDF5 results file
    kdim: int = 2 # dimensionality of lattice
    nk: int = 12 # Number of k-points
    nq: int = 12 # Number of q-points
    niwf = 1000  # Number of positive fermionic frequencies for 1p-GF
    n4iwf: int = 100  # Number of positive fermionic frequencies for 2p-GF
    n4iwb: int = 100  # Number of positive bosonic frequencies for 2p-GF
    max_iter: int = 10
    lambda_decay: int = 1
    use_mpi: bool = True  # Flag for MPI parallelization
    dmft_dict = {} 
    g0: float = 0.0
    w0: float = 1.0
    lambda_type: str = "Pauli"
    file_name: str = "results"