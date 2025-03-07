from dataclasses import dataclass

@dataclass
class DGA_Config:
    """
    Configuration settings for the DGammaA calculation. 
    We should make sure that this can be also generated from a config.json file or sth similar 
    """
    hdf5_file: str  # Path to the DMFT HDF5 results file
    kdim: int = 2 # dimensionality of lattice
    nk: int = 30 # Number of k-points
    nq: int = 2 # Number of q-points
    niwf = 1000  # Number of positive fermionic frequencies for 1p-GF
    n4iwf: int = 100  # Number of positive fermionic frequencies for 2p-GF
    n4iwb: int = 100  # Number of positive bosonic frequencies for 2p-GF
    use_mpi: bool = True  # Flag for MPI parallelization
    dmft_dict = {} 