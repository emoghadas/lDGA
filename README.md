# lDGA
"Ladder-DGammaA implementation for Hubbard-Holstein model"

### Developer Installation

_Below are instructions for an editable install of the package. The c++ extension uses the [nanobind](https://nanobind.readthedocs.io/en/latest/index.html) package._

1. Clone the repo
   ```sh
   git clone https://github.com/emoghadas/lDGA.git
   ```
3. Install nanobing and scikit-build-core package in your conda/virtual env
   ```sh
   python -m pip install nanobind scikit-build-core[pyproject]
   ```
   or
   ```sh
   conda install -c conda-forge nanobind scikit-build-core
   ```
4. In root directory run editable install
   ```sh
   pip install --no-build-isolation --config-settings=editable.rebuild=true -Cbuild-dir=build -ve.
   ``` 

<p align="right">(<a href="#readme-top">back to top</a>)</p>
