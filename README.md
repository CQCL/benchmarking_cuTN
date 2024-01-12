
# benchmarking_cuTN
Quick and dirty repository to benchmark pytket-cutensornet against Quimb and ITensors

## Requirements

- Python (+3.8)
- Julia (1.9)

- An NVIDIA GPU with compute capability +7.0
- cuQuantum Python

- pytket-cutensornet version 0.4.0
- Quimb
- ITensors.jl

## Installation instructions

For *cuQuantum* and *pytket-cutensornet* follow the instructions in https://github.com/CQCL/pytket-cutensornet. Installing through pip should allow you to choose the appropriate version.

For *Quimb*, follow https://quimb.readthedocs.io/en/latest/installation.html. Doing `pip install quimb` worked for me.

For *ITensors.jl*, enter the Julia REPL and press `]` to enter the package manager. Type `add ITensors`.

### Installing ITensors_MPS_interface.jl

Install pyCall:
- `pip install julia`
- In Python REPL, `import julia` followed by `julia.install()`

Test pyCall installation was successful by entering the Python REPL and typing:
 ```
from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import Base
Base.sind(60)
 ```

Move the contents of the folder `ITensors_MPS_interface` in this repository to `~/.julia/dev/ITensors_MPS_interface`. Then, open the Julia REPL and press `]` to enter the package manager. Locally install the package by typing:
 `dev ~/.julia/dev/ITensors_MPS_interface`.

## Contents

The three `*_mps.py` files at the root of this repository each run the circuits with a different library (pytket-cutensornet, ITensors and Quimb). The folder `ITensors_MPS_interface` contains the Julia package that interfaces between ITensors and `itensors_mps.py`.

### Circuits

The circuits used for the benchmarking can be found in the ZIP file `Circuits.zip`. Unzip and make sure the corresponding JSON files representing the circuits are all contained within a `Circuits/` folder at the root of the repository. These circuits are for Hamiltonian simulation on 2D lattices; all of them have 56 qubits and use the gateset `{Rz, Rx, ZZPhase, XXPhase}`.

### Results

The `Results` folder contains a CSV file with the results, along with the scripts necessary to create this CSV from the output of the different `*_mps.py` files at the root of this repository. The script `create_plots.py` reads the CSV file and generates some relevant figures; the directory `Results/Figures` contains the these.

![Comparison of pytket-cutensornet versus ITensors](https://github.com/CQCL/benchmarking_cuTN/blob/main/Results/Figures/cutn_ITensors.png)

The results show that the MPS algorithm in pytket-cutensornet runs faster than the same MPS algorithm implemented ITensors and Quimb. Most of the time, pytket-cutensornet takes 30% of the runtime of ITensors and 10% of the runtime of Quimb. This speed up is attributed to the use of cuTensorNet and NVIDIA GPUs.