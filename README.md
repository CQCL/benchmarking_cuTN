
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
