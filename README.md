
# benchmarking_cuTN
Quick and dirty repository to benchmark pytket-cutensornet's MPS methods against other libraries. Different branches contain different experiments:

- `main`: compare pytket-cutensornet's MPS versus ITensors (on CPU and GPU) and NVIDIA's own MPS implementation (cuTensorNet). The conclusions of the experiment appear in this Confluence page.
- `old_itensors_vs_quimb_vs_pytketcutn`: compare pytket-cutensornet's MPS versus ITensors (on CPU) and Quimb. Since Quimb was shown to perform considerably worse than ITensors, and the latter appears in the experiment in `main` branch, I consider this branch deprecated and only keep it for tracking. The conclusions of the experiment appear in this [Confluence page](https://cqc.atlassian.net/wiki/spaces/TKET/pages/2791374886/Benchmarking+against+ITensors+and+Quimb).

## Requirements

- Python (+3.8)
- Julia (1.9)

- An NVIDIA GPU with compute capability +7.0
- cuQuantum Python

- pytket-cutensornet version 0.6.0
- ITensors.jl
- CUDA.jl

## Installation instructions

For *cuQuantum* and *pytket-cutensornet* follow the instructions in https://github.com/CQCL/pytket-cutensornet. Installing through pip should allow you to choose the appropriate version.

For *ITensors.jl*, enter the Julia REPL and press `]` to enter the package manager. Type `add ITensors` followed by `add CUDA` and, finally, `dev ITensors_MPS_interface`.

## Contents

The main files to run the experiment for each of the libraries are:
- `itensors_mps.jl` -> Runs using ITensors (both CPU and GPU backends are available) run as `julia <path_to_tket_circ.json> "GPU" "chi" <chi_value>`, or replace with `"GPU"` with `"CPU"`. It is possible to set `"trunc_error"` instead of `"chi"`. The folder `ITensors_MPS_interface` contains an auxiliar Julia package that this script uses.
- `cutn_mps.py` -> Runs using cuTensorNet's high-level API implementation of MPS. Run as `python cutn_mps.py <path_to_tket_circ.json> <chi_value>`.
- `pytket-cuTN_mps.py` -> Run using pytket-cutensornet. Run as `python pytket-cuTN_mps.py <path_to_tket_circ.json> "chi" <chi_value>`. It is possible to set `"trunc_error"` instead of `"chi"`.

### Circuits

The circuits used for the benchmarking can be found in the ZIP file `Circuits.zip`. Unzip and make sure the corresponding JSON files representing the circuits are all contained within a `Circuits/` folder at the root of the repository. These circuits are for Hamiltonian simulation on 2D lattices; all of them have 56 qubits and use the gateset `{Rz, Rx, ZZPhase, XXPhase}`.

### Results

The `Results/chi300` folder contains a `comparison_table.csv` file with the results for an experiment with `chi=300`, along with the scripts necessary to create this CSV from the output of the different main script files described above; each of the `results_*.csv` files correspond to the output of the corresponding main script. The script `plots.py` reads the CSV file and generates some relevant figures.

**NOTE**: the file `Results/chi300/results_itensors.csv` contains both the results for CPU and GPU. However, the GPU results come from using `ITensorGPU`, rather than `ITensors` on GPUs via `CUDA.jl`. The latter is the approach currently in this repository, while the former is a [soon-to-be-deprecated library](https://github.com/ITensor/ITensors.jl/issues/1396) that currently runs faster on GPUs than the latter. The code for `ITensorGPU` can be recovered by inspecting the commit history, but I do not recommend it, as it is very finicky to install and they intend to drop support for it. In theory, `ITensors` using `CUDA.jl` should eventually be running as fast (or faster) than `ITensorGPU`, but this is not the case at the time of writing.

The results show that the MPS algorithm in pytket-cutensornet runs faster than the same MPS algorithm implemented ITensors running on GPUs. Most of the time, pytket-cutensornet takes less than 60% of the runtime of ITensors. When comparing to NVIDIA's own MPS (in cuTensorNet), we find both libraries have similar performance.
