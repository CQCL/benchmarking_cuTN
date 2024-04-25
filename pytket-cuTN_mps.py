import sys
from pathlib import Path
import json
import time as t

from mpi4py import MPI

from pytket import Circuit
from pytket.extensions.cutensornet.structured_state import SimulationAlgorithm, simulate, Config, CuTensorNetHandle

comm = MPI.COMM_WORLD

if sys.argv[1] != "OURS":
    raise ValueError("If calling pytket-cutensornet method use OURS as the first argument.")

trunc_mode = sys.argv[2]
param = sys.argv[3]

if trunc_mode == "chi":
    chi = int(param)
    truncation_fidelity = None
elif trunc_mode == "trunc_error":
    trunc_error = float(param)
    assert trunc_error >= 0 and trunc_error <= 1
    truncation_fidelity = 1 - trunc_error
    chi = None
else:
    raise ValueError("Use either `chi` or `trunc_error`.")

rank = comm.Get_rank()
n_procs = comm.Get_size()

directory = Path("selected_circs/")

with open("our_results.csv", "a") as data:
  for k, file_path in enumerate(directory.iterdir()):
    if k % n_procs != rank: continue  # Skip if this process is not responsible of simulating this circuit

    with open(file_path, "r") as f:
        circ = Circuit.from_dict(json.load(f))

    with CuTensorNetHandle(rank) as libhandle:
        cfg = Config(
            chi=chi,
            truncation_fidelity=truncation_fidelity,
            value_of_zero=0,
        )
        try:
            start_time = t.time()
            mps = simulate(libhandle, circ, SimulationAlgorithm.MPSxGate, cfg)
            duration = t.time() - start_time
            fidelity = mps.fidelity

            filename = str(file_path).split("/")[-1]
            entry = f"{filename},OURS,{trunc_mode},{param},{duration},{fidelity}\n"
            data.write(entry)
            data.flush()
            print(entry)
            sys.stdout.flush()

        except Exception as e:
            print(f"Failed! {filename}, {e}")
            data.write(f"{filename},OURS,{trunc_mode},{param},nan,nan\n")
            data.flush()
