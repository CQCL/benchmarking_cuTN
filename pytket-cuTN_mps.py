import os
import sys
import json
import time as t

from mpi4py import MPI

import cirq
from pytket.extensions.cirq import cirq_to_tk
from pytket.extensions.cutensornet.mps import ContractionAlg, simulate, prepare_circuit, ConfigMPS, CuTensorNetHandle

comm = MPI.COMM_WORLD

chi = int(sys.argv[1])
rank = comm.Get_rank()
n_procs = comm.Get_size()

directory = os.fsencode("Circuits/")
data_name = f"Results/tmp/pytket-cutn_chi_{chi}.dat"

# Figure out which circuits have already been simulated
already_simulated = []
for r in range(n_procs):
    try:
        with open(f"{data_name}_{r}", "r") as f:
            lines = f.readlines()
            already_simulated += [l.split()[0] for l in lines]
    except FileNotFoundError:
        continue

# Simulate the rest
with open(f"{data_name}_{rank}", "a") as data:
  for k, file in enumerate(os.listdir(directory)):
    if k % n_procs != rank: continue  # Skip if this process is not responsible of simulating this circuit

    filename = os.fsdecode(file)
    if not filename.endswith(".json"):
        raise RuntimeError(f"File {filename} in the Circuits/ directory is not a json file.")

    if filename in already_simulated: continue  # Skip if circuit has already been simulated

    print(f"Simulating {filename}")
    sys.stdout.flush()

    circ = cirq_to_tk(cirq.read_json("Circuits/"+filename))
    circ, _ = prepare_circuit(circ)

    try:
      start_time = t.time()
      with CuTensorNetHandle(rank) as libhandle:
          cfg = ConfigMPS(chi=chi)
          mps = simulate(libhandle, circ, ContractionAlg.MPSxGate, cfg)
      duration = t.time() - start_time
      fidelity = mps.fidelity

      entry = f"{filename} {duration} {fidelity}\n"
      data.write(entry)
      data.flush()

    except Exception as e:
      print(f"Failed! {filename}, {e}")
      data.write(f"{filename} nan nan\n")
      data.flush()
