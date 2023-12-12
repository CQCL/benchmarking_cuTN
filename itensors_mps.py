import os
import sys
import json
import time as t

from mpi4py import MPI

import cirq
from pytket.circuit import OpType
from pytket.extensions.cirq import cirq_to_tk
from pytket.extensions.cutensornet.mps import prepare_circuit

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import ITensors_MPS_interface


comm = MPI.COMM_WORLD

chi = int(sys.argv[1])
rank = comm.Get_rank()
n_procs = comm.Get_size()

directory = os.fsencode("Circuits/")

with open(f"Results/tmp/itensors_chi_{chi}.dat_{rank}", "a") as data:
  for k, file in enumerate(os.listdir(directory)):
    if k % n_procs != rank: continue  # Skip if this process is not responsible of simulating this circuit

    filename = os.fsdecode(file)
    if not filename.endswith(".json"):
        raise RuntimeError(f"File {filename} in the Circuits/ directory is not a json file.")

    print(f"Simulating {filename}")
    sys.stdout.flush()

    circ = cirq_to_tk(cirq.read_json("Circuits/"+filename))
    circ, _ = prepare_circuit(circ)

    # Convert the circuit to a list of gates, in a format that can be passed down to Julia
    gates = []
    for g in circ.get_commands():
        qubits = [q.index[0] for q in g.qubits]
        if g.op.type == OpType.Rx:
            gates.append(("Rx", qubits, g.op.params))
        elif g.op.type == OpType.ZZPhase:
            gates.append(("ZZPhase", qubits, g.op.params))
        elif g.op.type == OpType.XXPhase:
            gates.append(("XXPhase", qubits, g.op.params))
        elif g.op.type == OpType.SWAP:
            gates.append(("SWAP", qubits, []))
        else:
            raise Exception(f"Unknown gate {gate.op.type}")

    results = ITensors_MPS_interface.simulate(circ.n_qubits, gates, chi)
    duration = results[0]
    fidelity = results[1]

    entry = f"{filename} {duration} {fidelity}\n"
    data.write(entry)
    data.flush()