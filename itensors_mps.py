from pytket.circuit import OpType
import time as t

import json
import cirq
from pytket.extensions.cirq import cirq_to_tk

from pytket.extensions.cutensornet.mps import prepare_circuit

from julia.api import Julia
jl = Julia(compiled_modules=False)
from julia import ITensors_MPS_interface

selected_circs = [
  'TFIM_square_obc_Jz=1.0_hx=2.0_dt=0.5_n_trotter_steps=2_Lx=8_Ly=7.json',
  'TFIM_square_obc_Jz=1.0_hx=4.0_dt=0.1_n_trotter_steps=12_Lx=8_Ly=7.json',
  'XZ_honeycomb_PBC_J=1_dt=0.1_n_trotter_steps=6_Lx=4_Ly=7.json',
  'XZ_square_obc_J=1_dt=0.1_n_trotter_steps=12_Lx=8_Ly=7.json',
]

chi = 100
circ_id = 0

filename = selected_circs[circ_id]
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

print(f"Simulating {filename} using max virtual bond dimension of {chi}")
start_time = t.time()
fidelity = ITensors_MPS_interface.simulate(circ.n_qubits, gates, chi)
print(f"Time to simulate: {t.time() - start_time} seconds")

print(f"Fidelity estimate (lower bound): {abs(fidelity)}")
