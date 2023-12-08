from quimb.tensor import *

from pytket.circuit import OpType
import math
import time as t

import json
import cirq
from pytket.extensions.cirq import cirq_to_tk

from pytket.extensions.cutensornet.mps import prepare_circuit

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
n_gates = len(circ.get_commands())

print(f"Simulating {filename} using max virtual bond dimension of {chi}")

start_time = t.time()
# The args for gate_opts can be found at https://quimb.readthedocs.io/en/latest/autoapi/quimb/tensor/decomp/index.html#quimb.tensor.decomp.svd_truncated
mps = CircuitMPS(circ.n_qubits, gate_opts={"max_bond": chi, "renorm": 0})  # We choose not to renormalise, so that we can estimate the fidelity at the end fr>two_qubit_count = 0
for i, gate in enumerate(circ.get_commands()):
    #print(f"{round(100*i/n_gates)}%")
    if gate.op.type == OpType.Rx:
        mps.rx(math.pi*gate.op.params[0], gate.qubits[0].index[0])
    elif gate.op.type == OpType.ZZPhase:
        mps.rzz(math.pi*gate.op.params[0], gate.qubits[0].index[0], gate.qubits[1].index[0])
    elif gate.op.type == OpType.XXPhase:
        mps.rxx(math.pi*gate.op.params[0], gate.qubits[0].index[0], gate.qubits[1].index[0])
    elif gate.op.type == OpType.SWAP:
        mps.swap(gate.qubits[0].index[0], gate.qubits[1].index[0])
    else:
        raise Exception(f"Unknown gate {gate.op.type}")
print(f"Time to simulate: {t.time() - start_time} seconds")

fidelity = mps.psi.H @ mps.psi
print(f"Fidelity estimate (lower bound): {abs(fidelity)}")
