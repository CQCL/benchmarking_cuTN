import json
from pathlib import Path

import cirq
from pytket.circuit import OpType, Circuit, Qubit
from pytket.extensions.cirq import cirq_to_tk
from pytket.extensions.cutensornet.structured_state import prepare_circuit_mps

# Specify the directory path
directory_path = Path("Circuits")

# Iterate over all files in the directory
for i, file_path in enumerate(directory_path.iterdir()):
    filename = str(file_path).split("/")[-1]

    circ = cirq_to_tk(cirq.read_json("Circuits/"+filename))
    circ, _ = prepare_circuit_mps(circ)
    circ.rename_units({Qubit("node", i): Qubit(i) for i in range(circ.n_qubits)})

    # Add ZZPhase gates with angle 0 to avoid ITensorsGPU bug
    pre_circ = Circuit(circ.n_qubits)
    for q0, q1 in zip(circ.qubits[:-1], circ.qubits[1:]):
        pre_circ.ZZPhase(0.0, q0, q1)
    pre_circ.add_circuit(circ, circ.qubits)

    with open("tket_circuits/"+filename, "w") as f:
        json.dump(pre_circ.to_dict(), f, indent=2)

    print(f"{i+1}/480")
