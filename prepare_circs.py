import json
from pathlib import Path

import cirq
from pytket.circuit import OpType
from pytket.extensions.cirq import cirq_to_tk
from pytket.extensions.cutensornet.structured_state import prepare_circuit_mps

# Specify the directory path
directory_path = Path("Circuits")

# Iterate over all files in the directory
for i, file_path in enumerate(directory_path.iterdir()):
    filename = str(file_path).split("/")[-1]

    circ = cirq_to_tk(cirq.read_json("Circuits/"+filename))
    circ, _ = prepare_circuit_mps(circ)

    with open("tket_circuits/"+filename, "w") as f:
        json.dump(circ.to_dict(), f, indent=2)

    print(f"{i+1}/480")
