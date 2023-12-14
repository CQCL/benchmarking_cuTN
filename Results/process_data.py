import pandas as pd
import os

import cirq
from pytket.extensions.cirq import cirq_to_tk
from pytket.extensions.cutensornet.mps import prepare_circuit

from pytket.utils.stats import gate_counts
from pytket.circuit import OpType

chi = 300

succ_files = {"itensors": f"succ_itensors_chi_{chi}.dat", "quimb": f"succ_quimb_chi_{chi}.dat", "pytket-cutn": f"succ_pytket-cutn_chi_{chi}.dat"}


# Gather the properties of all circuits
circ_data = []
print("Counting gates on all circuits, this may take a while...")
for k, file in enumerate(os.listdir("../Circuits")):

    filename = os.fsdecode(file)
    if not filename.endswith(".json"):
        raise RuntimeError(f"File {filename} in the Circuits/ directory is not a json file.")

    circ = cirq_to_tk(cirq.read_json("../Circuits/"+filename))

    width = circ.n_qubits
    depth = circ.depth()

    #circ, _ = prepare_circuit(circ)
    counter = gate_counts(circ)

    circ_data.append([filename, width, depth, counter[OpType.ZZPhase]+counter[OpType.XXPhase]])
    print(f"\t {k+1}/480")

succ_df = pd.DataFrame(data=circ_data, columns=["circuit", "n_qubits", "depth", "n_2q_gates"])

# Gather the results of all simulations
for lib in ["itensors", "quimb", "pytket-cutn"]:

  with open(succ_files[lib], "r") as f:
    data = []

    for l in f.readlines():
      data.append(l.split())

    this_df = pd.DataFrame(data=data, columns=["circuit", lib+"_time", lib+"_fidelity"])
    succ_df = pd.merge(succ_df, this_df, how="outer", on="circuit")

succ_df.to_csv("results.csv")
