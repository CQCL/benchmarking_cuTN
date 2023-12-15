import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("results.csv")

# pytket-cutn vs ITensors scatter plot
filtered = df.loc[(df["itensors_time"] != "oom") & (df["itensors_time"] != "")]

plt.scatter(
  filtered["n_2q_gates"],
  100*filtered["pytket-cutn_time"].astype(float) / filtered["itensors_time"].astype(float),
  marker="x",
)
plt.title("Runtime pytket-cutn vs ITensors", fontsize=16)
plt.xlabel("Two-qubit gates (not inc. SWAPs)", fontsize=12)
plt.ylabel("pytket-cutn / ITensors (%)", fontsize=12)
plt.show()

# pytket-cutn vs Quimb scatter plot
filtered = df.loc[(df["quimb_time"] != "nan") & (df["quimb_time"] != "")]

plt.scatter(
  filtered["n_2q_gates"],
  100*filtered["pytket-cutn_time"].astype(float) / filtered["quimb_time"].astype(float),
  marker="x",
)
plt.title("Runtime pytket-cutn vs Quimb", fontsize=16)
plt.xlabel("Two-qubit gates (not inc. SWAPs)", fontsize=12)
plt.ylabel("pytket-cutn / Quimb (%)", fontsize=12)
plt.show()

# ITensors vs Quimb scatter plot
filtered = df.loc[(df["itensors_time"] != "oom") & (df["itensors_time"] != "") & (df["quimb_time"] != "nan") & (df["quimb_time"] != "")]

plt.scatter(
  filtered["n_2q_gates"],
  100*filtered["itensors_time"].astype(float) / filtered["quimb_time"].astype(float),
  marker="x",
)
plt.title("Runtime ITensors vs Quimb", fontsize=16)
plt.xlabel("Two-qubit gates (not inc. SWAPs)", fontsize=12)
plt.ylabel("ITensors / Quimb (%)", fontsize=12)
plt.show()

# Fidelity histogram
plt.hist(df["pytket-cutn_fidelity"])
plt.title("Fidelity histogram", fontsize=16)
plt.xlabel("Output state fidelity", fontsize=12)
plt.ylabel("Number of circuits", fontsize=12)
plt.show()

# Fidelity (> 0.1) scatter plot
filtered = df.loc[df["pytket-cutn_fidelity"] > 0.1]  # Remove fidelities that are essentially zero (these are already reported in prev figure)

plt.scatter(
  filtered["depth"],
  filtered["pytket-cutn_fidelity"].astype(float),
  marker="x",
)
plt.yticks(np.arange(0.1,1.1,0.1))
plt.title("Fidelity vs depth (for fid > 0.1)", fontsize=16)
plt.xlabel("Circuit depth", fontsize=12)
plt.ylabel("Output state fidelity", fontsize=12)
plt.show()
