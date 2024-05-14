import pandas as pd
import matplotlib.pyplot as plt

# Comparing ours vs NVIDIA's implementation
df = pd.read_csv("comparison_table.csv")
df = df.dropna(subset=['time_ours', 'time_cutn']).drop(columns=["time_gpu", "time_cpu", "fidelity"])

gate_df = pd.read_csv("gate_count.csv")
df = pd.merge(df, gate_df, how="inner", left_on="filename", right_on="circuit").drop(columns=["circuit"])

plt.scatter(
  df["n_2q_gates"],
  100*df["time_ours"].astype(float) / df["time_cutn"].astype(float),
  marker="x",
)
plt.plot([100, 5000], [100, 100], linestyle='--', color="gray")

plt.title("Runtime pytket-cutn vs NVIDIA", fontsize=16)
plt.xlabel("Two-qubit gates (not inc. SWAPs)", fontsize=12)
plt.ylabel("pytket-cutn / NVIDIA (%)", fontsize=12)
plt.ylim([75,105])
plt.show()


# Comparing ours vs ITensorGPU
df = pd.read_csv("comparison_table.csv")
df = df.dropna(subset=['time_ours', 'time_gpu']).drop(columns=["time_cutn", "time_cpu", "fidelity"])

gate_df = pd.read_csv("gate_count.csv")
df = pd.merge(df, gate_df, how="inner", left_on="filename", right_on="circuit").drop(columns=["circuit"])

plt.scatter(
  df["n_2q_gates"],
  100*df["time_ours"].astype(float) / df["time_gpu"].astype(float),
  marker="x",
)
plt.plot([100, 3000], [100, 100], linestyle='--', color="gray")

plt.title("Runtime pytket-cutn vs ITensorGPU", fontsize=16)
plt.xlabel("Two-qubit gates (not inc. SWAPs)", fontsize=12)
plt.ylabel("pytket-cutn / ITensorGPU (%)", fontsize=12)
plt.show()



# Comparing ITensors on CPU vs ITensorGPU
df = pd.read_csv("comparison_table.csv")
df = df.dropna(subset=['time_cpu', 'time_gpu']).drop(columns=["time_cutn", "time_ours", "fidelity"])

gate_df = pd.read_csv("gate_count.csv")
df = pd.merge(df, gate_df, how="inner", left_on="filename", right_on="circuit").drop(columns=["circuit"])

plt.scatter(
  df["n_2q_gates"],
  100*df["time_gpu"].astype(float) / df["time_cpu"].astype(float),
  marker="x",
)
plt.plot([100, 2500], [100, 100], linestyle='--', color="gray")

plt.title("Runtime ITensors (GPU vs CPU)", fontsize=16)
plt.xlabel("Two-qubit gates (not inc. SWAPs)", fontsize=12)
plt.ylabel("ITensorGPU / ITensors (%)", fontsize=12)
plt.show()
