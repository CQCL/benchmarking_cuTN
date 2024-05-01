import pandas as pd

df_itensors = pd.read_csv("results_itensors.csv", names=["filename","backend","trunc_mode","parameter","time","fidelity"])
df_itensors = df_itensors.drop(columns=["trunc_mode", "parameter", "fidelity"])
df_itensors["filename"] = df_itensors["filename"].str.split("/").str[-1]
df_itensors_cpu = df_itensors.loc[df_itensors["backend"]=="CPU"].drop(columns=["backend"])
df_itensors_gpu = df_itensors.loc[df_itensors["backend"]=="GPU"].drop(columns=["backend"])

df_ours = pd.read_csv("results_ours.csv", names=["filename","backend","trunc_mode","parameter","time","fidelity"])
df_ours = df_ours.drop(columns=["trunc_mode", "parameter", "backend"])

df_cutn = pd.read_csv("results_cutn.csv", names=["filename","backend","trunc_mode","parameter","time","fidelity"])
df_cutn = df_cutn.drop(columns=["trunc_mode", "parameter", "backend", "fidelity"])

merged_itensors_df = pd.merge(df_itensors_cpu, df_itensors_gpu, on='filename', how="outer", suffixes=('_cpu', '_gpu'))
merged_cutn_df = pd.merge(df_cutn, df_ours, on='filename', how="outer", suffixes=('_cutn', '_ours'))

merged_df = pd.merge(merged_itensors_df, merged_cutn_df, on='filename', how="outer")

print(merged_df)
