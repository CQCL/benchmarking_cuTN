import os
import sys
import json
import time as t

from cupy.cuda.runtime import getDeviceCount
from mpi4py import MPI

import cirq
from pytket.extensions.cirq import cirq_to_tk
from pytket.extensions.cutensornet.mps import ContractionAlg, simulate, prepare_circuit, ConfigMPS, CuTensorNetHandle

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

print(f"Simulating {filename} using max virtual bond dimension of {chi}")

start_time = t.time()
with CuTensorNetHandle() as libhandle:
    cfg = ConfigMPS(chi=chi, loglevel=20)
    mps = simulate(libhandle, circ, ContractionAlg.MPSxGate, cfg)

print(f"Time to simulate: {t.time() - start_time} seconds")

print(f"Fidelity estimate (lower bound): {mps.fidelity}")
