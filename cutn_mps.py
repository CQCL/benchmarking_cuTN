# Code adapted from https://github.com/NVIDIA/cuQuantum/blob/main/python/samples/cutensornet/high_level/mps_sampling_example.py

import sys
import json
from pathlib import Path
from time import time

import cupy as cp
import numpy as np

from pytket import Circuit

import cuquantum
from cuquantum import cutensornet as cutn

dev = cp.cuda.Device()  # get current device

# Quantum state parameters
file_path = Path(sys.argv[1])
chi = int(sys.argv[2])

with open(file_path, "r") as f:
    circ = Circuit.from_dict(json.load(f))

#############
# cuTensorNet
#############

handle = cutn.create()
stream = cp.cuda.Stream()
data_type = cuquantum.cudaDataType.CUDA_C_64F
num_qubits = circ.n_qubits
qubits_dims = (2, ) * num_qubits # qubit size

# Allocate device memory for the final MPS state
max_extent = chi
mps_tensor_extents = []
mps_tensor_strides = []
mps_tensors = []
mps_tensor_ptrs = []
for i in range(num_qubits):
    if i == 0:
        extents = (2, max_extent)
    elif i == num_qubits - 1:
        extents = (max_extent, 2)
    else:
        extents = (max_extent, 2, max_extent)
    mps_tensor_extents.append(extents)
    tensor = cp.zeros(extents, dtype='complex128')
    mps_tensors.append(tensor)
    mps_tensor_ptrs.append(tensor.data.ptr)
    mps_tensor_strides.append([stride_in_bytes // tensor.itemsize for stride_in_bytes in tensor.strides])

free_mem = dev.mem_info[0]
# use half of the total free size
scratch_size = free_mem // 2
scratch_space = cp.cuda.alloc(scratch_size)
print(f"Allocated {scratch_size} bytes of scratch memory on GPU")

# Create the vacuum quantum state
quantum_state = cutn.create_state(handle, cutn.StatePurity.PURE, num_qubits, qubits_dims, data_type)
print("Created the initial quantum state")

# Construct the quantum circuit state with gate application.
# NOTE: this is *not* contracting the gates yet.
gate_list = []
for cmd in circ.get_commands():

    # Load the gate's unitary to the GPU memory
    gate_unitary = cmd.op.get_unitary().astype(dtype=np.complex128, copy=False, order='F')
    gate_tensor = cp.asarray(gate_unitary, dtype=np.complex128)

    qs = cmd.qubits
    if len(qs) == 1:
        q0 = circ.qubits.index(qs[0])
        # CuTensorNet requires you to keep all of the gates in memory... or make these
        # mutable.
        gate_list.append(gate_tensor.copy())
        # Add the gate
        tensor_id = cutn.state_apply_tensor_operator(
            handle=handle,
            tensor_network_state=quantum_state,
            num_state_modes=1,
            state_modes=(q0, ),
            tensor_data=gate_list[-1].data.ptr,
            tensor_mode_strides=0,
            immutable=1,
            adjoint=0,
            unitary=1
        )
    elif len(qs) == 2:
        q0 = circ.qubits.index(qs[0])
        q1 = circ.qubits.index(qs[1])
        # Reshape into a rank-4 tensor
        gate_tensor = cp.reshape(gate_tensor, (2, 2, 2, 2), order='F')

        # CuTensorNet requires you to keep all of the gates in memory... or make these
        # mutable.
        gate_list.append(gate_tensor.copy())
        # Add the gate
        tensor_id = cutn.state_apply_tensor_operator(
            handle=handle,
            tensor_network_state=quantum_state,
            num_state_modes=2,
            state_modes=(q0, q1),
            tensor_data=gate_list[-1].data.ptr,
            tensor_mode_strides=0,
            immutable=1,
            adjoint=0,
            unitary=1
        )
    else:
        raise Exception("Cannot apply an n-qubit gate with n>2")

print("Quantum gates applied")

# Specify the target MPS state
cutn.state_finalize_mps(handle, quantum_state, cutn.BoundaryCondition.OPEN, mps_tensor_extents, mps_tensor_strides)
print("Set the final MPS representation")

# Configure the MPS computation
svd_algorithm_dtype = cutn.state_get_attribute_dtype(cutn.StateAttribute.MPS_SVD_CONFIG_ALGO)
svd_algorithm = np.array(cutn.TensorSVDAlgo.GESVD, dtype=svd_algorithm_dtype)
cutn.state_configure(handle, quantum_state,
    cutn.StateAttribute.MPS_SVD_CONFIG_ALGO, svd_algorithm.ctypes.data, svd_algorithm.dtype.itemsize)

# Prepare the specified quantum circuit for MPS computation
work_desc = cutn.create_workspace_descriptor(handle)
cutn.state_prepare(handle, quantum_state, scratch_size, work_desc, stream.ptr)
print("Prepared the specified quantum circuit for MPS computation")

workspace_size_d = cutn.workspace_get_memory_size(handle,
    work_desc, cutn.WorksizePref.RECOMMENDED, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH)
if workspace_size_d <= scratch_size:
    cutn.workspace_set_memory(handle, work_desc, cutn.Memspace.DEVICE, cutn.WorkspaceKind.SCRATCH, scratch_space.ptr, workspace_size_d)
else:
    print("Error:Insufficient workspace size on Device")
    cutn.destroy_workspace_descriptor(work_desc)
    cutn.destroy_state(quantum_state)
    cutn.destroy(handle)
    del scratch
    print("Free resource and exit.")
    exit()
print("Set the workspace buffer for MPS computation")

# Compute the final MPS state
print("Computing the MPS...")
time_begin = time()
extents_out, strides_out = cutn.state_compute(handle, quantum_state, work_desc, mps_tensor_ptrs, stream.ptr)
duration = time() - time_begin
print(f"\tFinished! Took {duration} seconds")

# If a lower extent is found during runtime, the cupy.ndarray container must be adjusted to reflect the lower extent
for i, (extent_in, extent_out) in enumerate(zip(mps_tensor_extents, extents_out)):
    if extent_in != tuple(extent_out):
        stride_out = [s * mps_tensors[0].itemsize for s in strides_out[i]]
        mps_tensors[i] = cp.ndarray(extent_out, dtype=mps_tensors[i].dtype, memptr=mps_tensors[i].data, strides=stride_out)
print("Computed the final MPS representation")

cutn.destroy_workspace_descriptor(work_desc)
cutn.destroy_state(quantum_state)
cutn.destroy(handle)
del scratch_space
print("Free resource and exit.")

# Save results
filename = str(file_path).split("/")[-1]
entry = f"{filename},CUTN,chi,{chi},{duration},--\n"
print(entry)

with open("results_cutn.csv", "a") as data:
    data.write(entry)
