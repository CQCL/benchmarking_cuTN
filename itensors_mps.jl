using ITensors_MPS_interface

filename = ARGS[1]
compute_type = ARGS[2]
trunc_mode = ARGS[3]

# Warm up to reduce overhead from Julia JIT. These usually take around 200 seconds.
simulate(filename, compute_type; chi=8)
simulate(filename, compute_type; trunc_error=0.1)

# Simulate the requested circuit
if trunc_mode == "chi"
    duration, fidelity = simulate(filename, compute_type; chi=parse(Int64, ARGS[4]))
elseif trunc_mode == "trunc_error"
    duration, fidelity = simulate(filename, compute_type; trunc_error=parse(Float64, ARGS[4]))
else
    throw("Select either `chi` or `trunc_error`")
end

# Append results to CSV file
file_path = "results_itensors.csv"
file = open(file_path, "a")

# Data to append
new_data = [filename, compute_type, trunc_mode, ARGS[4], duration, fidelity]

# Write the new data to the CSV file
write(file, join(new_data, ",") * "\n")

# Close the file
close(file)
