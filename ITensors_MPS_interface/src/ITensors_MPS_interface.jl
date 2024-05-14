module ITensors_MPS_interface

using JSON3
using ITensors
using CUDA

export simulate

# Gate definitions, since TKET's are slightly different
function ITensors.op(::OpName"TKET_Rz", t::SiteType"Qubit"; α::Number)
    θ = π*α/2
    return [
        exp(-im * θ) 0
        0 exp(im * θ)
    ]
end

function ITensors.op(::OpName"TKET_Rx", t::SiteType"Qubit"; α::Number)
    θ = π*α/2
    return [
        cos(θ) -im*sin(θ)
        -im*sin(θ) cos(θ)
    ]
end

function ITensors.op(::OpName"TKET_ZZPhase", t::SiteType"Qubit"; α::Number)
    θ = π*α/2
    return [
        exp(-im * θ) 0 0 0
        0 exp(im * θ) 0 0
        0 0 exp(im * θ) 0
        0 0 0 exp(-im * θ)
    ]
end

function ITensors.op(::OpName"TKET_XXPhase", t::SiteType"Qubit"; α::Number)
    θ = π*α/2
    return [
        cos(θ) 0 0 -im*sin(θ)
        0 cos(θ) -im*sin(θ) 0
        0 -im*sin(θ) cos(θ) 0
        -im*sin(θ) 0 0 cos(θ)
    ]
end

# Build and simulate the given circuit (as a list of gates)
function simulate(file_path::String, processor::String; chi=nothing, trunc_error=nothing)
    
    circ_json = JSON3.read(file_path)
    n_qubits = length(circ_json["qubits"])

    proc = nothing
    if processor == "CPU"
        proc = identity
    elseif processor == "GPU"
        proc = cu
    else
        throw("The argument `processor` must be either CPU or GPU.")
    end

    site_inds = siteinds("Qubit", n_qubits)
    gates::Vector{ITensor} = []

    for cmd in circ_json["commands"]
        gate_type = cmd["op"]["type"]
        q0 = 1 + cmd["args"][1][2][1]
        q1 = 0
        if length(cmd["args"]) == 2
            q1 = 1 + cmd["args"][2][2][1]
            @assert q0-q1==1 || q1-q0==1
        end

        if gate_type == "Rz"
            angle = parse(Float64, cmd["op"]["params"][1])
            append!(gates, [proc(ITensors.op("TKET_Rz", site_inds, q0; α=angle))])
        elseif gate_type == "Rx"
            angle = parse(Float64, cmd["op"]["params"][1])
            append!(gates, [proc(ITensors.op("TKET_Rx", site_inds, q0; α=angle))])
        elseif gate_type == "ZZPhase"
            angle = parse(Float64, cmd["op"]["params"][1])
            append!(gates, [proc(ITensors.op("TKET_ZZPhase", site_inds, q0, q1; α=angle))])
        elseif gate_type == "XXPhase"
            angle = parse(Float64, cmd["op"]["params"][1])
            append!(gates, [proc(ITensors.op("TKET_XXPhase", site_inds, q0, q1; α=angle))])
        elseif gate_type == "SWAP"
            append!(gates, [proc(ITensors.op("SWAP", site_inds, q0, q1))])
        else
            throw("Unrecognised gate.")
        end
    end

    # Simulate the circuit
    ψ = proc(MPS(site_inds, "0"))
    duration = 0
    if !isnothing(chi) && isnothing(trunc_error)
        duration = @elapsed begin
            ψ = apply(gates, ψ; maxdim=chi)
        end  # elapsed
    elseif !isnothing(trunc_error) && isnothing(chi)
        duration = @elapsed begin
            ψ = apply(gates, ψ; cutoff=trunc_error)
        end  # elapsed
    else
        throw("Please choose a value for either chi or trunc_error")
    end

    fidelity = -1
    if processor == "CPU"
        fidelity = norm(ψ)^2
    end

    return [duration, fidelity]
end

end  #module
