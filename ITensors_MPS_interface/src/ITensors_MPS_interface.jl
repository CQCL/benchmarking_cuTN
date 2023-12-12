module ITensors_MPS_interface

using ITensors

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
function simulate(n_qubits::Int64, circuit, chi::Int64)
    site_inds = siteinds("Qubit", n_qubits)
    gates::Vector{ITensor} = []

    for (name, qubits, params) in circuit
        if name == "Rz"
            append!(gates, [op("TKET_Rz", site_inds, 1+qubits[1]; α=params[1])])
        elseif name == "Rx"
            append!(gates, [op("TKET_Rx", site_inds, 1+qubits[1]; α=params[1])])
        elseif name == "ZZPhase"
            append!(gates, [op("TKET_ZZPhase", site_inds, 1+qubits[1], 1+qubits[2]; α=params[1])])
        elseif name == "XXPhase"
            append!(gates, [op("TKET_XXPhase", site_inds, 1+qubits[1], 1+qubits[2]; α=params[1])])
        elseif name == "SWAP"
            append!(gates, [op("SWAP", site_inds, 1+qubits[1], 1+qubits[2])])
        else
            error("KernelPkg error: Unrecognised gate.")
        end
    end

    # Simulate the circuit
    duration = @elapsed begin
        for g in gates
            ψ = apply(g, MPS(site_inds, "0"); maxdim=chi, cutoff=1e-16)
            normalize!(ψ)
        end
    end  # elapsed

    return duration
end

end  #module
