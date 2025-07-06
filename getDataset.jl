using Yao, CUDA, Yao.AD, Yao.EasyBuild, KrylovKit
using CSV, DataFrames
using Optimisers
using Printf
using Zygote
using Random
using Plots

include("vqe/circuits.jl")
include("vqe/hamiltonians.jl")

Random.seed!(42)


function getcluster(n, λ, circuit; steps=2000, lr=0.001, cuda=false)
    h = cluster_ising_2(n, λ)
    circuit = circuit
    opt = Optimisers.setup(Optimisers.ADAM(lr), parameters(circuit))
    creg = zero_state(n)

    if cuda
        creg = zero_state(n)
    end

   
    tolerance = 1e-6
    max_consecutive_steps = 50
    consecutive_count = 0
    previous_energy = Inf
    f = 0
    exact,_ = eigsolve(mat(h), 1, :SR, ishermitian=true)
    for i in 1:steps
        

        ps = parameters(circuit)
        _, grads = expect'(h, creg => circuit)
        Optimisers.update!(opt, ps, grads)
        dispatch!(circuit, ps)
        energy = expect(h, creg=>circuit)
        if abs(energy - previous_energy) < tolerance
            consecutive_count += 1
        else
            consecutive_count = 0
        end
        previous_energy = energy

        f = abs(exact[1] - energy)
        if consecutive_count >= max_consecutive_steps
            println("Converged at step $i")
            println("Energy = $energy, grad = $(norm(grads)), fidelity=$f")
            break
        end
        println("Epochs = $i ,Energy = $energy, grad = $(norm(grads)), f=$f")
    end
    return parameters(circuit)
    
end


N = 11
d = 11
circuit = t_circuit(N, d)
pms = ones(nparameters(circuit))
circuit = dispatch!(circuit, pms)

for λ=0.000:0.001:0.8
    r = getcluster(N, λ, circuit, steps=5000, lr=0.001, cuda=false)
    df = DataFrame(column1=r)
    formatted_λ = @sprintf("%.3f", λ)
    CSV.write("Datasets/cluster_2/$(N)_$(d)/$(formatted_λ).csv", df)
end