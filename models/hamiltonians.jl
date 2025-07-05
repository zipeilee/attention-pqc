function transverse_ising_(nbit::Int, h::Number; periodic::Bool=true)
    ising_term = map(1:(periodic ? nbit : nbit-1)) do i
        repeat(nbit,Z,(i,i%nbit+1))
    end |> sum
    -ising_term - h*sum(map(i->put(nbit,i=>X), 1:nbit))  # 取负号
end

function cluster_ising_3(nbit::Int, h2)
    c_term = map(1:nbit-2) do i
        repeat(nbit, Z, i) * repeat(nbit, X, i+1) * repeat(nbit, Z, i+2)
    end |> sum


    h2_term = map(1:nbit-1) do i
        repeat(nbit, X, i, i+1) 
    end |> sum

    h1_term = map(1:nbit) do i
        repeat(nbit, X, i)
    end |> sum

    h1 = 1.0
    -c_term - h1 * h1_term - h2 * h2_term
end

function cluster_ising_2(nbit::Int, λ::Number)
    c_term = map(1:nbit) do i
        repeat(nbit, X, (i==1 ? nbit : i-1)) * repeat(nbit, Z, i) * repeat(nbit, X, i%nbit+1)
    end |> sum

    λ_term = map(1:nbit) do i
        repeat(nbit, Y, i, i%nbit+1)
    end |> sum

    -c_term + λ * λ_term
end