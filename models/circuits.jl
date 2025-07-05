rx_layer(nbit::Int,g_list) = chain(nbit, [put(nbit, i=>rot(X,g_list[i])) for i=1:nbit])
rz_layer(nbit::Int,h_list) = chain(nbit, [put(nbit, i=>rot(Z,h_list[i])) for i=1:nbit])
rzz_layer(nbit::Int, phi_list) = chain(nbit, [put(nbit, (i,i%nbit+1)=>rot(kron(Z,Z), 1/2 * phi_list[i])) for i in 1:nbit-1])
ent_layer(nbit::Int, entanger=cnot) =  chain(nbit, entanger(nbit, i, j) for (i, j) in map(i->(i=>i%nbit+1), 1:nbit))
ry_layer(nbit::Int,g_list) = chain(nbit, [put(nbit, i=>rot(Y,g_list[i])) for i=1:nbit])
cz_layer(nbit::Int) = chain(nbit, cz(i,i%nbit+1) for i in 1:nbit)

function simplified_two_design_circuit(nbit::Int, d)

    circuit = chain(nbit, [put(nbit, i=>rot(Y,1)) for i=1:nbit])

    for _ in 1:d
        even_part = chain(nbit, cz(i, i%nbit+1) for i in 1:2:nbit-1)
        even_rotations = chain(nbit, [put(nbit, i=>rot(Y,1)) for i in 1:nbit-1])

        odd_part = chain(nbit, cz(i, i%nbit+1) for i in 2:2:nbit-1)
        odd_rotations = chain(nbit, [put(nbit, i=>rot(Y,1)) for i in 2:nbit])
        push!(circuit, even_part)
        push!(circuit, even_rotations)
        push!(circuit, odd_part)
        push!(circuit, odd_rotations)
    end

    circuit

end

function t_single_layer(nbit::Int)

    circuit = chain(nbit)

    even_rx1 = rx_layer(nbit, rand(nbit))
    even_rz = rz_layer(nbit, rand(nbit))
    even_rx2 = rx_layer(nbit, rand(nbit))
    even_cz = chain(nbit, cz(i, i%nbit+1) for i in 1:2:nbit-1)
        
    odd_rx1 = rx_layer(nbit, rand(nbit))
    odd_rz = rz_layer(nbit, rand(nbit))
    odd_rx2 = rx_layer(nbit, rand(nbit))   
    odd_cz = chain(nbit, cz(i, i%nbit+1) for i in 2:2:nbit-1)

    push!(circuit, even_rx1, even_rz, even_rx2, even_cz, odd_rx1, odd_rz, odd_rx2, odd_cz)

    return circuit
    
end

function t_circuit(nbit::Int, d::Int)
    circuit = chain(nbit)

    for _ in 1:d
        push!(circuit, t_single_layer(nbit))
    end
    circuit
end