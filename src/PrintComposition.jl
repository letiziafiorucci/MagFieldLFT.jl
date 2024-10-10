"""
C: Coefficients of the state in the Slater determinant basis
U: Matrix whose columns are coefficients of basis states in the Slater determinant basis.
   This matrix must be unitary, i.e., the basis must be orthonormal!
labels: Unique identifiers for the basis states (e.g. quantum numbers)
thresh: Total percentage to which the printed contributions must at least sum up to (default: 0.98)
"""
function print_composition(C::Vector{T1}, U::Matrix{T2}, labels::Vector{String}, thresh::Number=0.98, io::IO=stdout) where {T1 <: Number, T2 <: Number}
    U_list = [U[:,i:i] for i in 1:size(U)[2]]
    print_composition(C, U_list, labels, thresh, io)
end

"""
Same as before, but now you don't pass a square matrix U with basis vectors, but a list of (in general rectangular)
matrices U. This is used if some labels are used for whole groups of basis vectors (e.g., if
we want to print the percentage of a certain total spin).
"""
function print_composition(C::Vector{T1}, U_list::Vector{Matrix{T2}}, labels::Vector{String}, thresh::Number=0.99, io::IO=stdout) where {T1 <: Number, T2 <: Number}
    percentages = [real(C'*U*U'*C) for U in U_list]
    p = sortperm(percentages, rev=true)
    percentages_sorted = percentages[p]
    labels_sorted = labels[p]
    total_percentage = 0.0
    i = 1
    while total_percentage < thresh
        @printf(io, "%6.2f%%  %s\n", percentages_sorted[i]*100, labels_sorted[i])
        total_percentage += percentages_sorted[i]
        i += 1
    end
end

"""
This function assumes that the eigenvalues are sorted (i.e., equal eigenvalues have neighboring indices)
"""
function group_eigenvalues(values::Vector{T}, thresh::Real=1.0e-8) where T<:Real
    indices = Vector{Vector{Int64}}(undef, 0)
    unique_values = Vector{T}(undef, 0)
    current_value = values[1]
    current_indices = [1]
    for i in 2:length(values)
        if abs(values[i] - current_value) < thresh
            push!(current_indices, i)
        else
            push!(indices, current_indices)
            push!(unique_values, current_value)
            current_indices = [i]
            current_value = values[i]
        end
    end
    push!(indices, current_indices)
    push!(unique_values, current_value)
    return unique_values, indices
end

function adapt_basis(C_list::Vector{Matrix{T1}}, labels_list::Vector{Vector{Float64}}, op::HermMat) where {T1<:Number}
    C_list_new = Vector{Matrix{ComplexF64}}(undef, 0)
    labels_list_new = Vector{Vector{Float64}}(undef, 0)
    for i in 1:length(C_list)
        C = C_list[i]
        op_C = Hermitian(C'*op*C)
        vals, vecs = eigen(op_C)
        unique_vals, indices = group_eigenvalues(vals)
        for j in 1:length(unique_vals)
            V = vecs[:, indices[j]]
            labels = deepcopy(labels_list[i])
            push!(labels, unique_vals[j])
            push!(C_list_new, C*V)
            push!(labels_list_new, labels)
        end
    end
    return C_list_new, labels_list_new
end

"""
Turn a value that might almost be integer or half-integer into one that is exactly integer or half-integer.
Type of returned value: Float64
"""
function exactspinQN(value::Real)
    return round(2*value+1e-10)/2     # add small number before rounding to always get 0.0 instead of -0.0
end

"""
Given an eigenvalue of a squared angular momentum operator J2, i.e. value = J(J+1),
return the corresponding total angular momentum quantum number J.
"""
function J2_to_J(value::Real)
    J = sqrt(value+0.25)-0.5
    return exactspinQN(J)
end

function prettylabels_Hnonrel_S2_Sz(labels_list::Vector{Vector{Float64}})
    return [@sprintf("E = %10.6f, S = %4.1f, M_S = %4.1f", l[1], J2_to_J(l[2]), exactspinQN(l[3])) for l in labels_list]
end

function adapt_basis_Hnonrel_S2_Sz(param::LFTParam)
    exc = calc_exclists(param.l,param.nel)
    Hnonrel = calc_H_nonrel(param, exc)
    Sx, Sy, Sz = calc_S(param.l, exc)
    S2 = Sx*Sx + Sy*Sy + Sz*Sz
    C_list1, labels_list1 = adapt_basis([Matrix{Float64}(1.0I, exc.Dim, exc.Dim)], [Vector{Float64}(undef, 0)], Hermitian(Hnonrel))
    C_list2, labels_list2 = adapt_basis(C_list1, labels_list1, Hermitian(S2))
    C_list3, labels_list3 = adapt_basis(C_list2, labels_list2, Hermitian(Sz))
    str_labels_list3 = prettylabels_Hnonrel_S2_Sz(labels_list3)
    return C_list3, str_labels_list3
end

term_sym = Dict(0 => "S",
                1 => "P",
                2 => "D",
                3 => "F",
                4 => "G",
                5 => "H",
                6 => "I",
                7 => "K",
                8 => "L",
                9 => "M",
                10 => "N",
                11 => "O",
                12 => "Q",
                13 => "R",
                14 => "T",
                15 => "U",
                16 => "V")

function spinQNlabel(S::Real)
    if abs(S-round(S)) < 1e-10
        label = "$(Int64(round(S)))"
    elseif abs(2S-round(2S)) < 1e-10
        numerator = Int64(round(2S))
        label = "$(numerator)/2"
    else
        println(S)
        error("Spin quantum number may only take integer or half-integer values!")
    end
    return label
end

"""
format: How are the states labeled. Default: "QN" (quantum numbers). Other option: "term_symbols".
"""
function prettylabels_L2_S2_J2_Jz(labels_list::Vector{Vector{Float64}}, format::String="QN")
    if format == "QN"
        return [@sprintf("(L, S, J, M_J) = (%4.1f,%4.1f,%4.1f,%4.1f)", J2_to_J(l[1]), J2_to_J(l[2]), J2_to_J(l[3]), exactspinQN(l[4])) for l in labels_list]
    elseif format == "term_symbols"
        return [@sprintf("Term: %d%s%s, M_J = %s", Int(round(2*J2_to_J(l[2])+1)), term_sym[Int(round(J2_to_J(l[1])))], spinQNlabel(J2_to_J(l[3])), spinQNlabel(l[4])) for l in labels_list]
    end
end

function adapt_basis_L2_S2_J2_Jz(param::LFTParam, format::String="QN")
    exc = calc_exclists(param.l,param.nel)
    Sx, Sy, Sz = calc_S(param.l, exc)
    Lx, Ly, Lz = calc_L(param.l, exc)
    Jx = Lx+Sx
    Jy = Ly+Sy
    Jz = Lz+Sz
    S2 = Sx*Sx + Sy*Sy + Sz*Sz
    L2 = Lx*Lx + Ly*Ly + Lz*Lz
    J2 = Jx*Jx + Jy*Jy + Jz*Jz
    C_list1, labels_list1 = adapt_basis([Matrix{Float64}(1.0I, exc.Dim, exc.Dim)], [Vector{Float64}(undef, 0)], Hermitian(L2))
    C_list2, labels_list2 = adapt_basis(C_list1, labels_list1, Hermitian(S2))
    C_list3, labels_list3 = adapt_basis(C_list2, labels_list2, Hermitian(J2))
    C_list4, labels_list4 = adapt_basis(C_list3, labels_list3, Hermitian(Jz))
    str_labels_list4 = prettylabels_L2_S2_J2_Jz(labels_list4, format)
    return C_list4, str_labels_list4
end