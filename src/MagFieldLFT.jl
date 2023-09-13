module MagFieldLFT

using LinearAlgebra

function iscanonical(orblist::Vector{T}) where T <: Int
    for i in 2:length(orblist)
        if orblist[i]<=orblist[i-1]
            return false
        end
    end
    return true
end

function create_SDs(nel::Int, norb::Int)
    SDs = Vector{Vector{Int64}}()
    nspinorb = 2*norb
    spinorbrange = 1:nspinorb
    allranges = Tuple(repeat([spinorbrange], nel))
    for indices in CartesianIndices(allranges)
        orblist = reverse(collect(Tuple(indices)))
        if iscanonical(orblist)
            push!(SDs, orblist)
        end
    end
    return SDs
end

function U_complex2real(l::Int)
    dim = 2l+1
    U = im*zeros(dim,dim)
    mvalues = l:-1:-l
    for i in 1:dim
        m = mvalues[i]
        if m>0
            U[i,i] = (-1)^m / sqrt(2)
            U[dim+1-i,i] = 1/sqrt(2)
        elseif m==0
            U[i,i] = 1.0
        else
            U[i,i] = im/sqrt(2)
            U[dim+1-i,i] = -(-1)^m * im/sqrt(2)
        end
    end
    return U
end

c2_0 = Matrix(1.0I, 5,5)
c2_2 = (1/7)*[-2 sqrt(6) -2 0 0;
            -sqrt(6) 1 1 -sqrt(6) 0;
            -2 -1 2 -1 -2;
            0 -sqrt(6) 1 1 -sqrt(6);
            0 0 -2 sqrt(6) -2]
c2_4 = (1/21)*[1 -sqrt(5) sqrt(15) -sqrt(35) sqrt(70);
               sqrt(5) -4 sqrt(30) -sqrt(40) sqrt(35);
               sqrt(15) -sqrt(30) 6 -sqrt(30) sqrt(15);
               sqrt(35) -sqrt(40) sqrt(30) -4 sqrt(5);
               sqrt(70) -sqrt(35) sqrt(15) -sqrt(5) 1]

c3_0 = Matrix(1.0I, 7,7)
c3_2 = (1/15)*[-5 5 -sqrt(10) 0 0 0 0;
               -5 0 sqrt(15) -sqrt(20) 0 0 0;
               -sqrt(10) -sqrt(15) 3 sqrt(2) -sqrt(24) 0 0;
               0 -sqrt(20) -sqrt(2) 4 -sqrt(2) -sqrt(20) 0;
               0 0 -sqrt(24) sqrt(2) 3 -sqrt(15) -sqrt(10);
               0 0 0 -sqrt(20) sqrt(15) 0 -5;
               0 0 0 0 -sqrt(10) 5 -5]
c3_4 = (1/33)*[3 -sqrt(30) sqrt(54) -sqrt(63) sqrt(42) 0 0;
               sqrt(30) -7 4*sqrt(2) -sqrt(3) -sqrt(14) sqrt(70) 0;
               sqrt(54) -4*sqrt(2) 1 sqrt(15) -sqrt(40) sqrt(14) sqrt(42);
               sqrt(63) -sqrt(3) -sqrt(15) 6 -sqrt(15) -sqrt(3) sqrt(63);
               sqrt(42) sqrt(14) -sqrt(40) sqrt(15) 1 -4*sqrt(2) sqrt(54);
               0 sqrt(70) -sqrt(14) -sqrt(3) 4*sqrt(2) -7 sqrt(30);
               0 0 sqrt(42) -sqrt(63) sqrt(54) -sqrt(30) 3]
c3_6 = (5/429)*[-1 sqrt(7) -sqrt(28) sqrt(84) -sqrt(210) sqrt(462) -sqrt(924);
                -sqrt(7) 6 -sqrt(105) 4*sqrt(14) -sqrt(378) sqrt(504) -sqrt(462);
                -sqrt(28) sqrt(105) -15 5*sqrt(14) -sqrt(420) sqrt(378) -sqrt(210);
                -sqrt(84) 4*sqrt(14) -5*sqrt(14) 20 -5*sqrt(14) 4*sqrt(14) -sqrt(84);
                -sqrt(210) sqrt(378) -sqrt(420) 5*sqrt(14) -15 sqrt(105) -sqrt(28);
                -sqrt(462) sqrt(504) -sqrt(378) 4*sqrt(14) -sqrt(105) 6 -sqrt(7);
                -sqrt(924) sqrt(462) -sqrt(210) sqrt(84) -sqrt(28) sqrt(7) -1]

"""
Keys are tuples (l,k), where l is the angular momentum
quantum number (e.g. l=2 for d orbitals) and k the index
of the Slater--Condon parameters.
"""
c_matrices = Dict((2,0) => c2_0,
                  (2,2) => c2_2,
                  (2,4) => c2_4,
                  (3,0) => c3_0,
                  (3,2) => c3_2,
                  (3,4) => c3_4,
                  (3,6) => c3_6)

"""
Calculate lz operator in basis of complex spherically symmetric orbitals.
"""
calc_lz(l::Int) = diagm(l:-1:-l)

function calc_lplusminus(l::Int, sign::Int)
    @assert Int64(abs(sign)) == 1       # sign may only be +1 or -1
    dim = 2l+1
    mvalues = l:-1:-l
    op = zeros(dim,dim)
    for i_prime in 1:dim
        m_prime = mvalues[i_prime]
        for i in 1:dim
            m = mvalues[i]
            if m_prime == (m + sign)
                op[i_prime, i] = sqrt(l*(l+1) - m*(m+sign))
            end
        end
    end
    return op
end

function calc_lops(l::Int)
    return calc_lz(l), calc_lplusminus(l,+1), calc_lplusminus(l,-1)
end

"""
We do not make use of the symmetries of ERIs since their
number is anyway small in single-shell LFT with d or f orbitals.
This simplifies the implementation.
This function calculates ERIs of complex orbitals
"""
function calcERIs_complex(l::Int, F::Dict{Int64, Float64})
    dim = 2l+1
    m = l:-1:-l
    ERIs = zeros(dim, dim, dim, dim)
    for p in 1:dim, q in 1:dim, r in 1:dim, s in 1:dim
        if m[p]-m[q] == m[s]-m[r]
            for k in 0:2:2l
                ERIs[p,q,r,s] += c_matrices[(l,k)][p,q] * c_matrices[(l,k)][s,r] * F[k]
            end
        end
    end
    return ERIs
end

function calcERIs_real(l::Int, F::Dict{Int64, Float64})
    dim = 2l+1
    U = U_complex2real(l)
    # The following loop structure has O(8) complexity instead of O(5)
    # for the ERI transformation. But we can afford it for such a
    # small number of orbitals (5 or 7), and it makes the code simpler.
    ERIs_complex = calcERIs_complex(l, F)
    ERIs_real = im*zeros(dim, dim, dim, dim)
    for p in 1:dim, q in 1:dim, r in 1:dim, s in 1:dim
        for t in 1:dim, u in 1:dim, v in 1:dim, w in 1:dim
            ERIs_real[t,u,v,w] += ERIs_complex[p,q,r,s] * U[p,t] * U[q,u] * U[r,v] * U[s,w]
        end
    end
    @assert norm(imag(ERIs_real)) < 1e-12
    return real(ERIs_real)
end

function spinorb2orbindex(P::Int)
    p = (P+1) ÷ 2
    if (P+1)%2 == 0
        sigma = 'α'
    else
        sigma = 'β'
    end
    return p,sigma
end

"""
For a given Slater determinant (Vector of spin orbital indices in canonical order),
this function returns a list of either the alpha or the beta orbitals, depending on the
chosen channel.
For each spin orbital that belongs to the requested channel, a tuple of the position
in the Slater determinant (electron index) and the corresponding orbital index is returned.
"""
function occ_list(SD::Vector{Int64}, channel::Char)
    list = Array{Tuple{Int64,Int64}}(undef,0)
    for i in 1:length(SD)
        p,sigma = spinorb2orbindex(SD[i])
        if sigma == channel
            push!(list, (i,p))
        end
    end
    return list
end

"""
Outline of implementation:

function calc_H_nonrel(hLFT, F_k)
    ERIs = calc_ERIs(F_k)
    h_mod = calc_hmod(hLFT, ERIs)
    H_single = calc_singletop(h_mod)   # same function can also be used for angular momentum matrices
    H_double = calc_double_exc(ERIs)
    return H_single + H_double
end

"""


end