module MagFieldLFT

using LinearAlgebra, Permutations, OutputParser

export read_AILFT_params_ORCA

function iscanonical(orblist::Vector{T}) where T <: Int
    for i in 2:length(orblist)
        if orblist[i]<=orblist[i-1]
            return false
        end
    end
    return true
end

function create_SDs(nel::Int, norb::Int)
    nspinorb = 2*norb
    SDs = [[P] for P in 1:(nspinorb-nel+1)]
    if length(SDs[1])<nel
        construct_SDs_recursive!(SDs, nspinorb, nel)
    end
    return SDs
end

function construct_SDs_recursive!(SDs::Vector{Vector{Int64}}, nspinorb::Int, nel::Int)
    oldSDs = deepcopy(SDs)
    deleteat!(SDs, 1:length(SDs))
    for SD in oldSDs
        for P in (SD[end]+1):nspinorb
            newSD = deepcopy(SD)
            push!(newSD, P)
            push!(SDs, newSD)
        end
    end
    if length(SDs[1])<nel
        construct_SDs_recursive!(SDs, nspinorb, nel)
    end
    return nothing
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

c1_0 = Matrix(1.0I, 3, 3)
c1_2 = [-1 sqrt(3) -sqrt(6);
        -sqrt(3) 2 -sqrt(3);
        -sqrt(6) sqrt(3) -1]

c2_0 = Matrix(1.0I, 5, 5)
c2_2 = [-2 sqrt(6) -2 0 0;
        -sqrt(6) 1 1 -sqrt(6) 0;
        -2 -1 2 -1 -2;
        0 -sqrt(6) 1 1 -sqrt(6);
        0 0 -2 sqrt(6) -2]
c2_4 = [1 -sqrt(5) sqrt(15) -sqrt(35) sqrt(70);
        sqrt(5) -4 sqrt(30) -sqrt(40) sqrt(35);
        sqrt(15) -sqrt(30) 6 -sqrt(30) sqrt(15);
        sqrt(35) -sqrt(40) sqrt(30) -4 sqrt(5);
        sqrt(70) -sqrt(35) sqrt(15) -sqrt(5) 1]

c3_0 = Matrix(1.0I, 7, 7)
c3_2 = [-5 5 -sqrt(10) 0 0 0 0;
        -5 0 sqrt(15) -sqrt(20) 0 0 0;
        -sqrt(10) -sqrt(15) 3 sqrt(2) -sqrt(24) 0 0;
        0 -sqrt(20) -sqrt(2) 4 -sqrt(2) -sqrt(20) 0;
        0 0 -sqrt(24) sqrt(2) 3 -sqrt(15) -sqrt(10);
        0 0 0 -sqrt(20) sqrt(15) 0 -5;
        0 0 0 0 -sqrt(10) 5 -5]
c3_4 = [3 -sqrt(30) sqrt(54) -sqrt(63) sqrt(42) 0 0;
        sqrt(30) -7 4*sqrt(2) -sqrt(3) -sqrt(14) sqrt(70) 0;
        sqrt(54) -4*sqrt(2) 1 sqrt(15) -sqrt(40) sqrt(14) sqrt(42);
        sqrt(63) -sqrt(3) -sqrt(15) 6 -sqrt(15) -sqrt(3) sqrt(63);
        sqrt(42) sqrt(14) -sqrt(40) sqrt(15) 1 -4*sqrt(2) sqrt(54);
        0 sqrt(70) -sqrt(14) -sqrt(3) 4*sqrt(2) -7 sqrt(30);
        0 0 sqrt(42) -sqrt(63) sqrt(54) -sqrt(30) 3]
c3_6 = [-1 sqrt(7) -sqrt(28) sqrt(84) -sqrt(210) sqrt(462) -sqrt(924);
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
c_matrices = Dict((1,0) => c1_0,
                  (1,2) => c1_2,
                  (2,0) => c2_0,
                  (2,2) => c2_2,
                  (2,4) => c2_4,
                  (3,0) => c3_0,
                  (3,2) => c3_2,
                  (3,4) => c3_4,
                  (3,6) => c3_6)

"""
Convert Racah parameters for a shell of d orbitals
into Slater--Condon parameters.
"""
function Racah2F(A::Real, B::Real, C::Real)
    return Dict(0 => A+(7/5)*C, 2 => B+(1/7)*C, 4 => C/35)
end

"""
Calculate lz operator in basis of complex atomic orbitals.
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

"""
Returns lz, lplus, lminus in basis of complex atomic orbitals of angular momentum l.
"""
function calc_lops_complex(l::Int)
    return calc_lz(l), calc_lplusminus(l,+1), calc_lplusminus(l,-1)
end

function calc_lops_real(l::Int)
    lz, lplus, lminus = calc_lops_complex(l)
    U = U_complex2real(l)
    return U'*lz*U, U'*lplus*U, U'*lminus*U
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
            ERIs_real[t,u,v,w] += ERIs_complex[p,q,r,s] * U[p,t]' * U[q,u] * U[r,v]' * U[s,w]
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

function orb2spinorbindex(p::Int, sigma::Char)
    @assert sigma == 'α' || sigma == 'β'
    if sigma == 'α'
        return 2p-1
    else
        return 2p
    end
end

"""
For a given Slater determinant (Vector of spin orbital indices in canonical order),
this function returns a list of either the alpha or the beta orbitals, depending on the
chosen channel.
For each spin orbital that belongs to the requested channel, a tuple of the position
in the Slater determinant (electron index) and the corresponding orbital index is returned.
"""
function occ_list(SD::Vector{Int64}, channel::Char)
    @assert channel == 'α' || channel == 'β'
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
SD: List of spin orbital indices in canonical order
norb: number of spatial orbitals
channel: 'α' or 'β'
"""
function unocc_list(SD::Vector{Int64}, norb::Int, channel::Char)
    @assert channel == 'α' || channel == 'β'
    list = Array{Int64}(undef,0)
    for p in 1:norb
        P = orb2spinorbindex(p,channel)
        if !(P in SD)
            push!(list, p)
        end
    end
    return list
end

function Z_summand(i::Int64, P::Int64, N::Int64, M::Int64)
    @assert i<=N
    if i==N
        return P-N
    else
        value = 0
        for m in (M-P+1):(M-i)
            value += binomial(m, N-i) - binomial(m-1, N-i-1)
        end
        return value
    end
end

"""
SD: List of spin orbital indices
M:  Total number of spin orbitals
"""
function SD2index(SD::Vector{Int64}, M::Int64)
    N = length(SD)
    I = 1
    for i in 1:N
        I += Z_summand(i, SD[i], N, M)
    end
    return I
end

function calc_exc_occ2unocc(SD::Vector{Int64}, norb::Int, sigma_p::Char, sigma_q::Char)
    @assert sigma_p == 'α' || sigma_p == 'β'
    @assert sigma_q == 'α' || sigma_q == 'β'
    occ = occ_list(SD, sigma_q)
    unocc = unocc_list(SD, norb, sigma_p)
    exc = Array{Tuple{Int64, Int64, Int64, Int64}}(undef,0)
    for (i,q) in occ
        for p in unocc
            P = orb2spinorbindex(p, sigma_p)   # spin orbital index we are exciting to
            SD_exc = deepcopy(SD)
            SD_exc[i] = P
            perm = sortperm(SD_exc)
            SD_exc_canonical = SD_exc[perm]
            perm = Permutation(perm)
            gamma = sign(perm)
            I = SD2index(SD_exc_canonical, 2norb)
            push!(exc, (I, p, q, gamma))
        end
    end
    return exc
end

function calc_exc_occ2self(SD::Vector{Int64}, norb::Int, channel::Char)
    @assert channel == 'α' || channel == 'β'
    occ = occ_list(SD, channel)
    exc = Array{Tuple{Int64, Int64, Int64, Int64}}(undef,0)
    K = SD2index(SD, 2norb)
    for (i,q) in occ
        push!(exc, (K, q, q, 1))
    end
    return exc
end

function calc_hmod(hLFT::Matrix{Float64}, ERIs::Array{Float64, 4})
    hmod = deepcopy(hLFT)
    dim = size(hmod)[1]
    for p in 1:dim
        for q in 1:dim
            for r in 1:dim
                hmod[p,q] -= 0.5* ERIs[p,r,r,q]
            end
        end
    end
    return hmod
end

"""
l: angular momentum of the partially filled shell (l=2 for d orbitals, l=3 for f orbitals).
N: number of electrons in partially filled shell
"""
function calc_exclists(l::Int, N::Int)
    norb = 2l+1
    SDs = create_SDs(N, norb)
    Dim = length(SDs)
    L_alpha = Vector{Vector{NTuple{4, Int64}}}(undef, 0)
    L_beta = Vector{Vector{NTuple{4, Int64}}}(undef, 0)
    L_plus = Vector{Vector{NTuple{4, Int64}}}(undef, 0)
    L_minus = Vector{Vector{NTuple{4, Int64}}}(undef, 0)
    for K in 1:Dim
        exc_alpha_diff = calc_exc_occ2unocc(SDs[K], norb, 'α', 'α')
        exc_beta_diff = calc_exc_occ2unocc(SDs[K], norb, 'β', 'β')
        exc_plus = calc_exc_occ2unocc(SDs[K], norb, 'α', 'β')
        exc_minus = calc_exc_occ2unocc(SDs[K], norb, 'β', 'α')
        exc_alpha_same = calc_exc_occ2self(SDs[K], norb, 'α')
        exc_beta_same = calc_exc_occ2self(SDs[K], norb, 'β')
        push!(L_alpha, [exc_alpha_same; exc_alpha_diff])
        push!(L_beta, [exc_beta_same; exc_beta_diff])
        push!(L_plus, exc_plus)
        push!(L_minus, exc_minus)
    end
    return L_alpha, L_beta, L_plus, L_minus
end

function calc_singletop(int::Matrix{T}, L_alpha::Vector{Vector{NTuple{4, Int64}}}, L_beta::Vector{Vector{NTuple{4, Int64}}}) where T<:Number
    Dim = length(L_alpha)
    H_single = convert(T, 0) * zeros(Dim, Dim)   # zero matrix with same type as int
    for J in 1:Dim
        for (Ii,pi,qi,gammai) in L_alpha[J]
            H_single[Ii,J] += int[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in L_beta[J]
            H_single[Ii,J] += int[pi,qi]*gammai
        end
    end
    return H_single
end

function calc_double_exc(ERIs::Array{Float64, 4}, L_alpha::Vector{Vector{NTuple{4, Int64}}}, L_beta::Vector{Vector{NTuple{4, Int64}}})
    Dim = length(L_alpha)
    norb = size(ERIs)[1]
    H_double = zeros(Dim, Dim)
    for K in 1:Dim
        X = zeros(Dim, norb, norb)
        for p in 1:norb, q in 1:norb
            for (Ii,pi,qi,gammai) in L_alpha[K]
                X[Ii,p,q] += ERIs[p,q,qi,pi] * gammai
            end
            for (Ii,pi,qi,gammai) in L_beta[K]
                X[Ii,p,q] += ERIs[p,q,qi,pi] * gammai
            end
        end
        for J in 1:Dim
            for (Ii,pi,qi,gammai) in L_alpha[K]
                H_double[Ii, J] += 0.5*gammai*X[J,pi,qi]
            end
            for (Ii,pi,qi,gammai) in L_beta[K]
                H_double[Ii, J] += 0.5*gammai*X[J,pi,qi]
            end
        end
    end
    return H_double
end

function calc_H_nonrel(hLFT::Matrix{Float64}, F::Dict{Int64, Float64}, L_alpha::Vector{Vector{NTuple{4, Int64}}}, L_beta::Vector{Vector{NTuple{4, Int64}}})
    norb = size(hLFT)[1]
    l = (norb-1)÷2
    ERIs = calcERIs_real(l, F)
    h_mod = calc_hmod(hLFT, ERIs)
    H_single = calc_singletop(h_mod, L_alpha, L_beta)
    H_double = calc_double_exc(ERIs, L_alpha, L_beta)
    return H_single + H_double
end

"""
zeta: SOC parameter
l: angular momentum of partially filled shell
L_alpha, L_beta, L_plus, L_minus: Lists of excitations and coupling coefficients.
"""
function calc_SOC(zeta::Float64, l::Int, L_alpha::Vector{Vector{NTuple{4, Int64}}}, L_beta::Vector{Vector{NTuple{4, Int64}}}, L_plus::Vector{Vector{NTuple{4, Int64}}}, L_minus::Vector{Vector{NTuple{4, Int64}}})
    Dim = length(L_alpha)
    H_SOC = im*zeros(Dim, Dim)
    lz, lplus, lminus = calc_lops_real(l::Int)
    for J in 1:Dim
        for (Ii,pi,qi,gammai) in L_minus[J]
            H_SOC[Ii,J] += 0.5*zeta*lplus[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in L_plus[J]
            H_SOC[Ii,J] += 0.5*zeta*lminus[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in L_alpha[J]
            H_SOC[Ii,J] += 0.5*zeta*lz[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in L_beta[J]
            H_SOC[Ii,J] -= 0.5*zeta*lz[pi,qi]*gammai
        end
    end
    return H_SOC
end

function calc_H_fieldfree(hLFT::Matrix{Float64}, F::Dict{Int64, Float64}, zeta::Float64, L_alpha::Vector{Vector{NTuple{4, Int64}}}, L_beta::Vector{Vector{NTuple{4, Int64}}}, L_plus::Vector{Vector{NTuple{4, Int64}}}, L_minus::Vector{Vector{NTuple{4, Int64}}})
    norb = size(hLFT)[1]
    l = (norb-1)÷2
    return calc_H_nonrel(hLFT, F, L_alpha, L_beta) + calc_SOC(zeta, l, L_alpha, L_beta, L_plus, L_minus)
end

function calc_L(l::Int, L_alpha::Vector{Vector{NTuple{4, Int64}}}, L_beta::Vector{Vector{NTuple{4, Int64}}})
    lz, lplus, lminus = calc_lops_real(l::Int)
    Lz = calc_singletop(lz, L_alpha, L_beta)
    Lplus = calc_singletop(lplus, L_alpha, L_beta)
    Lminus = calc_singletop(lminus, L_alpha, L_beta)
    Lx = (Lplus + Lminus)/2
    Ly = (Lplus - Lminus)/(2im)
    return Lx, Ly, Lz
end

function calc_S(l::Int, L_alpha::Vector{Vector{NTuple{4, Int64}}}, L_beta::Vector{Vector{NTuple{4, Int64}}}, L_plus::Vector{Vector{NTuple{4, Int64}}}, L_minus::Vector{Vector{NTuple{4, Int64}}})
    norb = 2l+1
    delta = Matrix{Float64}(1.0I, norb, norb)
    Dim = length(L_alpha)
    Sx = zeros(Dim, Dim)
    Sy = im*zeros(Dim, Dim)
    Sz = zeros(Dim, Dim)
    for J in 1:Dim
        for (Ii,pi,qi,gammai) in L_plus[J]
            Sx[Ii,J] += delta[pi,qi]*gammai/2
            Sy[Ii,J] += delta[pi,qi]*gammai/(2im)
        end
        for (Ii,pi,qi,gammai) in L_minus[J]
            Sx[Ii,J] += delta[pi,qi]*gammai/2
            Sy[Ii,J] -= delta[pi,qi]*gammai/(2im)
        end
        for (Ii,pi,qi,gammai) in L_alpha[J]
            Sz[Ii,J] += delta[pi,qi]*gammai/2
        end
        for (Ii,pi,qi,gammai) in L_beta[J]
            Sz[Ii,J] -= delta[pi,qi]*gammai/2
        end
    end
    return Sx, Sy, Sz
end

"""
Read parameters (in atomic units).
method: e.g. "CASSCF" or "NEVPT2"
TO DO: extend for f elements and for SOC parameter.
"""
function read_AILFT_params_ORCA(outfile::String, method::String)
    nel = parse_int(outfile, ["nel"], 0, 3)
    norb = parse_int(outfile, ["norb"], 0, 3)
    if norb == 5
        h_z2_z2     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 1, 1)
        h_xz_z2     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 1, 2)
        h_z2_yz     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 1, 3)
        h_x2y2_z2   = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 1, 4)
        h_z2_xy     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 1, 5)
        h_xz_xz     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 2, 2)
        h_xz_yz     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 2, 3)
        h_x2y2_xz   = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 2, 4)
        h_xz_xy     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 2, 5)
        h_yz_yz     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 3, 3)
        h_x2y2_yz   = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 3, 4)
        h_yz_xy     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 3, 5)
        h_x2y2_x2y2 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 4, 4)
        h_x2y2_xy   = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 4, 5)
        h_xy_xy     = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], 5, 5)
        A           = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Racah"], 2, 5)
        B           = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Racah"], 3, 2)
        C           = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Racah"], 4, 2)
        zeta        = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "ZETA_D"], 0, 2)/219474.63  # convert from cm-1 to Hartree
    end
    hLFT = Matrix{Float64}(undef, norb, norb)
    hLFT[1,1] = h_x2y2_x2y2
    hLFT[1,2] = hLFT[2,1] = h_x2y2_xz
    hLFT[1,3] = hLFT[3,1] = h_x2y2_z2
    hLFT[1,4] = hLFT[4,1] = h_x2y2_yz
    hLFT[1,5] = hLFT[5,1] = h_x2y2_xy
    hLFT[2,2] = h_xz_xz
    hLFT[2,3] = hLFT[3,2] = h_xz_z2
    hLFT[2,4] = hLFT[4,2] = h_xz_yz
    hLFT[2,5] = hLFT[5,2] = h_xz_xy
    hLFT[3,3] = h_z2_z2
    hLFT[3,4] = hLFT[4,3] = h_z2_yz
    hLFT[3,5] = hLFT[5,3] = h_z2_xy
    hLFT[4,4] = h_yz_yz
    hLFT[4,5] = hLFT[5,4] = h_yz_xy
    hLFT[5,5] = h_xy_xy
    return nel, norb, hLFT, A, B, C, zeta
end



end