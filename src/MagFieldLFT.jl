module MagFieldLFT

using LinearAlgebra, Permutations, OutputParser

export read_AILFT_params_ORCA, LFTParam

const kB = 3.166811563e-6    # Boltzmann constant in Eh/K
const alpha = 0.0072973525693  # fine structure constant

struct LFTParam
    nel::Int64
    norb::Int64
    hLFT::Matrix{Float64}
    F::Dict{Int64, Float64}
    zeta::Float64
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

struct ExcitationLists
    alpha::Vector{Vector{NTuple{4, Int64}}}
    beta::Vector{Vector{NTuple{4, Int64}}}
    plus::Vector{Vector{NTuple{4, Int64}}}
    minus::Vector{Vector{NTuple{4, Int64}}}
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
    return ExcitationLists(L_alpha, L_beta, L_plus, L_minus)
end

function calc_singletop(int::Matrix{T}, exc::ExcitationLists) where T<:Number
    Dim = length(exc.alpha)
    H_single = convert(T, 0) * zeros(Dim, Dim)   # zero matrix with same type as int
    for J in 1:Dim
        for (Ii,pi,qi,gammai) in exc.alpha[J]
            H_single[Ii,J] += int[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in exc.beta[J]
            H_single[Ii,J] += int[pi,qi]*gammai
        end
    end
    return H_single
end

function calc_double_exc(ERIs::Array{Float64, 4}, exc::ExcitationLists)
    Dim = length(exc.alpha)
    norb = size(ERIs)[1]
    H_double = zeros(Dim, Dim)
    for K in 1:Dim
        X = zeros(Dim, norb, norb)
        for p in 1:norb, q in 1:norb
            for (Ii,pi,qi,gammai) in exc.alpha[K]
                X[Ii,p,q] += ERIs[p,q,qi,pi] * gammai
            end
            for (Ii,pi,qi,gammai) in exc.beta[K]
                X[Ii,p,q] += ERIs[p,q,qi,pi] * gammai
            end
        end
        for J in 1:Dim
            for (Ii,pi,qi,gammai) in exc.alpha[K]
                H_double[Ii, J] += 0.5*gammai*X[J,pi,qi]
            end
            for (Ii,pi,qi,gammai) in exc.beta[K]
                H_double[Ii, J] += 0.5*gammai*X[J,pi,qi]
            end
        end
    end
    return H_double
end

function calc_H_nonrel(param::LFTParam, exc::ExcitationLists)
    norb = size(param.hLFT)[1]
    l = (norb-1)÷2
    ERIs = calcERIs_real(l, param.F)
    h_mod = calc_hmod(param.hLFT, ERIs)
    H_single = calc_singletop(h_mod, exc)
    H_double = calc_double_exc(ERIs, exc)
    return H_single + H_double
end

"""
zeta: SOC parameter
l: angular momentum of partially filled shell
exc: Lists of excitations and coupling coefficients.
"""
function calc_SOC(zeta::Float64, l::Int, exc::ExcitationLists)
    Dim = length(exc.alpha)
    H_SOC = im*zeros(Dim, Dim)
    lz, lplus, lminus = calc_lops_real(l::Int)
    for J in 1:Dim
        for (Ii,pi,qi,gammai) in exc.minus[J]
            H_SOC[Ii,J] += 0.5*zeta*lplus[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in exc.plus[J]
            H_SOC[Ii,J] += 0.5*zeta*lminus[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in exc.alpha[J]
            H_SOC[Ii,J] += 0.5*zeta*lz[pi,qi]*gammai
        end
        for (Ii,pi,qi,gammai) in exc.beta[J]
            H_SOC[Ii,J] -= 0.5*zeta*lz[pi,qi]*gammai
        end
    end
    return H_SOC
end

function calc_H_fieldfree(param::LFTParam, exc::ExcitationLists)
    norb = size(param.hLFT)[1]
    l = (norb-1)÷2
    H_fieldfree = calc_H_nonrel(param, exc) + calc_SOC(param.zeta, l, exc)
    return Hermitian(H_fieldfree)
end

function calc_L(l::Int, exc::ExcitationLists)
    lz, lplus, lminus = calc_lops_real(l::Int)
    Lz = calc_singletop(lz, exc)
    Lplus = calc_singletop(lplus, exc)
    Lminus = calc_singletop(lminus, exc)
    Lx = (Lplus + Lminus)/2
    Ly = (Lplus - Lminus)/(2im)
    return Lx, Ly, Lz
end

function calc_S(l::Int, exc::ExcitationLists)
    norb = 2l+1
    delta = Matrix{Float64}(1.0I, norb, norb)
    Dim = length(exc.alpha)
    Sx = zeros(Dim, Dim)
    Sy = im*zeros(Dim, Dim)
    Sz = zeros(Dim, Dim)
    for J in 1:Dim
        for (Ii,pi,qi,gammai) in exc.plus[J]
            Sx[Ii,J] += delta[pi,qi]*gammai/2
            Sy[Ii,J] += delta[pi,qi]*gammai/(2im)
        end
        for (Ii,pi,qi,gammai) in exc.minus[J]
            Sx[Ii,J] += delta[pi,qi]*gammai/2
            Sy[Ii,J] -= delta[pi,qi]*gammai/(2im)
        end
        for (Ii,pi,qi,gammai) in exc.alpha[J]
            Sz[Ii,J] += delta[pi,qi]*gammai/2
        end
        for (Ii,pi,qi,gammai) in exc.beta[J]
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
        hLFT = Matrix{Float64}(undef, norb, norb)
        for row in 1:norb
            for col in 1:norb
                hLFT[row,col] = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], row, col)
            end
        end
        perm = [4,2,1,3,5]    # change order from 0,1,-1,2,-2 to 2,1,0,-1,-2 (=x2-y2,xz,z2,yz,xy) 
        hLFT = hLFT[perm, perm]
        F0 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 2, 4)
        F2 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 3, 2)
        F4 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 4, 2)
        F = Dict(0 => F0, 2 => F2/49, 4 => F4/441)
        zeta = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "ZETA_D"], 0, 2)/219474.63  # convert from cm-1 to Hartree
    end
    if norb == 7
        hLFT = Matrix{Float64}(undef, norb, norb)
        for row in 1:norb
            for col in 1:norb
                hLFT[row,col] = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Orbital"], row, col)
            end
        end
        perm = [6,4,2,1,3,5,7]    # change order from 0,1,-1,2,-2,3,-3 to 3,2,1,0,-1,-2,-3
        hLFT = hLFT[perm, perm]
        F0 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 2, 2)
        F2 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 3, 2)
        F4 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 4, 2)
        F6 = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "Slater-Condon"], 5, 2)
        F = Dict(0 => F0, 2 => F2/15/15, 4 => F4/33/33, 6 => (5/429)^2 * F6)
        zeta = parse_float(outfile, ["AILFT MATRIX ELEMENTS ($method)", "ZETA_F"], 0, 2)/219474.63  # convert from cm-1 to Hartree
    end
    return LFTParam(nel, norb, hLFT, F, zeta)
end

const HermMat = Hermitian{ComplexF64, Matrix{ComplexF64}}   # Complex hermitian matrix

function calc_H_magfield(H_fieldfree::HermMat, L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}}, B::Vector{Float64})
    H_magfield = H_fieldfree + 0.5*(L[1]*B[1] + L[2]*B[2] + L[3]*B[3]) + S[1]*B[1] + S[2]*B[2] + S[3]*B[3]
    return Hermitian(H_magfield)    # this is important for the eigensolver to yield orthogonal eigenvectors
end

function fieldfree_GS_energy(H_fieldfree::HermMat)
    energies = eigvals(H_fieldfree)
    @assert isa(energies, Vector{Float64})  # energies need to be real
    return energies[1]
end

function calc_free_energy(H_fieldfree::HermMat, L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}}, B0_mol::Vector{Float64}, T::Real)
    energies, states = calc_solutions_magfield(H_fieldfree, L, S, B0_mol)
    kB = 3.166811563e-6    # Boltzmann constant in Eh/K
    beta = 1/(kB*T)
    energies_exp = exp.(-beta*energies)
    Z = sum(energies_exp)   # canonical partition function
    return -log(Z)/beta
end

function calc_magneticmoment_operator(L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}})
    Mel = Vector{Matrix{ComplexF64}}(undef, 3)
    for i in 1:3
        Mel[i] = -0.5*L[i] - S[i]
    end
    return Mel
end

"""
H_fieldfree: Hamiltonian in the absence of a magnetic field
H: Hamiltonian whose eigenstates and eigenenergies we want to calculate (relative to fieldfree E0)
"""
function calc_solutions(H_fieldfree::HermMat, H::HermMat)
    E0 = fieldfree_GS_energy(H_fieldfree)
    solution = eigen(H)
    @assert isa(solution.values, Vector{Float64})  # energies need to be real
    energies = solution.values .- E0         # All energies relative to fieldfree GS energy
    states = solution.vectors
    return energies, states
end

calc_solutions(H_fieldfree::HermMat) = calc_solutions(H_fieldfree, H_fieldfree)

function calc_solutions_magfield(H_fieldfree::HermMat, L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}}, B0_mol::Vector{Float64})
    H_magfield = calc_H_magfield(H_fieldfree, L, S, B0_mol)
    return calc_solutions(H_fieldfree, H_magfield)
end

function calc_Zel(energies::Vector{Float64}, T::Real)
    beta = 1/(kB*T)
    energies_exp = exp.(-beta*energies)
    Z = sum(energies_exp)   # canonical partition function
end

function calc_average_magneticmoment(energies::Vector{Float64}, states::Matrix{ComplexF64}, Mel::Vector{Matrix{ComplexF64}}, T::Real)
    beta = 1/(kB*T)
    energies_exp = exp.(-beta*energies)
    Zel = sum(energies_exp)   # canonical partition function
    Mel_eigenbasis = [states'*Melcomp*states for Melcomp in Mel]
    Mel_avg = [sum(energies_exp .* diag(Mel_eigenbasis_comp))/Zel for Mel_eigenbasis_comp in Mel_eigenbasis]
    @assert norm(imag(Mel_avg)) < 1e-12    # energies need to be real
    return real(Mel_avg)
end

"""
N_polar: Number of grid points for polar angle
N_azimuthal: Number of grid points for azimuthal angle
"""
function spherical_product_grid(N_polar::Integer, N_azimuthal::Integer)
    dtheta = pi/N_polar
    dphi = 2pi/N_azimuthal
    grid = Vector{Tuple{Float64, Float64, Float64}}(undef, 0)
    for theta in (dtheta/2):dtheta:(pi-dtheta/2)
        for phi in (dphi/2):dphi:(2pi-dphi/2)
            weight = sin(theta)*dtheta*dphi
            push!(grid, (theta, phi, weight))
        end
    end
    return grid
end

"""
The function f is assumed to return a list of values (vector-valued).
f is a function of points on the unit sphere, parametrized via a polar angle theta and an azimuthal angle phi.
"""
function integrate_spherical(f::Function, grid::Vector{Tuple{Float64, Float64, Float64}})
    dim = length(f(0,0))    # number of components in the output
    integrals = zeros(dim)
    for (theta, phi, weight) in grid
        integrals += weight*f(theta, phi)
    end
    return integrals
end

function calc_dipole_matrix(R::Vector{T}) where T<:Real
    R_length = norm(R)
    idmat = Matrix(1.0I, 3, 3)
    return ((3*R*R')/(R_length^2) - idmat)/(R_length^3)
end

"""
Calculate magnetic field created by a magnetic dipole moment (everything in atomic units).
"""
function dipole_field(M::Vector{T1}, R::Vector{T2}) where {T1<:Real, T2<:Real}
    return alpha^2 * (calc_dipole_matrix(R)*M)
end

"""
B0: field strength (scalar quantity).
The B0 vector in the laboratory frame is (0,0,B0)
"""
function calc_B0_mol(B0::Real, chi::Real, theta::Real)
    Bx = -sin(theta)*cos(chi)*B0
    By = sin(theta)*sin(chi)*B0
    Bz = cos(theta)*B0
    return [Bx, By, Bz]
end

function calc_Bind_avg_lab_z(chi::Real, theta::Real, Bind_avg_mol::Vector{Float64})
    return -sin(theta)*cos(chi)*Bind_avg_mol[1] +
            sin(theta)*sin(chi)*Bind_avg_mol[2] +
            cos(theta)*         Bind_avg_mol[3]
end

"""
theta: Euler angle: Angle between z axis (molecular frame) and Z axis (lab frame)
chi: Euler angle describing rotations of the molecule around its molecular frame z axis
H_fieldfree: Hamiltonian in the absence of a magnetic field (Slater determinant basis)
L: Orbital angular momentum operators (Slater determinant basis)
S: Total spin operators (Slater determinant basis)
Mel: Electronic magnetic dipole moment operators (Slater determinant basis)
R: Vectors from the points at which we want to know the induced field (typically nuclear positions) to the paramagnetic center (atomic units = Bohr)
B0: Magnitude of the external magnetic field (atomic units)
T: Temperature (Kelvin)
"""
function calc_integrands(theta::Real, chi::Real, H_fieldfree::HermMat, L::NTuple{3, Matrix{ComplexF64}}, S::Tuple{Matrix{Float64}, Matrix{ComplexF64}, Matrix{Float64}}, Mel::Vector{Matrix{ComplexF64}}, R::Vector{Vector{Float64}}, B0::Real, T::Real)
    B0_mol = calc_B0_mol(B0, chi, theta)
    energies, states = MagFieldLFT.calc_solutions_magfield(H_fieldfree, L, S, B0_mol)
    Zel = calc_Zel(energies, T)
    Mel_avg_mol = calc_average_magneticmoment(energies, states, Mel, T)
    Bind_avg_mol = [dipole_field(Mel_avg_mol, R_i) for R_i in R]
    Bind_avg_lab_z = [calc_Bind_avg_lab_z(chi, theta, B_i) for B_i in Bind_avg_mol]
    return [Zel*Bind_avg_lab_z; Zel]
end

function calc_operators_SDbasis(param::LFTParam)
    l=(param.norb-1)÷2
    exc = MagFieldLFT.calc_exclists(l,param.nel)
    H_fieldfree = MagFieldLFT.calc_H_fieldfree(param, exc)
    L = MagFieldLFT.calc_L(l, exc)
    S = MagFieldLFT.calc_S(l, exc)
    Mel = MagFieldLFT.calc_magneticmoment_operator(L,S)
    return H_fieldfree, L, S, Mel
end

"""
R: Vectors from the points at which we want to know the induced field (typically nuclear positions) to the paramagnetic center (atomic units = Bohr)
B0: Magnitude of the external magnetic field (atomic units)
T: Temperature (Kelvin)
"""
function calc_Bind(param::LFTParam, R::Vector{Vector{Float64}}, B0::Real, T::Real, grid::Vector{Tuple{Float64, Float64, Float64}})
    H_fieldfree, L, S, Mel = calc_operators_SDbasis(param)
    integrands(theta, chi) = calc_integrands(theta, chi, H_fieldfree, L, S, Mel, R, B0, T)
    integrals = integrate_spherical(integrands , grid)
    numerators = integrals[1:(end-1)]
    D = integrals[end]   # denominator (normalization)
    return [N/D for N in numerators]
end

function estimate_shifts_finitefield(param::LFTParam, R::Vector{Vector{Float64}}, B0::Real, T::Real, grid::Vector{Tuple{Float64, Float64, Float64}})
    Bind_values = calc_Bind(param, R, B0, T, grid)
    return (Bind_values / B0) * 1e6    # convert to ppm
end

struct DegenerateSet
    E::Float64              # energy of the states belonging to the set
    states::Vector{Int64}   # indices of the states belonging to the set
end

function determine_degenerate_sets(energies::Vector{Float64})
    degenerate_sets = Vector{DegenerateSet}(undef, 0)
    current_energy = energies[1]
    current_states = [1]
    for i in 2:length(energies)
        if abs(energies[i]-current_energy) < 1e-10                  # threshold is arbitrarily chosen. Need to think more about this
            push!(current_states, i)
        else
            push!(degenerate_sets, DegenerateSet(current_energy, current_states))
            current_energy = energies[i]
            current_states = [i]
        end
    end
    return degenerate_sets
end

function calc_susceptibility_vanVleck(param::LFTParam, T::Real)
    beta = 1/(kB*T)
    H_fieldfree, L, S, Mel = calc_operators_SDbasis(param)
    energies, states = calc_solutions(H_fieldfree)
    Zel0 = calc_Zel(energies, T)
    Mel_eigenbasis = [states'*Melcomp*states for Melcomp in Mel]
    degenerate_sets = determine_degenerate_sets(energies)
    chi = im*zeros(3, 3)
    for n in degenerate_sets
        n_indices = n.states
        m_indices = [i for i in 1:length(energies) if !(i in n_indices)]
        Boltzmann_factor = exp(-beta*n.E)
        Mel_part1 = [Melcomp[n_indices, n_indices] for Melcomp in Mel_eigenbasis]
        Mel_part2 = [Melcomp[n_indices, m_indices] for Melcomp in Mel_eigenbasis]
        Mel_eigenbasis_mod = deepcopy(Mel_eigenbasis)
        for m in 1:length(energies)
            for i in 1:3
                Mel_eigenbasis_mod[i][:, m] /= (n.E - energies[m])
            end
        end
        Mel_part2_mod = [Melcomp[n_indices, m_indices] for Melcomp in Mel_eigenbasis_mod]
        for k in 1:3
            for l in k:3
                part1 = beta*tr(Mel_part1[k] * Mel_part1[l]')
                part2 = -tr(Mel_part2[k] * Mel_part2_mod[l]')
                chi[k, l] += Boltzmann_factor * (part1 + part2 + part2')
                if k != l
                    chi[l, k] += Boltzmann_factor * (part1 + part2 + part2')
                end
            end
        end
    end
    chi *= (4pi*alpha^2)/Zel0
    @assert norm(imag(chi)) < 1e-12
    return real(chi)
end

"""
Returns the chemical shifts in ppm calculated according to the Kurland-McGarvey equation (point-dipole approximation)
R: Vectors from the points at which we want to know the induced field (typically nuclear positions) to the paramagnetic center (atomic units = Bohr)
T: Temperature (Kelvin)
"""
function calc_shifts_KurlandMcGarvey(param::LFTParam, R::Vector{Vector{Float64}}, T::Real)
    chi = calc_susceptibility_vanVleck(param, T)
    shifts = Vector{Float64}(undef, 0)
    for Ri in R
        D = calc_dipole_matrix(Ri)
        sigma = -(1/(4pi)) * chi * D
        shift = -(1/3)*tr(sigma)
        push!(shifts, shift)
    end
    shifts *= 1e6    # convert to ppm
    return shifts
end



end