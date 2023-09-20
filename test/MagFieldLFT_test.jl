using Test, LinearAlgebra

using MagFieldLFT

function test_iscanonical1()
    orbstring = [1,2,5,5,6]    # same orbital occurs twice: not canonical
    return MagFieldLFT.iscanonical(orbstring) == false
end

function test_iscanonical2()
    orbstring = [1,2,5,6]  
    return MagFieldLFT.iscanonical(orbstring) == true
end

function test_iscanonical3()
    orbstring = [2,1,5,6]  
    return MagFieldLFT.iscanonical(orbstring) == false
end

function test_createSDs()
    ref = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
    return MagFieldLFT.create_SDs(3,2) == ref
end

function test_createSDs2()
    nel = 1
    norb = 2
    ref = [[1], [2], [3], [4]]
    return MagFieldLFT.create_SDs(nel, norb) == ref
end

function test_U_complex2real()
    ref = [1/sqrt(2) 0 0 0 -im/sqrt(2);
           0 -1/sqrt(2) 0 im/sqrt(2) 0;
           0 0 1 0 0;
           0 1/sqrt(2) 0 im/sqrt(2) 0;
           1/sqrt(2) 0 0 0 im/sqrt(2)]
    return MagFieldLFT.U_complex2real(2) ≈ ref
end

function test_calc_lops()
    l=2
    ref_lz = [2 0 0 0 0;
              0 1 0 0 0;
              0 0 0 0 0;
              0 0 0 -1 0;
              0 0 0 0 -2]
    ref_lplus = [0 2 0 0 0;
                 0 0 sqrt(6) 0 0;
                 0 0 0 sqrt(6) 0;
                 0 0 0 0 2;
                 0 0 0 0 0]
    ref_lminus = [0 0 0 0 0;
                  2 0 0 0 0;
                  0 sqrt(6) 0 0 0;
                  0 0 sqrt(6) 0 0;
                  0 0 0 2 0]
    lz, lplus, lminus = MagFieldLFT.calc_lops_complex(l)
    return lz ≈ ref_lz && lplus ≈ ref_lplus && lminus ≈ ref_lminus
end

function test_calcERIs_complex()
    F = Dict(0=>3.2, 2=>(1/7/7)*1.72, 4=>(1/21/21)*2.20)
    l = 2
    ERIs = MagFieldLFT.calcERIs_complex(l, F)
    return ERIs[2,1,4,5] ≈ (-2.12/9) && ERIs[3,2,5,5] ≈ 0.0 && ERIs[2,2,5,5] ≈ (195.92/63)
end

function test_calcERIs_real()
    F = Dict(0=>3.2, 2=>(1/7/7)*1.72, 4=>(1/21/21)*2.20)
    l = 2
    ERIs = MagFieldLFT.calcERIs_real(l, F)
    return ERIs[2,2,5,5] ≈ (195.92/63) && ERIs[1,4,2,5] ≈ (-0.64/21)
end

function test_ERIs_symmetries()
    F = Dict(0=>3.2, 2=>1.72, 4=>2.20)
    l = 2
    ERIs = MagFieldLFT.calcERIs_real(l, F)
    @assert abs(ERIs[1,2,4,5]) > 1e-4
    t1 = ERIs[1,2,4,5] ≈ ERIs[2,1,4,5]
    t2 = ERIs[1,2,4,5] ≈ ERIs[1,2,5,4]
    t3 = ERIs[1,2,4,5] ≈ ERIs[2,1,5,4]
    t4 = ERIs[1,2,4,5] ≈ ERIs[4,5,1,2]
    t5 = ERIs[1,2,4,5] ≈ ERIs[5,4,1,2]
    t6 = ERIs[1,2,4,5] ≈ ERIs[4,5,2,1]
    t7 = ERIs[1,2,4,5] ≈ ERIs[5,4,2,1]
    return t1 && t2 && t3 && t4 && t5 && t6 && t7
end

function test_spinorb2orbindex()
    return MagFieldLFT.spinorb2orbindex(2) == (1, 'β') && MagFieldLFT.spinorb2orbindex(5) == (3, 'α')
end

function test_occ_list()
    SD = [1,3,4,7]
    occ_alpha_list = MagFieldLFT.occ_list(SD, 'α')
    occ_beta_list = MagFieldLFT.occ_list(SD, 'β')
    return occ_alpha_list == [(1, 1),(2, 2),(4, 4)] && occ_beta_list == [(3,2)]
end

function test_orb2spinorb_and_back()
    test1 = MagFieldLFT.spinorb2orbindex(MagFieldLFT.orb2spinorbindex(3,'α')) == (3,'α')
    test2 = MagFieldLFT.orb2spinorbindex(MagFieldLFT.spinorb2orbindex(17)...) == 17
    return test1 && test2
end

function test_unocc_list()
    SD = [1,2,4,5,6,11]
    norb = 7
    unocc_alpha_list = MagFieldLFT.unocc_list(SD, norb, 'α')
    unocc_beta_list = MagFieldLFT.unocc_list(SD, norb, 'β')
    return unocc_alpha_list == [2,4,5,7] && unocc_beta_list == [4,5,6,7]
end

function test_SD2index()
    N = 3
    norb = 7
    M = 2norb
    SDs = MagFieldLFT.create_SDs(N,norb)
    test1 = MagFieldLFT.SD2index(SDs[51], M) == 51
    test2 = MagFieldLFT.SD2index(SDs[32], M) == 32
    return test1 && test2
end

function test_Z_summand()
    M = 14
    N = 3
    i = 2
    P = 3
    return MagFieldLFT.Z_summand(i,P,N,M) == 11
end

function test_calc_exc_equal()
    SD = [1,2]
    norb = 3
    sigma_p = 'α'
    sigma_q = 'α'
    exc = MagFieldLFT.calc_exc_occ2unocc(SD, norb, sigma_p, sigma_q)
    return exc == [(6,2,1,-1), (8,3,1,-1)]
end

function test_calc_exc_minus()
    SD = [3,4]
    norb = 3
    sigma_p = 'β'
    sigma_q = 'α'
    exc = MagFieldLFT.calc_exc_occ2unocc(SD, norb, sigma_p, sigma_q)
    return exc == [(7,1,2,1), (14,3,2,-1)]
end

function test_calc_exc_occ2self()
    SD = [1,2,4]
    norb = 5
    exc_alpha_self = MagFieldLFT.calc_exc_occ2self(SD, norb, 'α')
    exc_beta_self = MagFieldLFT.calc_exc_occ2self(SD, norb,'β')
    test_alpha = exc_alpha_self == [(2,1,1,1)]
    test_beta = exc_beta_self == [(2,1,1,1), (2,2,2,1)]
    return test_alpha && test_beta
end

function test_calc_exclists()
    l = 1
    N = 2
    Lalpha, Lbeta, Lplus, Lminus = MagFieldLFT.calc_exclists(l,N)
    test1 = Lalpha[1] == [(1,1,1,1), (6,2,1,-1), (8,3,1,-1)]
    test2 = Lminus[10] == [(7,1,2,1), (14,3,2,-1)]
    return test1 && test2
end

"""
For 2 electrons in a shell of p orbitals, there are the following terms with their energies:
E(3P) = F0 - 5*F2
E(1D) = F0 + F2
E(1S) = F0 + 10*F2
(see Griffith (the theory of transition-metal ions) Chapter 4.5)
"""
function test_calc_H_nonrel1()
    l=1
    N=2
    Lalpha, Lbeta, Lplus, Lminus = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    F = Dict(0 => 1.0, 2 => 2.0)
    H = MagFieldLFT.calc_H_nonrel(hLFT, F, Lalpha, Lbeta)
    E = eigvals(H)
    return E[1]≈-9 && E[9]≈-9 && E[10]≈3 && E[14]≈3 && E[15]≈21
end

"""
For 2 electrons in a shell of d orbitals, there are the following terms with their energies:
E(3F) = A - 8*B
E(1D) = A - 3*B + 2*C
E(3P) = A + 7*B
E(1G) = A + 4*B + 2*C
E(1S) = A + 14*B + 7*C
(see Griffith (the theory of transition-metal ions) Table 4.6)
"""
function test_calc_H_nonrel2()
    l=2
    N=2
    Lalpha, Lbeta, Lplus, Lminus = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    H = MagFieldLFT.calc_H_nonrel(hLFT, F, Lalpha, Lbeta)
    E = eigvals(H)
    return E[1]≈-15 && E[22]≈3 && E[27]≈15 && E[36]≈17 && E[45]≈57
end

function test_calc_H_fieldfree()
    l=2
    N=2
    Lalpha, Lbeta, Lplus, Lminus = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    zeta = 0.5
    H = MagFieldLFT.calc_H_fieldfree(hLFT, F, zeta, Lalpha, Lbeta, Lplus, Lminus)
    E = eigvals(H)
    return E[1]≈E[5] && E[6]≈E[12] && E[13]≈E[21] && !(E[1]≈E[6]) && !(E[6]≈E[13])
end

"""
This test checks for the total orbital angular momentum expectation value L(L+1)
for the different terms.
"""
function test_calc_L()
    l=2
    N=2
    Lalpha, Lbeta, Lplus, Lminus = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    H = MagFieldLFT.calc_H_nonrel(hLFT, F, Lalpha, Lbeta)
    C = eigvecs(H)
    Lx, Ly, Lz = MagFieldLFT.calc_L(l, Lalpha, Lbeta)
    L2 = Lx*Lx + Ly*Ly + Lz*Lz
    L2val = diag(C'*L2*C)
    return L2val[1]≈12 && L2val[22]≈6 && L2val[27]≈2 && L2val[36]≈20 && (L2val[45]+1)≈1
end

"""
This test checks for the spin orbital angular momentum expectation value S(S+1)
for the different terms.
"""
function test_calc_S()
    l=2
    N=2
    Lalpha, Lbeta, Lplus, Lminus = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    H = MagFieldLFT.calc_H_nonrel(hLFT, F, Lalpha, Lbeta)
    C = eigvecs(H)
    Sx, Sy, Sz = MagFieldLFT.calc_S(l, Lalpha, Lbeta, Lplus, Lminus)
    S2 = Sx*Sx + Sy*Sy + Sz*Sz
    S2val = diag(C'*S2*C)
    return S2val[1]≈2 && (S2val[22]+1)≈1 && S2val[27]≈2 && (S2val[36]+1)≈1 && (S2val[45]+1)≈1
end

"""
This test checks for the total angular momentum expectation value J(J+1)
for the lowest three spin-orbit-coupled terms (originating from 3F term).
"""
function test_total_J()
    l=2
    N=2
    Lalpha, Lbeta, Lplus, Lminus = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    zeta = 0.5
    H = MagFieldLFT.calc_H_fieldfree(hLFT, F, zeta, Lalpha, Lbeta, Lplus, Lminus)
    C = eigvecs(H)
    Lx, Ly, Lz = MagFieldLFT.calc_L(l, Lalpha, Lbeta)
    Sx, Sy, Sz = MagFieldLFT.calc_S(l, Lalpha, Lbeta, Lplus, Lminus)
    Jx = Lx+Sx
    Jy = Ly+Sy
    Jz = Lz+Sz
    J2 = Jx*Jx + Jy*Jy + Jz*Jz
    J2val = diag(C'*J2*C)
    return J2val[1]≈6 && J2val[6]≈12 && J2val[13]≈20
end

function test_read_AILFT_params_ORCA()
    nel, norb, hLFT, F, zeta = read_AILFT_params_ORCA("CrF63-.out", "CASSCF")
    l=(norb-1)÷2
    Lalpha, Lbeta, Lplus, Lminus = MagFieldLFT.calc_exclists(l,nel)
    H = MagFieldLFT.calc_H_nonrel(hLFT, F, Lalpha, Lbeta)
    E = eigvals(H)
    E = (E .- E[1])*27.211    # take ground state energy as reference and convert to eV
    println(round(E[5], digits=3))
    println(round(E[13], digits=3))
    println(round(E[17], digits=3))
    println(round(E[end], digits=3))
    test1 = round(E[5], digits=3) == 1.638    # first excited quartet state
    test2 = round(E[13], digits=3) == 1.645   # third excited quartet state
    test3 = round(E[17], digits=3) == 2.398   # lowest doublet state
    test4 = round(E[end], digits=3) == 9.930  # highest doublet state
    return test1 && test2 && test3 && test4
end

function test_Ercomplex_SOC()
    nel, norb, hLFT, F, zeta = read_AILFT_params_ORCA("ErCl63-.out", "CASSCF")
    l=(norb-1)÷2
    Lalpha, Lbeta, Lplus, Lminus = MagFieldLFT.calc_exclists(l,nel)
    H = MagFieldLFT.calc_H_fieldfree(hLFT, F, zeta, Lalpha, Lbeta, Lplus, Lminus)
    E = eigvals(H)
    E = (E .- E[1])*219474.63  # take ground state energy as reference and convert to cm-1
    test1 = round(real(E[17]), digits=3) ==   5862.094
    test2 = round(real(E[43]), digits=3) ==   12235.462
    test3 = round(real(E[364]), digits=3) ==  118616.544
    return test1 && test2 && test3
end

@testset "MagFieldLFT.jl" begin
    @test test_iscanonical1()
    @test test_iscanonical2()
    @test test_iscanonical3()
    @test test_createSDs()
    @test test_createSDs2()
    @test test_U_complex2real()
    @test test_calc_lops()
    @test test_calcERIs_complex()
    @test test_calcERIs_real()
    @test test_spinorb2orbindex()
    @test test_occ_list()
    @test test_orb2spinorb_and_back()
    @test test_unocc_list()
    @test test_SD2index()
    @test test_Z_summand()
    @test test_ERIs_symmetries()
    @test test_calc_exc_equal()
    @test test_calc_exc_minus()
    @test test_calc_exc_occ2self()
    @test test_calc_exclists()
    @test test_calc_H_nonrel1()
    @test test_calc_H_nonrel2()
    @test test_calc_H_fieldfree()
    @test test_calc_L()
    @test test_calc_S()
    @test test_total_J()
    @test_broken test_read_AILFT_params_ORCA()
    @test test_Ercomplex_SOC()
end