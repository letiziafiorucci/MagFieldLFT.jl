using Test, LinearAlgebra

using MagFieldLFT

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
    exc = MagFieldLFT.calc_exclists(l,N)
    test1 = exc.alpha[1] == [(1,1,1,1), (6,2,1,-1), (8,3,1,-1)]
    test2 = exc.minus[10] == [(7,1,2,1), (14,3,2,-1)]
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
    exc = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    F = Dict(0 => 1.0, 2 => 2.0)
    param = LFTParam(N, norb, l, hLFT, F, 0.0)
    H = MagFieldLFT.calc_H_nonrel(param, exc)
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
    exc = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    param = LFTParam(N, norb, l, hLFT, F, 0.0)
    H = MagFieldLFT.calc_H_nonrel(param, exc)
    E = eigvals(H)
    return E[1]≈-15 && E[22]≈3 && E[27]≈15 && E[36]≈17 && E[45]≈57
end

function test_calc_H_fieldfree()
    l=2
    N=2
    exc = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    zeta = 0.5
    param = LFTParam(N, norb, l, hLFT, F, zeta)
    H = MagFieldLFT.calc_H_fieldfree(param, exc)
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
    exc = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    param = LFTParam(N, norb, l, hLFT, F, 0.0)
    H = MagFieldLFT.calc_H_nonrel(param, exc)
    C = eigvecs(H)
    Lx, Ly, Lz = MagFieldLFT.calc_L(l, exc)
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
    exc = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    param = LFTParam(N, norb, l, hLFT, F, 0.0)
    H = MagFieldLFT.calc_H_nonrel(param, exc)
    C = eigvecs(H)
    Sx, Sy, Sz = MagFieldLFT.calc_S(l, exc)
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
    exc = MagFieldLFT.calc_exclists(l,N)
    norb = 2l+1
    hLFT = zeros(norb,norb)
    A = 1
    B = 2
    C = 4
    F = MagFieldLFT.Racah2F(A,B,C)
    zeta = 0.5
    param = LFTParam(N, norb, l, hLFT, F, zeta)
    H = MagFieldLFT.calc_H_fieldfree(param, exc)
    C = eigvecs(H)
    Lx, Ly, Lz = MagFieldLFT.calc_L(l, exc)
    Sx, Sy, Sz = MagFieldLFT.calc_S(l, exc)
    Jx = Lx+Sx
    Jy = Ly+Sy
    Jz = Lz+Sz
    J2 = Jx*Jx + Jy*Jy + Jz*Jz
    J2val = diag(C'*J2*C)
    return J2val[1]≈6 && J2val[6]≈12 && J2val[13]≈20
end

function test_read_AILFT_params_ORCA()
    param = read_AILFT_params_ORCA("CrF63-.out", "CASSCF")
    exc = MagFieldLFT.calc_exclists(param)
    H = MagFieldLFT.calc_H_nonrel(param, exc)
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

# because of precision issues in the printed LFT parameters, the energies do not coincide exactly
# with what is printed in the ORCA output file!
# TO DO: change later.
function test_Ercomplex()
    param = read_AILFT_params_ORCA("ErCl63-.out", "CASSCF")
    exc = MagFieldLFT.calc_exclists(param)
    H = MagFieldLFT.calc_H_nonrel(param, exc)
    E = eigvals(H)
    E = (E .- E[1])*27.211  # take ground state energy as reference and convert to eV
    test1 = round(real(E[29]), digits=3) == 0.020
    test2 = round(real(E[53]), digits=3) == 2.194
    test3 = round(real(E[75]), digits=3) == 2.238
    return test1 && test2 && test3
end

function test_Ercomplex_SOC()
    param = read_AILFT_params_ORCA("ErCl63-.out", "CASSCF")
    exc = MagFieldLFT.calc_exclists(param)
    H = MagFieldLFT.calc_H_fieldfree(param, exc)
    E = eigvals(H)
    E = (E .- E[1])*219474.63  # take ground state energy as reference and convert to cm-1
    test1 = round(real(E[17]), digits=3) ==   5862.094
    test2 = round(real(E[43]), digits=3) ==   12235.462
    test3 = round(real(E[364]), digits=3) ==  118616.544
    return test1 && test2 && test3
end

function test_calc_free_energy()
    param = read_AILFT_params_ORCA("CrF63-.out", "CASSCF")
    exc = MagFieldLFT.calc_exclists(param)
    H_fieldfree = MagFieldLFT.calc_H_fieldfree(param, exc)
    S = MagFieldLFT.calc_S(param.l, exc)
    L = MagFieldLFT.calc_L(param.l, exc)
    B = [0.0, 0.0, 1.0e-5]
    T = 298.0
    F1 = MagFieldLFT.calc_free_energy(H_fieldfree, L, S, B, T)
    return F1 ≈ -0.0013082816934478216
end

function test_average_magnetic_moment()
    param = read_AILFT_params_ORCA("CrF63-.out", "CASSCF")
    exc = MagFieldLFT.calc_exclists(param)
    H_fieldfree = MagFieldLFT.calc_H_fieldfree(param, exc)
    S = MagFieldLFT.calc_S(param.l, exc)
    L = MagFieldLFT.calc_L(param.l, exc)
    B0_mol = [0.0, 0.0, 0.0]
    T = 298.0
    energies, states = MagFieldLFT.calc_solutions_magfield(H_fieldfree, L, S, B0_mol)
    Hderiv = MagFieldLFT.calc_Hderiv(L,S)
    Fderiv1 = MagFieldLFT.calc_F_deriv1(energies, states, Hderiv, T)
    Mel_avg = -Fderiv1
    return Mel_avg + [1.0, 1.0, 1.0] ≈ [1.0, 1.0, 1.0]    # magnetization is zero in absence of field
end

# At low temperature, magnetization should be that of the ground state (approximately MS=-3/2)
function test_average_magnetic_moment2()
    param = read_AILFT_params_ORCA("CrF63-.out", "CASSCF")
    exc = MagFieldLFT.calc_exclists(param)
    H_fieldfree = MagFieldLFT.calc_H_fieldfree(param, exc)
    S = MagFieldLFT.calc_S(param.l, exc)
    L = MagFieldLFT.calc_L(param.l, exc)
    B0_mol = [0.0, 0.0, 1.0e-4]
    T = 1.0
    energies, states = MagFieldLFT.calc_solutions_magfield(H_fieldfree, L, S, B0_mol)
    Mel = MagFieldLFT.calc_magneticmoment_operator(L,S)
    Mel_avg = MagFieldLFT.calc_average_magneticmoment(energies, states, Mel, T)
    return Mel_avg ≈ [-0.0003645214756898332, -1.2322563787476262e-13, 1.4631881898029349]
end

# At low temperature, magnetization should be that of the ground state (approximately MS=-1/2,
# having magnetization of <-1/2| Mz | -1/2> = - <-1/2|Sz|-1/2> = +1/2)
function test_average_magnetic_moment3()
    nel = 9
    norb = 5
    l = 2
    hLFT = diagm([0.3, 0.05, 0.0, 0.05, 0.1])   # energies of x2-y2, xz, z2, yz, xy
    F = Dict(0 => 0.0, 2 => 0.0, 4 => 0.0)    # does not matter for d9 system
    zeta = 0.0
    param = MagFieldLFT.LFTParam(nel, norb, l, hLFT, F, zeta)

    T = 0.0001
    B0_mol = [0, 0, 1.0e-7]

    H_fieldfree, L, S, Mel = MagFieldLFT.calc_operators_SDbasis(param)
    energies, states = MagFieldLFT.calc_solutions_magfield(H_fieldfree, L, S, B0_mol)
    Mel_avg_finitefield = MagFieldLFT.calc_average_magneticmoment(energies, states, Mel, T)

    return norm(Mel_avg_finitefield - [0.0, 0.0, 0.5]) < 1.0e-4
end

function test_integrate_spherical()
    Y_1_m1(theta,phi) = 0.5*sqrt(3/(2pi)) * exp(-im*phi)*sin(theta)
    Y_1_0(theta,phi) = 0.5*sqrt(3/pi) * cos(theta)
    grid = MagFieldLFT.spherical_product_grid(50,50)
    # Results of integration: inner products [<Y_1_m1|Y_1_m1>, <Y_1_m1|Y_1_0>, <Y_1_0|Y_1_0>]
    f(x,y) = [Y_1_m1(x,y)'*Y_1_m1(x,y), Y_1_m1(x,y)'*Y_1_0(x,y), Y_1_0(x,y)'*Y_1_0(x,y)]
    integrals = MagFieldLFT.integrate_spherical(f, grid)
    integrals = [round(x, digits=3) for x in integrals]
    return integrals ≈ [1.000, 0.000, 1.000]
end

function test_integrate_spherical_Lebedev()
    Y_1_m1(theta,phi) = 0.5*sqrt(3/(2pi)) * exp(-im*phi)*sin(theta)
    Y_1_0(theta,phi) = 0.5*sqrt(3/pi) * cos(theta)
    grid = lebedev_grids[25]
    # Results of integration: inner products [<Y_1_m1|Y_1_m1>, <Y_1_m1|Y_1_0>, <Y_1_0|Y_1_0>]
    f(x,y) = [Y_1_m1(x,y)'*Y_1_m1(x,y), Y_1_m1(x,y)'*Y_1_0(x,y), Y_1_0(x,y)'*Y_1_0(x,y)]
    integrals = MagFieldLFT.integrate_spherical(f, grid)
    integrals = [round(x, digits=10) for x in integrals]
    return integrals ≈ [1.000, 0.000, 1.000]
end

function test_dipole_matrix()
    R = [1,5,-2.0]
    dipmat = MagFieldLFT.calc_dipole_matrix(R)
    ref = [-0.005477225575051661 0.003042903097250923 -0.0012171612389003691;
    0.003042903097250923 0.009128709291752768 -0.006085806194501846;
    -0.0012171612389003691 -0.006085806194501846 -0.0036514837167011074]
    return dipmat ≈ ref
end

function test_dipole_field()
    R = [5,7,9]
    M = [-4.0, 3, 7.5]
    B_dip = MagFieldLFT.dipole_field(M, R)
    ref = [2.9330997775211083e-7, 1.7331548609510163e-7, 1.2230892547235214e-7]
    return B_dip ≈ ref
end

function test_determine_degenerate_sets()
    energies = [0.0, 1e-12, 1e-11, 2.2, 2.2+1e-11, 2.2+1e-9, 7, 9, 9, 9]
    degenerate_sets = MagFieldLFT.determine_degenerate_sets(energies)
    D = MagFieldLFT.DegenerateSet
    ref = [D(0.0, [1,2,3]), D(2.2, [4,5]), D(2.2+1e-9, [6]), D(7.0, [7]), D(9.0, [8,9,10])]
    passed = true
    for i in 1:length(degenerate_sets)
        passed = passed && (degenerate_sets[i].E == ref[i].E)
        passed = passed && (degenerate_sets[i].states == ref[i].states)
    end
    return passed
end

# in weak-field / high-temperature limit, finite-field magnetization should be linear in external field
function test_calc_susceptibility_vanVleck()
    param = read_AILFT_params_ORCA("NiSAL_HDPT.out", "CASSCF")
    T = 298
    B0_mol = [0, 0, 1.0e-7]

    # version 1
    chi = MagFieldLFT.calc_susceptibility_vanVleck(param, T)
    Mel_avg_linear = (1/(4pi*MagFieldLFT.alpha^2))*chi*B0_mol

    # version 2
    H_fieldfree, L, S, Mel = MagFieldLFT.calc_operators_SDbasis(param)
    energies, states = MagFieldLFT.calc_solutions_magfield(H_fieldfree, L, S, B0_mol)
    Mel_avg_finitefield = MagFieldLFT.calc_average_magneticmoment(energies, states, Mel, T)

    return norm(Mel_avg_finitefield - Mel_avg_linear) < 1.0e-10
end

function test_KurlandMcGarvey()
    bohrinangstrom = 0.529177210903
    # atom counting starting from 1 (total number of atoms is 49, NH proton is the last one)
    r_Ni = [0.000,   0.000,   0.000]                      # atom 33
    r_NH = [0.511,  -2.518,  -0.002]                      # atom 49
    r_CH1 = [1.053,   1.540,   3.541]                     # atom 23
    r_CH2 = [-0.961,  -1.048,  -3.741]                    # atom 32
    r_alpha1_alpha2prime_1 = [-1.500,  -3.452,   1.130]   # atom 44
    r_alpha1_alpha2prime_2 = [0.430,  -3.104,   2.402]    # atom 45
    R_NH                   = r_Ni - r_NH
    R_CH1                  = r_Ni - r_CH1
    R_CH2                  = r_Ni - r_CH2
    R_alpha1_alpha2prime_1 = r_Ni - r_alpha1_alpha2prime_1
    R_alpha1_alpha2prime_2 = r_Ni - r_alpha1_alpha2prime_2
    R = [R_NH, R_CH1, R_CH2, R_alpha1_alpha2prime_1, R_alpha1_alpha2prime_2] / bohrinangstrom

    param = read_AILFT_params_ORCA("NiSAL_HDPT.out", "CASSCF")
    T = 298   # I actually did not find in the paper at which temperature they recorded it!?
    shifts = MagFieldLFT.calc_shifts_KurlandMcGarvey(param, R, T)
    ref = [-96.1951957001265, 30.53047905687625, 30.175351679457314, -22.804411834183288, -19.50459482031643]
    # calculated PCS at CASSCF/NEVPT2/QDPT level according to SI of paper:
    # [-61.5, 21.7, 21.3, -15.5, -13.6]
    # My shifts are larger in magnitude by around 50%, but the relative size and sign is correct
    # This could well be an artifact of only using CASSCF for determining the AILFT parameters (and/or of the LFT approximation)
    return shifts ≈ ref
end

function test_KurlandMcGarvey_vs_finitefield()
    bohrinangstrom = 0.529177210903
    # atom counting starting from 1 (total number of atoms is 49, NH proton is the last one)
    r_Ni = [0.000,   0.000,   0.000]                      # atom 33
    r_NH = [0.511,  -2.518,  -0.002]                      # atom 49
    r_CH1 = [1.053,   1.540,   3.541]                     # atom 23
    r_CH2 = [-0.961,  -1.048,  -3.741]                    # atom 32
    r_alpha1_alpha2prime_1 = [-1.500,  -3.452,   1.130]   # atom 44
    r_alpha1_alpha2prime_2 = [0.430,  -3.104,   2.402]    # atom 45
    R_NH                   = r_Ni - r_NH
    R_CH1                  = r_Ni - r_CH1
    R_CH2                  = r_Ni - r_CH2
    R_alpha1_alpha2prime_1 = r_Ni - r_alpha1_alpha2prime_1
    R_alpha1_alpha2prime_2 = r_Ni - r_alpha1_alpha2prime_2
    R = [R_NH, R_CH1, R_CH2, R_alpha1_alpha2prime_1, R_alpha1_alpha2prime_2] / bohrinangstrom

    param = read_AILFT_params_ORCA("NiSAL_HDPT.out", "CASSCF")
    T = 298   # I actually did not find in the paper at which temperature they recorded it!?
    KMcG_shifts = MagFieldLFT.calc_shifts_KurlandMcGarvey(param, R, T)
    grid = MagFieldLFT.spherical_product_grid(100,100)
    B0 = 1.0e-6
    finitefield_shifts = MagFieldLFT.estimate_shifts_finitefield(param, R, B0, T, grid)
    return norm(KMcG_shifts - finitefield_shifts) < 0.1
end

function test_KurlandMcGarvey_vs_finitefield_Lebedev()
    bohrinangstrom = 0.529177210903
    # atom counting starting from 1 (total number of atoms is 49, NH proton is the last one)
    r_Ni = [0.000,   0.000,   0.000]                      # atom 33
    r_NH = [0.511,  -2.518,  -0.002]                      # atom 49
    r_CH1 = [1.053,   1.540,   3.541]                     # atom 23
    r_CH2 = [-0.961,  -1.048,  -3.741]                    # atom 32
    r_alpha1_alpha2prime_1 = [-1.500,  -3.452,   1.130]   # atom 44
    r_alpha1_alpha2prime_2 = [0.430,  -3.104,   2.402]    # atom 45
    R_NH                   = r_Ni - r_NH
    R_CH1                  = r_Ni - r_CH1
    R_CH2                  = r_Ni - r_CH2
    R_alpha1_alpha2prime_1 = r_Ni - r_alpha1_alpha2prime_1
    R_alpha1_alpha2prime_2 = r_Ni - r_alpha1_alpha2prime_2
    R = [R_NH, R_CH1, R_CH2, R_alpha1_alpha2prime_1, R_alpha1_alpha2prime_2] / bohrinangstrom

    param = read_AILFT_params_ORCA("NiSAL_HDPT.out", "CASSCF")
    T = 298   # I actually did not find in the paper at which temperature they recorded it!?
    KMcG_shifts = MagFieldLFT.calc_shifts_KurlandMcGarvey(param, R, T)
    grid = lebedev_grids[20]
    B0 = 1.0e-7
    finitefield_shifts = MagFieldLFT.estimate_shifts_finitefield(param, R, B0, T, grid)
    return norm(KMcG_shifts - finitefield_shifts) < 1.0e-6
end

function test_Fderiv2_numeric_vs_analytic()
    param = read_AILFT_params_ORCA("CrF63-.out", "CASSCF")
    B0_mol = [0.0, 0.0, 1.0e-4]
    h = 1.0e-10   # displacement for numerical derivative
    T = 1.0
    Fderiv1_0 = MagFieldLFT.calc_F_deriv1(param, T, B0_mol)
    Fderiv1_x = MagFieldLFT.calc_F_deriv1(param, T, B0_mol+[h, 0, 0])
    Fderiv1_y = MagFieldLFT.calc_F_deriv1(param, T, B0_mol+[0, h, 0])
    Fderiv1_z = MagFieldLFT.calc_F_deriv1(param, T, B0_mol+[0, 0, h])
    Fderiv2 = MagFieldLFT.calc_F_deriv2(param, T, B0_mol)
    Fderiv2_numeric = zeros(3,3)
    Fderiv2_numeric[1, :] = (1/h)*(Fderiv1_x - Fderiv1_0)
    Fderiv2_numeric[2, :] = (1/h)*(Fderiv1_y - Fderiv1_0)
    Fderiv2_numeric[3, :] = (1/h)*(Fderiv1_z - Fderiv1_0)
    return norm(Fderiv2-Fderiv2_numeric) < 0.1
end

function test_Fderiv3_numeric_vs_analytic()
    param = read_AILFT_params_ORCA("NiSAL_HDPT.out", "CASSCF")
    B0_mol = [0.0, 0.0, 1.0e-4]
    h = 1.0e-10   # displacement for numerical derivative
    T = 1.0
    Fderiv2_0 = MagFieldLFT.calc_F_deriv2(param, T, B0_mol)
    Fderiv2_x = MagFieldLFT.calc_F_deriv2(param, T, B0_mol+[h, 0, 0])
    Fderiv2_y = MagFieldLFT.calc_F_deriv2(param, T, B0_mol+[0, h, 0])
    Fderiv2_z = MagFieldLFT.calc_F_deriv2(param, T, B0_mol+[0, 0, h])
    Fderiv3 = MagFieldLFT.calc_F_deriv3(param, T, B0_mol)
    Fderiv3_numeric = zeros(3,3,3)
    Fderiv3_numeric[1, :, :] = (1/h)*(Fderiv2_x - Fderiv2_0)
    Fderiv3_numeric[2, :, :] = (1/h)*(Fderiv2_y - Fderiv2_0)
    Fderiv3_numeric[3, :, :] = (1/h)*(Fderiv2_z - Fderiv2_0)
    # note: elements of the tensor have magnitudes that are all larger than 1e5
    return norm(Fderiv3-Fderiv3_numeric) < 6000
end

function test_Fderiv4_numeric_vs_analytic()
    param = read_AILFT_params_ORCA("NiSAL_HDPT.out", "CASSCF")
    B0_mol = [0.0, 0.0, 1.0e-4]
    h = 1.0e-10   # displacement for numerical derivative
    T = 1.0
    Fderiv3_0 = MagFieldLFT.calc_F_deriv3(param, T, B0_mol)
    Fderiv3_x = MagFieldLFT.calc_F_deriv3(param, T, B0_mol+[h, 0, 0])
    Fderiv3_y = MagFieldLFT.calc_F_deriv3(param, T, B0_mol+[0, h, 0])
    Fderiv3_z = MagFieldLFT.calc_F_deriv3(param, T, B0_mol+[0, 0, h])
    Fderiv4 = MagFieldLFT.calc_F_deriv4(param, T, B0_mol)
    Fderiv4_numeric = zeros(3,3,3,3)
    Fderiv4_numeric[1, :, :, :] = (1/h)*(Fderiv3_x - Fderiv3_0)
    Fderiv4_numeric[2, :, :, :] = (1/h)*(Fderiv3_y - Fderiv3_0)
    Fderiv4_numeric[3, :, :, :] = (1/h)*(Fderiv3_z - Fderiv3_0)
    return rel_diff_norm(Fderiv4, Fderiv4_numeric) < 1e-4
end

function test_Fderiv4_numeric_vs_analytic_zerofield()
    param = read_AILFT_params_ORCA("NiSAL_HDPT.out", "CASSCF")
    B0_mol = [0.0, 0.0, 0.0]
    h = 1.0e-10   # displacement for numerical derivative
    T = 1.0
    Fderiv3_0 = MagFieldLFT.calc_F_deriv3(param, T, B0_mol)
    Fderiv3_x = MagFieldLFT.calc_F_deriv3(param, T, B0_mol+[h, 0, 0])
    Fderiv3_y = MagFieldLFT.calc_F_deriv3(param, T, B0_mol+[0, h, 0])
    Fderiv3_z = MagFieldLFT.calc_F_deriv3(param, T, B0_mol+[0, 0, h])
    Fderiv4 = MagFieldLFT.calc_F_deriv4(param, T, B0_mol)
    Fderiv4_numeric = zeros(3,3,3,3)
    Fderiv4_numeric[1, :, :, :] = (1/h)*(Fderiv3_x - Fderiv3_0)
    Fderiv4_numeric[2, :, :, :] = (1/h)*(Fderiv3_y - Fderiv3_0)
    Fderiv4_numeric[3, :, :, :] = (1/h)*(Fderiv3_z - Fderiv3_0)
    return rel_diff_norm(Fderiv4, Fderiv4_numeric) < 1e-4
end

function rel_diff_norm(value, ref)
    return norm(value-ref)/norm(ref)
end

function test_print_composition()
    C = [1,2,3]
    C = C/norm(C) # normalize
    U = [1 0 0; 0 1 0; 0 0 1]
    labels = ["ex", "ey", "ez"]
    thresh = 0.98
    buf = IOBuffer()
    MagFieldLFT.print_composition(C, U, labels, thresh, buf)
    printed_string = String(take!(buf))
    ref = """
     64.29%  ez
     28.57%  ey
      7.14%  ex
    """
    return printed_string == ref
end

function test_print_composition2()
    param = read_AILFT_params_ORCA("NiSAL_HDPT.out", "CASSCF")
    exc = MagFieldLFT.calc_exclists(param)
    H_fieldfree = MagFieldLFT.calc_H_fieldfree(param, exc)
    energies_rel, states_rel = MagFieldLFT.calc_solutions(H_fieldfree)
    H_nonrel = MagFieldLFT.calc_H_nonrel(param, exc)
end

function test_group_eigenvalues()
    values = [1,1,2,5,7,7,7,10]
    unique_values, indices = MagFieldLFT.group_eigenvalues(values)
    ref_values = [1,2,5,7,10]
    ref_indices = [[1,2], [3], [4], [5,6,7], [8]]
    return (unique_values == ref_values) && (indices == ref_indices)
end

function test_adapt_basis()
    param = read_AILFT_params_ORCA("NiSAL_HDPT.out", "CASSCF")
    exc = MagFieldLFT.calc_exclists(param)
    H = Hermitian(MagFieldLFT.calc_H_nonrel(param, exc))
    Sx, Sy, Sz = MagFieldLFT.calc_S(param.l, exc)
    S2 = Sx*Sx + Sy*Sy + Sz*Sz
    vals, vecs = eigen(H)
    C_list, labels_list = MagFieldLFT.adapt_basis([vecs], [Vector{Float64}(undef, 0)], Hermitian(S2))
    C_list2, labels_list2 = MagFieldLFT.adapt_basis(C_list, labels_list, Hermitian(Sz))
    #println(diag(C_list2[1]'*S2*C_list2[1]))
    #println(diag(C_list2[2]'*S2*C_list2[2]))
    println(labels_list2[1])
    println(labels_list2[2])
    println(labels_list2[3])
    println(labels_list2[4])
    return false
end


@testset "MagFieldLFT.jl" begin
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
    @test test_Ercomplex()
    @test test_calc_free_energy()
    @test test_average_magnetic_moment()
    @test test_average_magnetic_moment2()
    @test test_average_magnetic_moment3()
    @test test_integrate_spherical()
    @test test_integrate_spherical_Lebedev()
    @test test_dipole_matrix()
    @test test_dipole_field()
    @test test_determine_degenerate_sets()
    @test test_calc_susceptibility_vanVleck()
    @test test_KurlandMcGarvey()
    @test test_KurlandMcGarvey_vs_finitefield()
    @test test_KurlandMcGarvey_vs_finitefield_Lebedev()
    @test test_Fderiv2_numeric_vs_analytic()
    @test test_Fderiv3_numeric_vs_analytic()
    @test test_Fderiv4_numeric_vs_analytic()
    @test test_Fderiv4_numeric_vs_analytic_zerofield()
    @test test_print_composition()
    @test test_group_eigenvalues()
    @test_broken test_adapt_basis()
end
