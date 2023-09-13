using Test

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
    MagFieldLFT.create_SDs(3,2) == ref
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
    lz, lplus, lminus = MagFieldLFT.calc_lops(l)
    return lz ≈ ref_lz && lplus ≈ ref_lplus && lminus ≈ ref_lminus
end

function test_calcERIs_complex()
    F = Dict(0=>3.2, 2=>1.72, 4=>2.20)
    l = 2
    ERIs = MagFieldLFT.calcERIs_complex(l, F)
    return ERIs[2,1,4,5] ≈ (-2.12/9) && ERIs[3,2,5,5] ≈ 0.0 && ERIs[2,2,5,5] ≈ (195.92/63)
end

function test_calcERIs_real()
    F = Dict(0=>3.2, 2=>1.72, 4=>2.20)
    l = 2
    ERIs = MagFieldLFT.calcERIs_real(l, F)
    return ERIs[2,2,5,5] ≈ (-195.92/63) && ERIs[1,4,2,5] ≈ (-0.64/21)
end

@testset "MagFieldLFT.jl" begin
    @test test_iscanonical1()
    @test test_iscanonical2()
    @test test_iscanonical3()
    @test test_createSDs()
    @test test_U_complex2real()
    @test test_calc_lops()
    @test test_calcERIs_complex()
    @test test_calcERIs_real()
end