using Test

using MagFieldLFT

function test_iscanonical1()
    orbstring = [1,2,5,5,6]    # same orbital occurs twice: not canonical
    return iscanonical(orbstring) == false
end

function test_iscanonical2()
    orbstring = [1,2,5,6]  
    return iscanonical(orbstring) == true
end

function test_iscanonical3()
    orbstring = [2,1,5,6]  
    return iscanonical(orbstring) == false
end

function test_createSDs()
    ref = [[1,2,3], [1,2,4], [1,3,4], [2,3,4]]
    create_SDs(3,2) == ref
end

@testset "MagFieldLFT.jl" begin
    @test test_iscanonical1()
    @test test_iscanonical2()
    @test test_iscanonical3()
    @test test_createSDs()
end