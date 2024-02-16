using Test
using Random
import InsideOutSMC: categorical, inverse_cdf!

function test_categorical()
    weights = [0.5, 0.0, 0.5, 0.0]
    for _ = 1:10
        idx = categorical(weights)
        @test idx ∈ (1, 3)
    end
    uniform_rv = 0.35
    weights = fill(0.1, 10)
    @test categorical(uniform_rv, weights) == 4
end

@testset "Tests for categorical" begin
    test_categorical()
end

function test_inverse_cdf!()
    # Test case 1
    idx = zeros(Int, 3)
    uniforms = [0.1, 0.5, 0.9]
    weights = [0.2, 0.3, 0.5]
    expected = [1, 2, 3]
    @test inverse_cdf!(idx, uniforms, weights) == expected

    # Test case 2
    idx = zeros(Int, 4)
    uniforms = [0.2, 0.4, 0.6, 0.8]
    weights = [0.1, 0.2, 0.3, 0.4]
    expected = [2, 3, 3, 4]
    @test inverse_cdf!(idx, uniforms, weights) == expected
end

# Run the tests
@testset "Tests for inverse_cdf!" begin
    test_inverse_cdf!()
end
