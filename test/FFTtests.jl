using Test

using DistributedArrays
using FFTW

@testset "Distributed FFT Tests" begin

@testset "brfft Tests" begin
    rtol=1e-4
    atol=1e-5
    pids = workers()
    np = length(pids)
    D = drand(Complex{Float64}, (51, 100, 100), pids, [np,1,1])
    C = copy(Array(D))
    @test isapprox(Array(brfft(D, 100)), brfft(C,100))
end

end
