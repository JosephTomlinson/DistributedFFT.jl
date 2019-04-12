using Distributed
addprocs(2)
@everywhere begin 
    using Pkg
    Pkg.activate("..")
    using Test
    using DistributedFFT
end

include("FFTtests.jl")
