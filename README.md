# DistributedFFT.jl
Work in progress generalization of a distributed 3D BFFT I wrote for another project.

Currently only implements `brfft` but others are planned when I can find time.

Also only supports out of place, but inplace variants are planned once I can find time.

Package can be installed with

```julia
pkg> add https://github.com/JosephTomlinson/DistributedFFT.jl.git
```

Example usage
```julia
using Distributed
addprocs(4)
@everywhere begin
  using DistributedFFT
  using DistributedArrays
end
pids = workers()
np = length(pids)
D = drand(Complex{Float64}, (51, 100, 100), pids, [np,1,1])
C = brfft(D,100)
```
