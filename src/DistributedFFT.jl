module DistributedFFT

using Distributed
using DistributedArrays
using DistributedArrays.SPMD
using FFTW



export brfft

function transpose3D(x)
    permutedims(x,(2,1,3))
end


@views function alltoalltranspose3D(x, nx, recievearray; pids=workers())
    pstart = minimum(pids)
    np = length(pids)
    rank = myid() - pstart + 1
    ny = size(x,2)
    extrax = nx % np
    extray = ny % np
    if rank <= extrax
        haveextra = true
    else
        haveextra = false
    end
    if rank <= extray
        haveextray = true
    else
        haveextray = false
    end

    if haveextra
        dx = size(x,1) - 1
    else
        dx = size(x,1)
    end
    nz = size(x,3)
    dy = convert(Int64,floor(ny/np))


    xt = transpose3D(x)

    i=1
    while i < np
        i *= 2
    end
    if i == np
        pow2 = true
    else
        pow2 = false
    end

    if haveextra
        if haveextray
            offset = rank - 1
            recievearray[:,(1+dx*(rank-1)+offset):dx*rank+(offset+1),:] .= 
                        xt[(1+dy*(rank-1)+offset):dy*rank+(offset+1),:,:] 
        else
            offsetx = rank - 1
            offsety = extray
            recievearray[:,(1+dx*(rank-1)+offsetx):dx*rank+(offsetx+1),:] .= 
                        xt[(1+dy*(rank-1)+offsety):dy*rank+(offsety),:,:] 
        end
    else
        if haveextray
            offsetx = extrax
            offsety = rank - 1
            recievearray[:,(1+dx*(rank-1)+offsetx):dx*rank+(offsetx),:] .= 
                        xt[(1+dy*(rank-1)+offsety):dy*rank+(offsety+1),:,:] 
        else
            offsetx = extrax
            offsety = extray
            recievearray[:,(1+dx*(rank-1)+offsetx):dx*rank+offsetx,:] .= 
                        xt[(1+dy*(rank-1)+offsety):dy*rank+offsety,:,:] 
        end
    end

    for i = 1:(np-1)
        if pow2
            source = destination = ((rank-1) ⊻ i) + 1
        else
            source = (rank - i - 1 + np ) % np + 1
            destination = (rank + i - 1 + np) % np + 1
        end
        if destination <= extray
            offset = destination - 1
            SPMD.sendto(destination + pstart - 1, xt[(1+dy*(destination-1)+offset):dy*destination+(offset+1),:,:])
        else
            offset = extray
            SPMD.sendto(destination + pstart - 1, xt[(1+dy*(destination-1)+offset):dy*destination+offset,:,:])
        end

        if source <= extrax
            offset = source - 1
            recievearray[:,(1+dx*(source-1)+offset):dx*source+(offset+1),:] .= SPMD.recvfrom(source + pstart - 1)
        else
            offset = extrax
            recievearray[:,(1+dx*(source-1)+offset):dx*source+offset,:] .= SPMD.recvfrom(source + pstart - 1)
        end
        barrier(; pids=pids)
    end
end

function FFTW.brfft(D::DArray, d::Int, dims=1:ndims(D))
    @assert div(d,2) + 1 == size(D,dims[1])
    @assert length(dims) == 3 #Temporary
    pids = procs(D)
    np = length(pids)
    Dout = dzeros((d, size(D,dims[2]), size(D,dims[3])), pids, [np,1,1])
    spmd(_mybrfft, D, d, Dout, dims; pids=procs(D))
    return Dout
end

function _mybrfft(D, d, Dout, dims)
    function mygenerate_plans(D, z, dims)
        plan1 = plan_bfft(D[:L], Tuple(collect(dims[2:end])))
        plan2 = plan_brfft(z, size(z,3), (2,))
        return [plan1, plan2]
    end

    function myapply_plans(D, Dout, dims, plans, z)
        pids = procs(D)[:,1]
        x = plans[1]*D[:L]
        alltoalltranspose3D(x, size(D,dims[1]), z; pids=pids)
        #mul!(Dout[:L], plans[2], z)
        w = plans[2]*z
        alltoalltranspose3D(w, size(D,2), Dout[:L]; pids=pids)
    end
    
    pids = procs(D)
    pstart = minimum(pids)
    np = length(pids)
    rank = myid() - pstart + 1
    
    bcnmesh = size(D,dims[1])
    bymesh = size(D,dims[2])
    bzmesh = size(D,dims[3])
    
    transposehaveextra = (rank <= rem(size(D,dims[2]), np))
    if transposehaveextra
        transpose_size = (div(bymesh, np) + 1, bcnmesh, bzmesh)
    else
        transpose_size = (div(bymesh, np), bcnmesh, bzmesh)
    end
    
    z = zeros(Complex{Float64}, transpose_size)
    

    plans = mygenerate_plans(D, z, dims)
    
    myapply_plans(D, Dout, dims, plans, z)
end
end
