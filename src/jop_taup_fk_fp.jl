"""
    A = JopTauP_FK_FP(dom; z0=0.0, x0=0.0, Δz=10.0, Δx=10.0, ...])

where `A` is the 2D slant-stack operator mapping from `z-x` to `tau-p`.  The domain of the operator
is `nz` x `nx` with precision T, `Δz` and `z0` define the Z axis, and `Δx` and `x0` define the X axis, 
The additional named optional arguments along with their default values are:
  `vmax=5000` maximum velocity for ray parameter sampling
  `np`=201 number of ray parameters to sample in [-vmax,+vmax]
  `padz=1.0,padx=1.0` - fractional padding in depth and offset to apply before applying the Fourier transfrom
"""
function JopTauP_FK_FP(
        dom::JetAbstractSpace{T};
        vmax = 5000.0,
        np = 201,
        z0 = 0.0,
        x0 = 0.0,
        Δz = 10.0,
        Δx = 10.0,
        padz = 1.0,
        padx = 1.0) where {T}
    Δz < 0.0 && error("expected Δz > 0.0, got Δz=$(Δz)")
    Δx < 0.0 && error("expected Δh > 0.0, got Δx=$(Δx)")

    nz,nx = size(dom)

    nzfft = nextprod([2,3,5,7], round(Int, nz * (1 + padz)))
    nxfft = nextprod([2,3,5,7], round(Int, nx * (1 + padx)))

    z0,x0 = T(z0),T(x0)
    Δz,Δx = T(Δz),T(Δx)

    JopLn(
        dom = dom, 
        rng = JetSpace(T, nz, np), 
        df! = JopTauP_FK_FP_df!, 
        df′! = JopTauP_FK_FP_df′!,
        s = (nzfft=nzfft, nxfft=nxfft, z0=z0, x0=x0, Δz=Δz, Δx=Δx))
end
export JopTauP_FK_FP

# Forward tau-p transform: d = A * m
# compute tau-p model from z-x data -- d = fft' ptok xshift fft m
function JopTauP_FK_FP_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; nzfft, nxfft, z0, x0, Δz, Δx, kwargs...) where {T}
    nz, nx, np = size(m,1), size(m,2), size(d,2)
    fftscale = 1 / sqrt(nzfft * nxfft)

    write(stdout,"\n")
    @show nz,nzfft
    @show nx,nxfft
    @show np
    @show z0,x0
    @show Δz,Δx

    mpad = zeros(T, nzfft, nxfft)
    mpad[1:nz,1:nx] .= m
    
    # un-normalized forward Fourier transform
    M = fft(mpad) .* fftscale

    freqZ = convert(Array{T}, fftfreq(nzfft, 1/Δz))
    freqX = convert(Array{T}, fftfreq(nxfft, 1/Δx))

    # # spatial phase shift
    if abs(x0) > 0.0
        @info "applying spatial phase shift for x0=$(x0)"
        for kfftz = 1:nzfft
            for kfftx = 1:nxfft
                pshift = (- x0) * 2.0 * π * freqX[kfftx]
                M[kfftz,kfftx] *= cos(pshift) + im*sin(pshift)
            end
        end
    end

    # stretch f-k to f-p by sinc interpolation
    
    # un-normalized inverse Fourier transform
    dtmp = bfft(M) .* fftscale

    write(stdout,"\n")
    @show extrema(real.(m))
    @show extrema(real.(M))
    @show extrema(real.(dtmp))
    @show extrema(real.(d))

    write(stdout,"\n")
    @show nz,nx,np
    @show size(m)
    @show size(d)
    d .= real.(dtmp[1:nz,1:np])
end

# Adjoint tau-p transform: n = A' * d
# compute z-x data from tau-p model -- m = fft' xshift' ptok' fft d
function JopTauP_FK_FP_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; nzfft, nxfft, z0, x0, Δz, Δx, kwargs...) where {T}
    nz, nx, np = size(m,1), size(m,2), size(d,2)

    mpad = zeros(T, nzfft, nxfft)
    mpad[1:nz,1:nx] .= m

    M = fft(mpad)
    dtmp = similar(d)

    # m .= TX * (brfft(M, nzfft)[1:nz,1:nx])
    m
end
