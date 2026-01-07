"""
    A = JopTauP_FK(dom; t0=0.0, x0=0.0, Δt=10.0, Δx=10.0, ...])

where `A` is the 2D Tau-P operator mapping from `t-x` to `tau-p` via shift and sum in F-K.
The domain of the operator is `nt` by `nx` with precision T, `Δt` and `t0` define the T axis, 
and `Δx` and `x0` define the X axis.

The additional named optional arguments along with their default values are:
  `taperT=(0.0,0.0)` - beginning and end taper in the T-direction before transforming from `t-x` to `tau-p`
  `taperX=(0.0,0.0)` - beginning and end taper in the X-direction before transforming from `t-x` to `tau-p` 
  `vmin=5000` maximum velocity for ray parameter sampling (m/s)
  `np`=201 number of ray parameters to sample in [-vmin,+vmin]
  `padt=1.0,padx=1.0` - fractional padding in depth and offset to apply before applying the Fourier transfrom
  `weight=0` = if non-zero, apply semblance weighting as right preocnditioner
"""
function JopTauP_FK(
        dom::JetAbstractSpace{T};
        vmin = 500.0,
        np = 201,
        t0 = 0.0,
        x0 = 0.0,
        Δt = 10.0,
        Δx = 10.0,
        padt = 1.0,
        taperT = (0.0,0.0),
        taperX = (0.0,0.0),
        weight = 0
        ) where {T}
    Δt < 0.0 && error("expected Δt > 0.0, got Δt=$(Δt)")
    Δx < 0.0 && error("expected Δh > 0.0, got Δx=$(Δx)")

    nt,nx = size(dom)
    t0,x0,Δt,Δx,vmin = T(t0),T(x0),T(Δt),T(Δx),T(vmin)

    nfft = nextprod([2,3,5,7], round(Int, nt * (1 + padt)))
    nfreq = div(nfft,2) + 1

    # T-X taper
    taperTX = JopTaper(dom, (1,2), (taperT[1],taperX[1]), (taperT[2],taperX[2]))

    # semblance placeholder 
    S = ones(T, nfreq, np)

    JopLn(
        dom = dom, 
        rng = JetSpace(T, nt, np), 
        df! = JopTauP_FK_df!, 
        df′! = JopTauP_FK_df′!,
        s = (nfft=nfft, t0=t0, x0=x0, Δt=Δt, Δx=Δx, vmin=vmin, taperTX=taperTX, weight=weight, S=S))
end
export JopTauP_FK

# Forward Tau-P transform
# 1. Forward temporal Fourier transforms: T-X to F-X
# 2. Shift and sum for each P with temporal phase shifts: F-X to F-P
# 3. Inverse Temporal Fourier transform: F-P to Tau-P
function JopTauP_FK_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; nfft, t0, x0, Δt, Δx, vmin, taperTX, weight, S, kwargs...) where {T}
    nt, nx, np = size(m,1), size(m,2), size(d,2)
    
    # Forward 1D temporal Fourier transform
    # Use real-to-complex FFT: only non-negative temporal frequencies
    nfreq = div(nfft,2) + 1
    mtaper = taperTX * m
    M = zeros(Complex{T}, nfreq, nx)
    mtmp = zeros(T, nfft)
    for kx ∈ 1:nx
        mtmp[1:nt] .= mtaper[:,kx]
        M[:,kx] = rfft(mtmp) .* (1 / sqrt(nfft))
    end
    
    # shift and sum f-k to f-p by sinc interpolation
    D = zeros(Complex{T}, nfreq, np)
    pmin = - 1000 / vmin
    pmax = + 1000 / vmin
    frequencies = convert(Array{T}, fftfreq(nfft, 1 / Δt))
    frequencies = frequencies[1:nfreq] # only non-negative frequencies

    # compute semblance as right preconditioner
    if weight > 0
        @info "computing semblance weighting"
        for kp ∈ 1:np
            pp = pmin + (pmax - pmin) * (kp-1) / (np-1)

            for kfft = 1:nfreq
                semblance = Complex{T}(0)
                for kx = 1:nx
                    xx = x0 + Δx * (kx - 1)
                    tt = pp * xx
                    phaseshift = exp(+ im * 2π * frequencies[kfft] * tt)
                    zz = M[kfft,kx] * phaseshift
                    semblance += zz
                end
                S[kfft,kp] = real(semblance * conj(semblance))
            end
        end
    end

    # perform the Tau-P transform
    for kp ∈ 1:np
        pp = pmin + (pmax - pmin) * (kp-1) / (np-1)

        for kfft = 1:nfreq
            for kx = 1:nx
                xx = x0 + Δx * (kx - 1)
                tt = pp * xx
                phaseshift = exp(- im * 2π * frequencies[kfft] * tt)
                D[kfft,kp] += S[kfft,kp] * M[kfft,kx] * phaseshift
            end
        end
    end

    # Inverse 1D temporal Fourier transform (irfft over positive freqs)
    dtmp = zeros(T, nfft, np)
    for kp ∈ 1:np
        dtmp[:,kp] = irfft(D[:,kp] .* (1 / sqrt(nfft)), nfft)
    end

    d .= dtmp[1:nt,1:np]
end

# Adjoint Tau-P transform
# 3. Forward Temporal Fourier transform: Tau-P to F-P
# 2. Shift and sum for each P with temporal phase shifts: F-P to F-X
# 1. Inverse temporal Fourier transforms: F-X to T-X
function JopTauP_FK_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; nfft, t0, x0, Δt, Δx, vmin, taperTX, weight, S, kwargs...) where {T}
    nt, nx, np = size(m,1), size(m,2), size(d,2)

    # Pad to nfft and compute real-to-complex temporal FFT (positive freqs)
    nfreq = div(nfft,2) + 1
    dpad = zeros(T, nfft)
    D = zeros(Complex{T}, nfreq, np)
    for kp ∈ 1:np
        dpad[1:nt] .= d[:,kp]
        D[:,kp] = rfft(dpad) .* (1 / sqrt(nfft))
    end

    # shift and sum f-p to f-k by sinc interpolation
    M = zeros(Complex{T}, nfreq, nx)
    pmin = - 1000 / vmin
    pmax = + 1000 / vmin
    frequencies = convert(Array{T}, fftfreq(nfft, 1 / Δt))
    frequencies = frequencies[1:nfreq] # only non-negative frequencies

    for kp ∈ 1:np
        pp = pmin + (pmax - pmin) * (kp-1) / (np-1)

        for kfft = 1:nfreq
            for kx = 1:nx
                xx = x0 + Δx * (kx - 1)
                tt = pp * xx
                phaseshift = exp(+ im * 2π * frequencies[kfft] * tt)
                M[kfft,kx] += S[kfft,kp] * D[kfft,kp] * phaseshift
            end
        end
    end

    # Inverse temporal transform from positive freqs back to time
    mtmp = zeros(T, nfft, nx)
    for kx ∈ 1:nx
        mtmp[:,kx] = irfft(M[:,kx] .* (1 / sqrt(nfft)), nfft)
    end

    m .= taperTX * mtmp[1:nt,1:nx]
end
