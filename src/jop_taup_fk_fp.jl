"""
    A = JopTauP_FK_FP(dom; t0=0.0, x0=0.0, Δt=10.0, Δx=10.0, ...])

where `A` is the 2D Tau-P operator mapping from `t-x` to `tau-p` via interpolation from F-K to F-P.
The domain of the operator is `nt` by `nx` with precision T, `Δt` and `t0` define the T axis, 
and `Δx` and `x0` define the X axis.

The additional named optional arguments along with their default values are:
  `taperT=(0.0,0.0)` - beginning and end taper in the T-direction before transforming from `t-x` to `tau-p`
  `taperX=(0.0,0.0)` - beginning and end taper in the X-direction before transforming from `t-x` to `tau-p` 
  `vmin=5000` maximum velocity for ray parameter sampling (m/s)
  `np`=201 number of ray parameters to sample in [-vmin,+vmin]
  `padt=1.0,padx=1.0` - fractional padding in depth and offset to apply before applying the Fourier transfrom
  `interpsinc=1` - use linear (0) or sinc (1) interpolation for the F-K to F-P mapping, default is sinc  
"""
function JopTauP_FK_FP(
        dom::JetAbstractSpace{T};
        vmin = 500.0,
        np = 201,
        t0 = 0.0,
        x0 = 0.0,
        Δt = 10.0,
        Δx = 10.0,
        padt = 1.0,
        padx = 1.0,
        taperT = (0.0,0.0),
        taperX = (0.0,0.0),
        interpsinc = 1) where {T}
    Δt < 0.0 && error("expected Δt > 0.0, got Δt=$(Δt)")
    Δx < 0.0 && error("expected Δh > 0.0, got Δx=$(Δx)")

    nt,nx = size(dom)
    t0,x0,Δt,Δx,vmin = T(t0),T(x0),T(Δt),T(Δx),T(vmin)

    nfft_t = nextprod([2,3,5,7], round(Int, nt * (1 + padt)))
    nfft_x = nextprod([2,3,5,7], round(Int, nx * (1 + padx)))
    @show nt,nfft_t
    @show nx,nfft_x

    # build the interpolation matrix
    (indexPtoK_T, indexPtoK_X, indexPtoK_P, matrixPtoK) = interpolation_matrix(nt,nx,np,nfft_t,nfft_x,Δt,Δx,vmin; interpsinc=interpsinc)

    # T-X taper
    taperTX = JopTaper(dom, (1,2), (taperT[1],taperX[1]), (taperT[2],taperX[2]))

    JopLn(
        dom = dom, 
        rng = JetSpace(T, nt, np), 
        df! = JopTauP_FK_FP_df!, 
        df′! = JopTauP_FK_FP_df′!,
        s = (nfft_t=nfft_t, nfft_x=nfft_x, t0=t0, x0=x0, Δt=Δt, Δx=Δx, taperTX=taperTX,
            indexPtoK_T=indexPtoK_T, indexPtoK_X=indexPtoK_X, indexPtoK_P=indexPtoK_P, matrixPtoK=matrixPtoK))
end
export JopTauP_FK_FP

# build the interpolation matrix for the tau-p transform: p = kx / kt
# interpsinc: linear(0) or sinc(1) interpolation, default linear
# sinclength: number of points to use in the sinc interpolation kernel, default 8
function interpolation_matrix(nt::Int64, nx::Int64, np::Int64, nfft_t::Int64, nfft_x::Int64, 
        Δt::T, Δx::T, vmin::T; interpsinc::Int64=0) where {T}
    pmin = - 1000 / vmin
    pmax = + 1000 / vmin
    pvalues = [pmin + (pmax - pmin) * (i-1) / (np-1) for i in 1:np]
    Δp = pvalues[2] - pvalues[1]

    ktvalues = convert(Array{T}, fftfreq(nfft_t, 1 / Δt))
    kxvalues = convert(Array{T}, fftfreq(nfft_x, 1 / Δx))

    indexPtoK_T = Int64[]
    indexPtoK_X = Int64[]
    indexPtoK_P = Int64[]
    matrixPtoK = T[]

    # start at kfft_t=2 to avoid kt=0 and division by zero in p = kx / kt

    if interpsinc == 0
        @info "linear interpolation"
        for kfft_t ∈ 2:nfft_t
            kt = ktvalues[kfft_t]

            for kfft_x ∈ 1:nfft_x
                kx = kxvalues[kfft_x]
                pp = kx / kt

                if pp > pmin && pp < pmax
                    kp = Int64(floor((pp - pmin) / Δp) + 1)
                    kp = clamp(kp, 1, np-1)
                    p1 = pvalues[kp+0]
                    p2 = pvalues[kp+1]
                    dp = (pp - p1) / (p2 - p1)

                    push!(indexPtoK_T, kfft_t)
                    push!(indexPtoK_X, kfft_x)
                    push!(indexPtoK_P, kp + 0)
                    push!(matrixPtoK, 1 - dp)
                    
                    push!(indexPtoK_T, kfft_t)
                    push!(indexPtoK_X, kfft_x)
                    push!(indexPtoK_P, kp + 1)
                    push!(matrixPtoK, dp)
                end
            end
        end
    else
        @info "sinc interpolation"
        sinclength = 8
		sinclength2 = max(1, div(sinclength,2))
        tiny = 2^(-24)

        for kfft_t ∈ 2:nfft_t
            kt = ktvalues[kfft_t]

            for kfft_x ∈ 1:nfft_x
                kx = kxvalues[kfft_x]
                pp = kx / kt

                kp1 = Int64(floor((pp - sinclength2 * Δp - pmin) / Δp) + 1)
                kp2 = Int64(floor((pp + sinclength2 * Δp - pmin) / Δp) + 1)

                kp1 = clamp(kp1, 1, np)
                kp2 = clamp(kp2, 1, np)

                sum = 0.0
                for kp ∈ kp1:kp2
                    x = (pp - pvalues[kp]) / Δp;

                    push!(indexPtoK_T, kfft_t)
                    push!(indexPtoK_X, kfft_x)
                    push!(indexPtoK_P, kp)

                    if abs(x) > tiny
                        # push!(matrixPtoK, sin(π * x) / (π * x))   # more artifacts
                        push!(matrixPtoK, sin(x) / x)
                    else
                        # push!(matrixPtoK, 1)
                        push!(matrixPtoK, cos(x))
                    end 
                    sum += matrixPtoK[end]
                end
                # @show sum
            end
        end
    end

    (indexPtoK_T, indexPtoK_X, indexPtoK_P, matrixPtoK)
end

# Forward Tau-P transform
# 1. Forward temporal and spatial Fourier transforms: T-X to F-K
# 2. Spatial phase shift to center x0 at the origin
# 3. Map from F-K to F-P using precomputed purely real interpolation matrix
# 4. Inverse Temporal Fourier transform: F-P to Tau-P
function JopTauP_FK_FP_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; nfft_t, nfft_x, t0, x0, Δt, Δx, taperTX, indexPtoK_T, indexPtoK_X, indexPtoK_P, matrixPtoK, kwargs...) where {T}
    nt, nx, np = size(m,1), size(m,2), size(d,2)
    
    mtmp = zeros(T, nfft_t, nfft_x)
    mtmp[1:nt,1:nx] .= taperTX * m
    
    # Forward 2D Fourier temporal and spatial transforms
    M = fft(mtmp) .* (1 / sqrt(nfft_t * nfft_x))
    
    # spatial phase shift
    if abs(x0) > 0.0
        kx = convert(Array{T}, fftfreq(nfft_x, 1/Δx))
        for kfft_x = 1:nfft_x
            pshift = exp(- im * 2π * kx[kfft_x] * x0) 
            for kfft_t = 1:nfft_t
                M[kfft_t,kfft_x] *= pshift
            end
        end
    end

    # stretch f-k to f-p by sinc interpolation
    D = zeros(Complex{T}, nfft_t, np)
    kx = convert(Array{T}, fftfreq(nfft_x, 1/Δx))
    for k ∈ eachindex(matrixPtoK)
        kfft_t = indexPtoK_T[k]
        kfft_x = indexPtoK_X[k]
        kp     = indexPtoK_P[k]
        D[kfft_t,kp] += matrixPtoK[k] * M[kfft_t,kfft_x];
    end

    # Inverse 1D temporal Fourier transform
    dtmp = zeros(Complex{T}, nfft_t, np)
    for kp ∈ 1:np
        dtmp[:,kp] = bfft(D[:,kp]) .* (1 / sqrt(nfft_t))
    end

    d .= real.(dtmp[1:nt,1:np])
end

# Adjoint Tau-P transform
# 1. Forward Temporal Fourier transform: Tau-P to F-P
# 2. Map from F-P to F-K using precomputed purely real interpolation matrix
# 3. Spatial phase shift to center x0 at the origin
# 4. Inverse temporal and spatial Fourier transforms: F-K to T-X
function JopTauP_FK_FP_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; nfft_t, nfft_x, t0, x0, Δt, Δx, taperTX, indexPtoK_T, indexPtoK_X, indexPtoK_P, matrixPtoK, kwargs...) where {T}
    nt, nx, np = size(m,1), size(m,2), size(d,2)

    dtmp = zeros(Complex{T}, nfft_t, np)
    dtmp[1:nt,1:np] .= d
    
    # Forward 1D Fourier temporal and spatial transforms
    D = zeros(Complex{T}, nfft_t, np)
    for kp ∈ 1:np
        D[:,kp] = fft(dtmp[:,kp]) .* (1 / sqrt(nfft_t))
    end

    # stretch f-p to f-k by sinc interpolation
    M = zeros(Complex{T}, nfft_t, nfft_x)
    kx = convert(Array{T}, fftfreq(nfft_x, 1/Δx))
    for k ∈ eachindex(matrixPtoK)
        kfft_t = indexPtoK_T[k]
        kfft_x = indexPtoK_X[k]
        kp     = indexPtoK_P[k]
        M[kfft_t,kfft_x] += matrixPtoK[k] * D[kfft_t,kp];
    end

    # spatial phase shift
    if abs(x0) > 0.0
        kx = convert(Array{T}, fftfreq(nfft_x, 1/Δx))
        for kfft_x = 1:nfft_x
            pshift = exp(+ im * 2π * kx[kfft_x] * x0) 
            for kfft_t = 1:nfft_t
                M[kfft_t,kfft_x] *= pshift
            end
        end
    end

    # Inverse 2D Fourier temporal and spatial transforms
    mtmp = bfft(M) .* (1 / sqrt(nfft_t * nfft_x))

    m .= taperTX * real.(mtmp[1:nt,1:nx])
end
