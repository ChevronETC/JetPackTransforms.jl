"""
    A = JopTauP_FK_FP(dom; t0=0.0, x0=0.0, Δt=10.0, Δx=10.0, ...])

where `A` is the 2D Tau-P operator mapping from `t-x` to `tau-p`.  The domain of the operator
is `nt` by `nx` with precision T, `Δt` and `t0` define the T axis, and `Δx` and `x0` define the X axis, 
The additional named optional arguments along with their default values are:
  `vmin=5000` maximum velocity for ray parameter sampling (m/s)
  `np`=201 number of ray parameters to sample in [-vmin,+vmin]
  `padt=1.0,padx=1.0` - fractional padding in depth and offset to apply before applying the Fourier transfrom
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
        padx = 1.0) where {T}
    Δt < 0.0 && error("expected Δt > 0.0, got Δt=$(Δt)")
    Δx < 0.0 && error("expected Δh > 0.0, got Δx=$(Δx)")

    nt,nx = size(dom)
    t0,x0,Δt,Δx,vmin = T(t0),T(x0),T(Δt),T(Δx),T(vmin)

    nfft_t = nextprod([2,3,5,7], round(Int, nt * (1 + padt)))
    nfft_x = nextprod([2,3,5,7], round(Int, nx * (1 + padx)))

    # build the interpolation matrix
    (indexPtoK_T, indexPtoK_X, indexPtoK_P, matrixPtoK) = interpolation_matrix(nt,nx,np,nfft_t,nfft_x,Δt,Δx,vmin)

    write(stdout,"\n")
    @show extrema(indexPtoK_T)
    @show extrema(indexPtoK_X)
    @show extrema(indexPtoK_P)
    @show extrema(matrixPtoK)

    JopLn(
        dom = dom, 
        rng = JetSpace(T, nt, np), 
        df! = JopTauP_FK_FP_df!, 
        df′! = JopTauP_FK_FP_df′!,
        s = (nfft_t=nfft_t, nfft_x=nfft_x, t0=t0, x0=x0, Δt=Δt, Δx=Δx, 
            indexPtoK_T=indexPtoK_T, indexPtoK_X=indexPtoK_X, indexPtoK_P=indexPtoK_P, matrixPtoK=matrixPtoK))
end
export JopTauP_FK_FP

# build the interpolation matrix for the tau-p transform: p = kx / kt
# interpsinc: linear(0) or sinc(1) interpolation, default linear
# sinclength: number of points to use in the sinc interpolation kernel, default 8
function interpolation_matrix(nt::Int64, nx::Int64, np::Int64, nfft_t::Int64, nfft_x::Int64, 
        Δt::T, Δx::T, vmin::T; interpsinc::Int64=0, sinclength::Int64=8) where {T}
    pmin = - 1000 / vmin
    pmax = + 1000 / vmin
    pvalues = [pmin + (pmax - pmin) * (i-1) / (np-1) for i in 1:np]
    pmin,pmax = extrema(pvalues)
    Δp = pvalues[2] - pvalues[1]
    @show pvalues[1], pvalues[end], Δp, vmin, 1000/vmin

    ktvalues = convert(Array{T}, fftfreq(nfft_t, 1 / Δt))
    kxvalues = convert(Array{T}, fftfreq(nfft_x, 1 / Δx))

    @show extrema(ktvalues)
    @show extrema(kxvalues)

    indexPtoK_T = Int64[]
    indexPtoK_X = Int64[]
    indexPtoK_P = Int64[]
    matrixPtoK = T[]

    # start at kfft_t=2 to avoid kt=0 and division by zero in p = kx / kt
    for kfft_t ∈ 2:nfft_t
        kt = ktvalues[kfft_t]

        for kfft_x ∈ 1:nfft_x
            kx = kxvalues[kfft_x]
            pp = kx / kt

            if pp > pmin && pp < pmax
                ip = Int64(floor((pp - pmin) / Δp) + 1)
                ip = clamp(ip, 1, np-1)
                p1 = pvalues[ip+0]
                p2 = pvalues[ip+1]
                dp = (pp - p1) / (p2 - p1)

                # @printf("it,ix,kt,kx,p1,pp,p2,dp; %4d %4d %+8.6f %+8.6f %+8.6f %+8.6f %+8.6f %+8.6f\n",kfft_t,kfft_x,kt,kx,p1,pp,p2,dp)

                push!(indexPtoK_T, kfft_t)
                push!(indexPtoK_X, kfft_x)
                push!(indexPtoK_P, ip + 0)
                push!(matrixPtoK, dp)
                # push!(matrixPtoK, 1 - dp)

                push!(indexPtoK_T, kfft_t)
                push!(indexPtoK_X, kfft_x)
                push!(indexPtoK_P, ip + 1)
                push!(matrixPtoK, 1 - dp)
                # push!(matrixPtoK, dp)
            end
        end
    end

    # write(stdout,"\n")
    # @show size(indexPtoK_T)
    # @show size(indexPtoK_X)
    # @show size(indexPtoK_P)
    # @show size(matrixPtoK)

    (indexPtoK_T, indexPtoK_X, indexPtoK_P, matrixPtoK)
end

# Forward Tau-P transform
# 1. Forward temporal and spatial Fourier transforms: T-X to F-K
# 2. Spatial phase shift to center x0 at the origin
# 3. Map from F-K to F-P using precomputed purely real interpolation matrix
# 4. Inverse Temporal Fourier transform: F-P to Tau-P
function JopTauP_FK_FP_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; nfft_t, nfft_x, t0, x0, Δt, Δx, indexPtoK_T, indexPtoK_X, indexPtoK_P, matrixPtoK, kwargs...) where {T}
    nt, nx, np = size(m,1), size(m,2), size(d,2)
    
    mtmp = zeros(T, nfft_t, nfft_x)
    mtmp[1:nt,1:nx] .= m
    
    # Forward Fourier temporal and spatial transforms
    M = fft(mtmp) .* (1 / sqrt(nfft_t * nfft_x))
    
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

    # stretch f-k to f-p by sinc interpolation
    D = zeros(Complex{T}, nfft_t, np)
    for k ∈ eachindex(matrixPtoK)
        kfft_t = indexPtoK_T[k]
        kfft_x = indexPtoK_X[k]
        kp     = indexPtoK_P[k]
        D[kfft_t,kp] += matrixPtoK[k] * M[kfft_t,kfft_x];
    end

    # Inverse temporal Fourier transform
    dtmp = zeros(Complex{T}, nfft_t, np)
    for kp ∈ 1:np
        dtmp[:,kp] = bfft(D[:,kp]) .* (1 / sqrt(nfft_t))
    end

    d .= real.(dtmp[1:nt,1:np])
end

# Adjoint Tau-P transform
# 1. Forward Temporal Fourier transform: F-P to Tau-P
# 2. Map from F-P to F-K using precomputed purely real interpolation matrix
# 3. Spatial phase shift to center x0 at the origin
# 4. Inverse temporal and spatial Fourier transforms: F-K to T-X
function JopTauP_FK_FP_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; nfft_t, nfft_x, t0, x0, Δt, Δx, indexPtoK_T, indexPtoK_X, indexPtoK_P, matrixPtoK, kwargs...) where {T}
    nt, nx, np = size(m,1), size(m,2), size(d,2)

    dtmp = zeros(Complex{T}, nfft_t, np)
    dtmp[1:nt,1:np] .= d
    
    # Forward Fourier temporal and spatial transforms
    D = zeros(Complex{T}, nfft_t, np)
    for kp ∈ 1:np
        D[:,kp] = fft(dtmp[:,kp]) .* (1 / sqrt(nfft_t))
    end

    # stretch f-p to f-k by sinc interpolation
    M = zeros(Complex{T}, nfft_t, nfft_x)
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
            pshift = exp(- im * 2π * kx[kfft_x] * x0) 
            for kfft_t = 1:nfft_t
                M[kfft_t,kfft_x] *= pshift
            end
        end
    end

    mtmp = bfft(M) .* (1 / sqrt(nfft_t * nfft_x))

    m .= real.(mtmp[1:nt,1:nx])
end
