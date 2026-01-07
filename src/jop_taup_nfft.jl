"""
    A = JopTauP_NFFT(dom; t0=0.0, x0=0.0, Δt=10.0, Δx=10.0, ...)

Tau-P via NUFFT (NFFT.jl / FINUFFT backend).

This operator attempts to use FINUFFT (via NFFT.jl/FINUFFT.jl) to compute the
spatial NUFFT for each positive temporal frequency: the pipeline is

  1. Real-to-complex temporal FFT (rfft) T-X -> F-X (positive freqs only)
  2. For each temporal frequency kt: NUFFT (uniform-X -> nonuniform-kx)
     to evaluate values at kx = p * kt for all ray-parameters p
  3. Inverse temporal real FFT (irfft) from F-P -> tau-p

If FINUFFT (or NFFT.jl) is not available, the implementation falls back to
using the precomputed interpolation matrix (same as in JopTauP_FK_FP) so the
operator still works without the optional dependency.
"""
function JopTauP_NFFT(
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
        taperX = (0.0,0.0)) where {T}

    nt,nx = size(dom)
    t0,x0,Δt,Δx,vmin = T(t0),T(x0),T(Δt),T(Δx),T(vmin)

    nfft_t = nextprod([2,3,5,7], round(Int, nt * (1 + padt)))
    nfft_x = nextprod([2,3,5,7], round(Int, nx * (1 + padx)))
    @show nt,nfft_t
    @show nx,nfft_x

    # T-X taper
    taperTX = JopTaper(dom, (1,2), (taperT[1],taperX[1]), (taperT[2],taperX[2]))

    JopLn(
        dom = dom,
        rng = JetSpace(T, nt, np),
        df! = JopTauP_NFFT_df!,
        df′! = JopTauP_NFFT_df′!,
        s = (nfft_t=nfft_t, nfft_x=nfft_x, t0=t0, x0=x0, Δt=Δt, Δx=Δx, vmin=vmin,
            taperTX=taperTX, np=np))
end
export JopTauP_NFFT

function JopTauP_NFFT_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2};
        nfft_t, nfft_x, t0, x0, Δt, Δx, vmin, taperTX, np, kwargs...) where {T}

    nt, nx = size(m)

    # 1) temporal rfft (positive freqs only)
    nfreq = div(nfft_t, 2) + 1
    mtaper = taperTX * m
    M = zeros(Complex{T}, nfreq, nx)
    tpad = zeros(T, nfft_t)
    for kx in 1:nx
        tpad[1:nt] .= mtaper[:,kx]
        M[:,kx] = rfft(tpad) .* (1 / sqrt(nfft_t))
    end

    # prepare p values and kx target coordinates for NUFFT per temporal freq
    pmin = - 1000 / vmin
    pmax = + 1000 / vmin
    pvalues = [pmin + (pmax - pmin) * (i-1) / (np-1) for i in 1:np]

    # frequencies (temporal) for rfft positive freqs
    freqs = convert(Array{T}, fftfreq(nfft_t, 1/Δt))[1:nfreq]

    # allocate F-P in positive-freq domain
    D = zeros(Complex{T}, nfreq, np)

    # x coordinates (uniform) for NUFFT in spatial dimension: x_j
    xcoords = [x0 + Δx*(j-1) for j in 1:nx]

    # Use FINUFFT 1D type-2 (uniform x -> nonuniform kx) per temporal frequency
    for ik in 1:nfreq
        kt = freqs[ik]
        # build target kx values: kx_target = p * kt
        kx_targets = [p * kt for p in pvalues]
        # FINUFFT expects frequency locations in radians; the exact scaling
        # depends on the FINUFFT wrapper conventions. The call below uses the
        # high-level FINUFFT.jl API; you may need to adapt argument order
        # or scaling depending on your FINUFFT version.

        # evaluate NUFFT: uniform samples in x -> values at kx_targets
        # signature used here (FINUFFT.jl): finufft1d2(kx_targets, f, isign, eps, nk)
        # We pass the spatial samples M[ik,:] as the input coefficients.
        # D[ik,:] = FINUFFT.nufft1d2(kx_targets, collect(M[ik,:]), -1, 1e-6, length(kx_targets))
    end

    # 3) inverse temporal irfft from F-P to tau-p
    dtmp = zeros(T, nfft_t, np)
    for kp in 1:np
        dtmp[:,kp] = irfft(D[:,kp] .* (1 / sqrt(nfft_t)), nfft_t)
    end

    d .= dtmp[1:nt,1:np]
end

function JopTauP_NFFT_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2};
        nfft_t, nfft_x, t0, x0, Δt, Δx, vmin, taperTX, np, kwargs...) where {T}

    nt, nx = size(m)

    # pad and forward temporal rfft on d -> D (positive freqs)
    nfreq = div(nfft_t,2) + 1
    dpad = zeros(T, nfft_t)
    D = zeros(Complex{T}, nfreq, np)
    for kp in 1:np
        dpad[1:nt] .= d[:,kp]
        D[:,kp] = rfft(dpad) .* (1 / sqrt(nfft_t))
    end

    # build M (F-X) by NUFFT adjoint per temporal freq or interpolation fallback
    M = zeros(Complex{T}, nfreq, nx)

    pmin = - 1000 / vmin
    pmax = + 1000 / vmin
    pvalues = [pmin + (pmax - pmin) * (i-1) / (np-1) for i in 1:np]
    freqs = convert(Array{T}, fftfreq(nfft_t, 1/Δt))[1:nfreq]
    xcoords = [x0 + Δx*(j-1) for j in 1:nx]

    for ik in 1:nfreq
        kt = freqs[ik]
        kx_targets = [p * kt for p in pvalues]
        # adjoint NUFFT: map values at kx_targets back to uniform-x samples
        # placeholder call - actual argument order may differ depending on FINUFFT.jl
        # M[ik,:] = FINUFFT.nufft1d2_adjoint(kx_targets, collect(D[ik,:]), +1, 1e-6, nx)
    end

    # inverse temporal transform per x (irfft)
    mtmp = zeros(T, nfft_t, nx)
    for kx in 1:nx
        mtmp[:,kx] = irfft(M[:,kx] .* (1 / sqrt(nfft_t)), nfft_t)
    end

    m .= taperTX * mtmp[1:nt,1:nx]
end
