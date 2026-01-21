"""
    A = JopSlantStack(dom[; dz=1.0, dh=1.0, h0=0.0, ...])

where `A` is the 2D slant-stack operator mapping for `z-h` to `z-θ` (depth mode) or `t-h` to `tau-p` (time mode).
The domain of the operator is `nz` x `nh` with precision T, `dz` is the depth spacing (or time interval),
`dh` is the offset spacing, and `h0` is the origin of the offset axis.  The additional named optional arguments
along with their default values are,

* `mode="depth` - choose between "depth" and "time" to specify if the input domain is `z-h` or `t-h`.
* `theta=-60:1.0:60` - range of incidence angles used when `mode="depth"`.
* `p=range(-dz/dh,dz/dh,128)` - ray parameter sampling used when `mode="time"`
* `padz=0.0,padh=0.0` - fractional padding in depth and offset to apply before applying the Fourier transform
* `taperz=(0,0)` - beginning and end taper (fractional) in the z-direction (or t-direction) before transforming from `z-h` to `kz-kh`
* `taperh=(0,0)` - beginning and end taper (fractional) in the h-direction before transforming from `z-h` to `kz-kh`
* `taperkz=(0,0)` - beginning and end taper (fractional) in the kz-direction (or frequency) before transforming from `kz-kh` to `z-h` or `f-kh` to `t-h`
* `taperkh=(0,0)` - beginning and end taper (fractional) in the kh-direction before transforming from `kz-kh` to `z-h` or `f-kh` to `t-h`

# Notes

* It should be possible to extend this operator to 3D
* If the mode is "time", then `padz` is the padding for the time dimension, `dz` is the time sampling interval, `taperz` is the taper for the time dimension and `tapkerkz` is the taper for frequency.
* For mode="depth", typically theta needs to cover both positive and negative angles and must be < 90 degrees.
"""
function JopSlantStack(
        dom::JetAbstractSpace{T};
        theta = collect(-60.0:1.0:60.0),
        p = nothing,
        dz = 1.0,
        dh = 1.0,
        h0 = 0.0,
        padz = 0.0,
        padh = 0.0,
        taperz = (0,0),
        taperh = (0,0),
        taperkz = (0,0),
        taperkh = (0,0),
        mode = "depth") where {T}
    mode ∈ ("depth", "time") || error("expected mode to be either 'depth' or 'time', got mode=$(mode)")
    dz < 0.0 && error("expected dz>0.0, got dz=$(dz)")
    dh < 0.0 && error("expected dh>0.0, got dh=$(dh)")
    mode == "depth" && any(abs.(theta) .>= 90.0) && error("for mode='depth', all theta values must be < 90 degrees in absolute value")

    nz,nh = size(dom)

    # kz
    nzfft = nextprod([2,3,5,7], round(Int, nz*(1 + padz)))
    kn = pi/dz
    dk = kn/nzfft
    kz = dk*[0:div(nzfft,2)+1;]

    # symmetric zero padding for z
    nz_pad_half = div(nzfft - nz, 2)
    nz_pad_rng = (nz_pad_half+1):(nz_pad_half+nz)

    # kh
    nhfft = nextprod([2,3,5,7], round(Int, nh*(1+padh)))
    kh = 2 * fftfreq(nhfft, pi/dh)

    # tan(theta) - used for mode=="depth"
    tant = @. tan(deg2rad(theta))

    # p - used for mode=="time"
    if p === nothing
        p = collect(range(-dz/dh, dz/dh; length=128))
    end

    # conversions
    kz,kh,tant,p = map(x->convert(Array{T,1}, x), (kz,kh,tant,p))
    nzfft,nhfft = map(x->convert(Int64, x), (nzfft,nhfft))
    h0 = T(h0)

    # tapers
    TX = JopTaper(dom, (1,2), (taperz[1],taperh[1]), (taperz[2],taperh[2]))
    TK = JopTaper(JetSpace(Complex{eltype(dom)},div(nzfft,2)+1,mode=="time" ? length(p) : length(tant)), (1,2), (taperkz[1], taperkh[1]), (taperkz[2], taperkh[2]), mode=(:normal,:fftshift))

    JopLn(dom = dom, rng = JetSpace(T, nz, mode == "time" ? length(p) : length(tant)), df! = JopSlantStack_df!, df′! = JopSlantStack_df′!,
        s = (;mode, nzfft, nz_pad_rng, nhfft, kz, kh, tant, p, h0, TX, TK))
end
export JopSlantStack

function JopSlantStack_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; mode, nzfft, nz_pad_rng, nhfft, kz, kh, tant, p, h0, TX, TK, kwargs...) where {T}
    nh, np, dh = size(m,2), mode == "time" ? length(p) : length(tant), abs(kh[2]-kh[1])

    mpad = zeros(T, nzfft, nhfft)
    mpad[nz_pad_rng,1:nh] = TX*m

    M = rfft(mpad)

    compute_kh = mode == "depth" ? slantstack_compute_kh_from_kz : slantstack_compute_kh_from_frequency
    is_out_of_bounds = (ikh_m1, ikh_p1)->(ikh_m1 < 1 || ikh_p1 > nhfft)

    D = zeros(eltype(M), size(M,1), np)
    for ikz = 1:div(nzfft,2)+1, ip = 1:np
        ikh_m1, ikh_p1, _kh = compute_kh(ikz, ip, tant, p, kz, kh, nhfft)

        is_out_of_bounds(ikh_m1, ikh_p1) && continue

        if ikh_m1 == ikh_p1
            D[ikz,ip] = M[ikz,ikh_m1]*exp(-im*kh[ikh_m1]*h0)
            continue
        end

        a_m1 = (kh[ikh_p1] - _kh)/dh
        a_p1 = 1 - a_m1

        d_p1 = M[ikz,ikh_p1]*exp(-im*kh[ikh_p1]*h0)
        d_m1 = M[ikz,ikh_m1]*exp(-im*kh[ikh_m1]*h0)

        D[ikz,ip] = a_m1*d_m1 + a_p1*d_p1
    end

    _d = brfft(TK*D, nzfft, 1)
    d .= _d[nz_pad_rng,1:np] ./ nzfft
end

function JopSlantStack_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; mode, nzfft, nz_pad_rng, nhfft, kz, kh, tant, p, h0, TX, TK, kwargs...) where {T}
    nh, np, dh = size(m,2), mode == "time" ? length(p) : length(tant), abs(kh[2]-kh[1])

    dpad = zeros(T, nzfft, np)
    dpad[nz_pad_rng,:] = d

    D = TK * (rfft(dpad, 1) ./ nzfft)

    compute_kh = mode == "depth" ? slantstack_compute_kh_from_kz : slantstack_compute_kh_from_frequency
    is_out_of_bounds = (ikh_m1, ikh_p1)->(ikh_m1 < 1 || ikh_p1 > nhfft)

    M = zeros(Complex{T}, div(nzfft,2)+1, nhfft)
    for ikz = 1:div(nzfft,2)+1, ip = 1:np
        ikh_m1, ikh_p1, _kh = compute_kh(ikz, ip, tant, p, kz, kh, nhfft)

        is_out_of_bounds(ikh_m1, ikh_p1) && continue

        if ikh_m1 == ikh_p1
            M[ikz,ikh_p1] += D[ikz,ip]*exp(im*kh[ikh_p1]*h0)
            continue
        end

        a_m1 = (kh[ikh_p1] - _kh)/dh
        a_p1 = 1 - a_m1

        m_p1 = D[ikz,ip]*exp(im*kh[ikh_p1]*h0)
        M[ikz,ikh_p1] += a_p1*m_p1

        m_m1 = D[ikz,ip]*exp(im*kh[ikh_m1]*h0)
        M[ikz,ikh_m1] += a_m1*m_m1
    end

    m .= TX * (brfft(M, nzfft)[nz_pad_rng,1:nh])
end

@inline function slantstack_compute_kh_from_kz(ikz::Int64, ip::Int64, tant, p, kz, kh, nhfft)
    _kh = -kz[ikz]*tant[ip]

    ikh_m1 = floor(Int64, _kh/kh[2]) + 1
    ikh_p1 = ceil(Int64, _kh/kh[2]) + 1

    ikh_m1 = ikh_m1 < 1 ? nhfft + ikh_m1 : ikh_m1
    ikh_p1 = ikh_p1 < 1 ? nhfft + ikh_p1 : ikh_p1

    ikh_m1, ikh_p1, _kh
end

@inline function slantstack_compute_kh_from_frequency(iω::Int64, ip::Int64, tant, p, ω, kh, nhfft)
    _kh = p[ip]*ω[iω]

    ikh_m1 = floor(Int64, _kh/kh[2]) + 1
    ikh_p1 = ceil(Int64, _kh/kh[2]) + 1

    ikh_m1 = ikh_m1 < 1 ? nhfft + ikh_m1 : ikh_m1
    ikh_p1 = ikh_p1 < 1 ? nhfft + ikh_p1 : ikh_p1

    ikh_m1, ikh_p1, _kh
end
