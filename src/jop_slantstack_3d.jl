"""
    A = JopSlantStack3D(dom[; dz=1.0, dhx=1.0, dhy=1.0, hx0=0.0, hy0=0.0, ...])

where `A` is the 3D slant-stack operator mapping for `z-hx-hy` to `z-θ-ϕ` (depth mode) or `t-hx-hy` to `tau-px-py` (time mode).
The domain of the operator is `nz` x `nhx` x `nhy` with precision T, `dz` is the depth spacing (or time interval),
`dhx` and `dhy` are the offset spacings, and `hx0` and `hy0` are the origins of the offset axes. The additional named optional arguments
along with their default values are,

* `mode="depth` - choose between "depth" and "time" to specify if the input domain is `z-hx-hy` or `t-hx-hy`.
* `theta=-60:1.0:60` - range of incidence angles used when `mode="depth"`.
* `phi=0:45.0:315` - range of azimuth angles used when `mode="depth"`.
* `px=range(-dz/dhx,dz/dhx,128)` - ray parameter sampling used when `mode="time"`
* `py=range(-dz/dhy,dz/dhy,128)` - ray parameter sampling used when `mode="time"`
* `padz=0.0,padhx=0.0,padhy=0.0` - fractional padding in depth and offset to apply before applying the Fourier transform
* `taperz=(0,0)` - beginning and end taper (fractional) in the z-direction (or t-direction) before transforming from `z-hx-hy` to `kz-khx-khy` or `t-hx-hy` to `f-khx-khy`
* `taperhx=(0,0)` - beginning and end taper (fractional) in the hx-direction before transforming from `z-hx-hy` to `kz-khx-khy` or `t-hx-hy` to `f-khx-khy`
* `taperhy=(0,0)` - beginning and end taper (fractional) in the hy-direction before transforming from `z-hx-hy` to `kz-khx-khy` or `t-hx-hy` to `f-khx-khy`
* `taperkz=(0,0)` - beginning and end taper (fractional) in the kz-direction (or frequency) before transforming from `kz-theta-phi` to `z-hx-hy` or `f-px-py` to `t-hx-hy`
* `taperkhx=(0,0)` - beginning and end taper (fractional) in the khx-direction before transforming from `kz-theta-phi` to `z-hx-hy` or `f-px-py` to `t-hx-hy`
* `taperkhy=(0,0)` - beginning and end taper (fractional) in the khy-direction before transforming from `f-px-py` to `t-hx-hy`

# Notes

* If the mode is "time", then `padz` is the padding for the time dimension, `dz` is the time sampling interval, `taperz` is the taper for the time dimension and `tapkerkz` is the taper for frequency.
* For mode="depth", typically theta needs to cover both positive and negative angles and must be < 90 degrees, phi needs to cover the full 360 degree range.
* For mode="depth", no taper is applied to azimuth angle 
"""
function JopSlantStack3D(
        dom::JetAbstractSpace{T};
        theta = collect(-60.0:1.0:60.0),
        phi = collect(0.0:45.0:315.0),
        px = nothing,
        py = nothing,
        dz = 1.0,
        dhx = 1.0,
        dhy = 1.0,
        hx0 = 0.0,
        hy0 = 0.0,
        padz = 0.0,
        padhx = 0.0,
        padhy = 0.0,
        taperz = (0,0),
        taperhx = (0,0),
        taperhy = (0,0),
        taperkz = (0,0),
        taperkhx = (0,0),
        taperkhy = (0,0),
        mode = "depth") where {T}
    mode ∈ ("depth", "time") || error("expected mode to be either 'depth' or 'time', got mode=$(mode)")
    dz < 0.0 && error("expected dz>0.0, got dz=$(dz)")
    dhx < 0.0 && error("expected dhx>0.0, got dhx=$(dhx)")
    dhy < 0.0 && error("expected dhy>0.0, got dhy=$(dhy)")
    mode == "depth" && any(abs.(theta) .>= 90.0) && error("for mode='depth', all theta values must be < 90 degrees in absolute value")

    nz,nhx,nhy = size(dom)

    # kz
    nzfft = nextprod([2,3,5,7], round(Int, nz*(1 + padz)))
    kn = pi/dz
    dk = kn/nzfft
    kz = dk*[0:div(nzfft,2)+1;]

    # khx
    nhxfft = nextprod([2,3,5,7], round(Int, nhx*(1+padhx)))
    khx = 2 * fftfreq(nhxfft, pi/dhx)

    # khy
    nhyfft = nextprod([2,3,5,7], round(Int, nhy*(1+padhy)))
    khy = 2 * fftfreq(nhyfft, pi/dhy)

    # tan(theta) - used for mode=="depth"
    tant = @. tan(deg2rad(theta))

    # p - used for mode=="time"
    if px === nothing
        px = collect(range(-dz/dhx, dz/dhx; length=128))
    end
    if py === nothing
        py = collect(range(-dz/dhy, dz/dhy; length=128))
    end

    # conversions
    kz,khx,khy,tant,px,py = map(x->convert(Array{T,1}, x), (kz,khx,khy,tant,px,py))
    nzfft,nhxfft,nhyfft = map(x->convert(Int64, x), (nzfft,nhxfft,nhyfft))
    hx0 = T(hx0)
    hy0 = T(hy0)

    # tapers
    TX = JopTaper(dom, (1,2,3), (taperz[1],taperhx[1],taperhy[1]), (taperz[2],taperhx[2],taperhy[2]))
    TK = mode == "time" ? JopTaper(JetSpace(Complex{eltype(dom)},div(nzfft,2)+1, length(px), length(py)), (1,2,3), (taperkz[1], taperkhx[1], taperkhy[1]), (taperkz[2], taperkhx[2], taperkhy[2]), mode=(:normal,:fftshift))

    JopLn(dom = dom, rng = JetSpace(T, nz, mode == "time" ? length(p) : length(tant)), df! = JopSlantStack_df!, df′! = JopSlantStack_df′!,
        s = (;mode, nzfft, nhfft, kz, kh, tant, p, h0, TX, TK))
end
export JopSlantStack

function JopSlantStack_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; mode, nzfft, nhfft, kz, kh, tant, p, h0, TX, TK, kwargs...) where {T}
    nh, np, dh = size(m,2), mode == "time" ? length(p) : length(tant), abs(kh[2]-kh[1])
    nz = size(d, 1)

    mpad = zeros(T, nzfft, nhfft)
    mpad[1:nz,1:nh] = TX*m

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
    d .= _d[1:nz,1:np] ./ nzfft
end

function JopSlantStack_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; mode, nzfft, nhfft, kz, kh, tant, p, h0, TX, TK, kwargs...) where {T}
    nh, np, dh = size(m,2), mode == "time" ? length(p) : length(tant), abs(kh[2]-kh[1])
    nz = size(d, 1)

    dpad = zeros(T, nzfft, np)
    dpad[1:nz,:] = d

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

    m .= TX * (brfft(M, nzfft)[1:nz,1:nh])
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
