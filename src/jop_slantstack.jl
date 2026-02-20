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
* `taperz=(0,0)` - beginning and end taper (fractional) in the z-direction (or t-direction) before transforming from `z-h` to `kz-kh` or `t-h` to `f-kh`
* `taperh=(0,0)` - beginning and end taper (fractional) in the h-direction before transforming from `z-h` to `kz-kh` or `t-h` to `f-kh`
* `taperkz=(0,0)` - beginning and end taper (fractional) in the kz-direction (or frequency) before transforming from `kz-theta` to `z-h` or `f-p` to `t-h`
* `taperkh=(0,0)` - beginning and end taper (fractional) in the kh-direction before transforming from `kz-theta` to `z-h` or `f-p` to `t-h`

# Notes

* If the mode is "time", then `padz` is the padding for the time dimension, `dz` is the time sampling interval, `taperz` is the taper for the time dimension and `tapkerkz` is the taper for frequency.
* For mode="depth", typically theta needs to cover both positive and negative angles and must be < 90 degrees.
"""
function JopSlantStack(
        dom::JetAbstractSpace{T};
        theta = collect(-45.0:1.0:45.0),
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
    kz = rfftfreq(nzfft, 2*pi / dz)

    # kh
    nhfft = nextprod([2,3,5,7], round(Int, nh*(1+padh)))
    kh = fftfreq(nhfft, 2*pi / dh)

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

"""
    A = JopSlantStackShiftSum(dom[; dz=1.0, ...])

where `A` is the 2D slant-stack operator mapping for `z-h` to `z-θ` (depth mode) or `t-h` to `tau-p` (time mode).
The slant stacking is performed directly on the input domain by shifting and summing along the offset axis.
The domain of the operator is `nz` x `nh` with precision T, `dz` is the depth spacing (or time interval),
`h`, if provided, is the array of offsets (can be irregular). If not provided, a regular grid is assumed with same sampling as dz.
The additional named optional arguments along with their default values are,

* `mode="depth` - choose between "depth" and "time" to specify if the input domain is `z-h` or `t-h`.
* `theta=-60:1.0:60` - range of incidence angles used when `mode="depth"`.
* `p=range(-dz/max(diff(h)),dz/max(diff(h)),128)` - ray parameter sampling used when `mode="time"`
* `taperz=(0,0)` - beginning and end taper (fractional) in the z-direction (or t-direction) before shifting and summing

# Notes

* If the mode is "time", then `dz` is the time sampling interval, `taperz` is the taper for the time dimension.
* For mode="depth", typically theta needs to cover both positive and negative angles and must be < 90 degrees.
"""
function JopSlantStackShiftSum(
        dom::JetAbstractSpace{T};
        theta = collect(-45.0:1.0:45.0),
        p = nothing,
        dz = 1.0,
        h = nothing,
        taperz = (0,0),
        mode = "depth") where {T}
    mode ∈ ("depth", "time") || error("expected mode to be either 'depth' or 'time', got mode=$(mode)")
    dz < 0.0 && error("expected dz>0.0, got dz=$(dz)")
    h === nothing && (h = collect(0.0:dz:(size(dom,2)-1)*dz))
    length(h) != size(dom,2) && error("length of h=$(length(h)) must match the number of offsets in the domain=$(size(dom,2))")
    dhmax = maximum(diff(sort(h)))
    mode == "depth" && any(abs.(theta) .>= 90.0) && error("for mode='depth', all theta values must be < 90 degrees in absolute value")

    nz,nh = size(dom)

    # tan(theta) - used for mode=="depth"
    tant = @. tan(deg2rad(theta))

    # p - used for both modes
    if mode == "depth"
        p = - tant
    elseif p === nothing
        p = collect(range(-dz/dhmax, dz/dhmax; length=128))
    end

    # conversions
    h,p = map(x->convert(Array{T,1}, x), (h,p))
    dz = T(dz)

    # taper
    TZ = JopTaper(dom, (1,), (taperz[1],), (taperz[2],))

    JopLn(dom = dom, rng = JetSpace(T, nz, length(p)), df! = JopSlantStackShiftSum_df!, df′! = JopSlantStackShiftSum_df′!,
        s = (;dz, h, p, TZ))
end
export JopSlantStackShiftSum

function JopSlantStackShiftSum_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; dz, h, p, TZ, kwargs...) where {T}
    nz, nh, np = size(m,1), size(m,2), length(p)

    d .= 0
    mtap = TZ * m
    @threads for ip = 1:np
        holder = zeros(T, nz)
        for ih = 1:nh
            shift = + h[ih] * p[ip] / dz
            WaveFD.shiftforward!(WaveFD.shiftfilter(shift), holder, @view(mtap[:,ih]))
            d[:,ip] .+= holder
        end
    end
    d
end

function JopSlantStackShiftSum_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; dz, h, p, TZ, kwargs...) where {T}
    nz, nh, np = size(m,1), size(m,2), length(p)

    @show length(h), length(p)

    mtap = zeros(T, nz, nh)
    @threads for ih = 1:nh
        acc = zeros(T, nz)
        holder = zeros(T, nz)
        for ip = 1:np
            shift = + h[ih] * p[ip] / dz
            WaveFD.shiftadjoint!(WaveFD.shiftfilter(shift), holder, @view(d[:,ip]))
            acc .+= holder
        end
        mtap[:,ih] .= acc
    end
    m = TZ' * mtap
    m
end