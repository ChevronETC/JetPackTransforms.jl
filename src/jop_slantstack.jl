"""
    A = JopSlantStack(dom[; dz=1.0, dh=1.0, h0=0.0, ...])

where `A` is the 2D slant-stack operator mapping for `z-h` to `z-θ` (depth mode) or `t-h` to `tau-p` (time mode).
The domain of the operator is `nz` x `nh` with precision T, `dz` is the depth spacing (or time interval),
`dh` is the offset spacing, and `h0` is the origin of the offset axis.  The additional named optional arguments
along with their default values are,

* `mode="depth` - choose between "depth" and "time" to specify if the input domain is `z-h` or `t-h`
* `theta=-45:1.0:45` - range of incidence angles used when `mode="depth"`
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

* `mode="depth` - choose between "depth" and "time" to specify if the input domain is `z-h` or `t-h`
* `theta=-45:1.0:45` - range of incidence angles used when `mode="depth"`
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
            if abs(shift) < nz
                WaveFD.shiftforward!(WaveFD.shiftfilter(shift), holder, @view(mtap[:,ih]))
                d[:,ip] .+= holder
            end
        end
    end
    d
end

function JopSlantStackShiftSum_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; dz, h, p, TZ, kwargs...) where {T}
    nz, nh, np = size(m,1), size(m,2), length(p)

    mtap = zeros(T, nz, nh)
    @threads for ih = 1:nh
        holder = zeros(T, nz)
        for ip = 1:np
            shift = + h[ih] * p[ip] / dz
            if abs(shift) < nz
                WaveFD.shiftadjoint!(WaveFD.shiftfilter(shift), holder, @view(d[:,ip]))
                @views mtap[:,ih] .+= holder
            end
        end
    end
    m = TZ' * mtap
    m
end




"""
    A = JopSlantStackShiftSum3D(dom[; dz=1.0, ...])

where `A` is the 3D slant-stack operator mapping for `z-hx-hy` to `z-θ-ϕ` (depth mode) or `t-hx-hy` to `tau-px-py` (time mode).
The slant stacking is performed directly on the input domain by shifting and summing along the offset axis.
The domain of the operator is `nz` x `nh` (for irregular) or `nz` x `nhx` x `nhy` (for regular) with precision T, `dz` is the depth spacing (or time interval),
`hx` and `hy`, if provided, are the arrays of offsets (can be irregular). If not provided, a regular grid is assumed with same sampling as dz.
The additional named optional arguments along with their default values are,

* `mode="depth` - choose between "depth" and "time" to specify if the input domain is `z-h` or `t-h`
* `theta=-45:1.0:45` - range of incidence angles used when `mode="depth"`
* `phi=0:45:135` - range of image azimuth angles used when `mode="depth"`
* `dip=0.0` - geologic dip used when `mode="depth"`
* `azimuth=0.0` - geologic azimuth used when `mode="depth"`
* `px=range(-dz/max(diff(hx)),dz/max(diff(hx)),64)` - ray parameter sampling used when `mode="time"`
* `py=range(-dz/max(diff(hy)),dz/max(diff(hy)),64)` - ray parameter sampling used when `mode="time"`
* `taperz=(0,0)` - beginning and end taper (fractional) in the z-direction (or t-direction) before shifting and summing

# Notes

* If the mode is "time", then `dz` is the time sampling interval, `taperz` is the taper for the time dimension.
* For mode="depth", typically theta needs to cover both positive and negative angles and must be < 90 degrees.
* For mode="depth", the incidence angle depends on the geologic dip and azimuth (see 3-D Seismic Imaging by Prof. Biondo Biondi, Chapter 6). If not provided, this dependence is ignored.
"""
function JopSlantStackShiftSum3D(
        dom::JetAbstractSpace{T};
        theta = collect(-45.0:1.0:45.0),
        phi = collect(0.0:45.0:135.0),
        px = nothing,
        py = nothing,
        dz = 1.0,
        hx = nothing,
        hy = nothing,
        taperz = (0,0),
        dip = 0.0,
        azimuth = 0.0,
        mode = "depth") where {T}
    mode ∈ ("depth", "time") || error("expected mode to be either 'depth' or 'time', got mode=$(mode)")
    dz < 0.0 && error("expected dz>0.0, got dz=$(dz)")
    if mode == "time" || all(iszero, dip)
        dip, azimuth = 0, 0
    end
    (typeof(dip) != typeof(azimuth)) && error("dip and azimuth should be of the same type")
    isa(dip, Array) && (length(dip) != length(azimuth)) && error("if dip is an array, its length=$(length(dip)) should match the length of azimuth=$(length(azimuth))")
    isa(dip, Array) && (length(dip) != size(dom, 1)) && error("if dip is an array, its length=$(length(dip)) should match the first dimension of the domain=$(size(dom, 1))")

    nd = ndims(dom) # if nd = 2, the offsets will be considered as totally irregular
    nd ∉ (2,3) && error("expected domain to be either 2D or 3D, got ndims=$(nd)")
    (nd == 2) && (hx === nothing || hy === nothing) && error("for 2D irregular (non-lattice) offsets, hx and hy should be provided") 
    ndy = (nd == 2 ? 2 : 3)
    hx === nothing && (hx = collect(0.0:dz:(size(dom,2)-1)*dz))
    hy === nothing && (hy = collect(0.0:dz:(size(dom,3)-1)*dz))
    length(hx) != size(dom,2) && error("length of hx=$(length(hx)) must match the number of x-offsets in the domain=$(size(dom,2))")
    length(hy) != size(dom,ndy) && error("length of hy=$(length(hy)) must match the number of y-offsets in the domain=$(size(dom,ndy))")
    dhxmax = maximum(diff(sort(hx)))
    dhymax = maximum(diff(sort(hy)))
    mode == "depth" && any(abs.(theta) .>= 90.0) && error("for mode='depth', theta values must be < 90 degrees in absolute value")
    mode == "depth" && (any(phi .> 180.0) || any(phi .< 0.0)) && error("for mode='depth', phi values must be between 0 and 180 degrees")

    nz = size(dom, 1)
    nhx = size(dom, 2)
    nhy = size(dom, ndy)

    # tan(theta), cos(phi) - used for mode=="depth"
    tant = @. tan(deg2rad(theta))
    phi = @. deg2rad(phi)

    # p - used for both modes
    if mode == "depth"
        px = - tant
        py = phi
    elseif px === nothing || py === nothing
        (px === nothing) && (px = collect(range(-dz/dhxmax, dz/dhxmax; length=64)))
        (py === nothing) && (py = collect(range(-dz/dhymax, dz/dhymax; length=64)))
    end

    # conversions
    hx,hy,px,py = map(x->convert(Array{T,1}, x), (hx,hy,px,py))
    if !isa(dip, Array)
        dip = T(dip)
        azimuth = T(azimuth)
    else
        dip, azimuth = map(x->convert(Array{T,1}, x), (dip, azimuth))
    end
    dz = T(dz)

    # taper
    TZ = JopTaper(dom, (1,), (taperz[1],), (taperz[2],))

    JopLn(dom = dom, rng = JetSpace(T, nz, length(px), length(py)), df! = JopSlantStackShiftSum3D_df!, df′! = JopSlantStackShiftSum3D_df′!,
        s = (;mode, dz, hx, hy, px, py, dip, azimuth, TZ))
end
export JopSlantStackShiftSum3D

function JopSlantStackShiftSum3D_df!(d::AbstractArray{T,3}, m::AbstractArray{T,2}; dip, azimuth, mode, dz, hx, hy, px, py, TZ, kwargs...) where {T}
    if isa(dip, Array)
        JopSlantStackShiftSum3D_df_vector!(d, m; dip, azimuth, mode, dz, hx, hy, px, py, TZ)
    else
        JopSlantStackShiftSum3D_df_scalar!(d, m; dip, azimuth, mode, dz, hx, hy, px, py, TZ)
    end
end

function JopSlantStackShiftSum3D_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,3}; dip, azimuth, mode, dz, hx, hy, px, py, TZ, kwargs...) where {T}
    if isa(dip, Array)
        JopSlantStackShiftSum3D_df_vector′!(m, d; dip, azimuth, mode, dz, hx, hy, px, py, TZ, kwargs...)
    else
        JopSlantStackShiftSum3D_df_scalar′!(m, d; dip, azimuth, mode, dz, hx, hy, px, py, TZ, kwargs...)
    end
end

function JopSlantStackShiftSum3D_df_scalar!(d::AbstractArray{T,3}, m::AbstractArray{T,2}; dip::Real, azimuth::Real, mode, dz, hx, hy, px, py, TZ) where {T}
    nz, nh, npx, npy = size(m,1), size(m,2), length(px), length(py)

    if mode == "time"
        compute_shift = slantstack_shift_px_py
    elseif dip == 0
        compute_shift = slantstack_shift_theta_phi
    else
        compute_shift = slantstack_shift_theta_phi_geologic
    end

    d .= 0
    mtap = TZ * m
    @threads for ipx = 1:npx
        holder = zeros(T, nz)
        for ipy = 1:npy
            num = tan(dip)^2 *sin(azimuth - py[ipy])*cos(azimuth - py[ipy])
            denom = sqrt(1 + (tan(dip)*sin(azimuth - py[ipy]))^2)
            for ih = 1:nh
                shift = compute_shift(1, ih, ih, ipx, ipy, hx, hy, px, py, dz, num, denom)
                if abs(shift) < nz
                    WaveFD.shiftforward!(WaveFD.shiftfilter(shift), holder, @view(mtap[:,ih]))
                    d[:,ipx,ipy] .+= holder
                end
            end
        end
    end
    d
end

function JopSlantStackShiftSum3D_df_vector!(d::AbstractArray{T,3}, m::AbstractArray{T,2}; dip::AbstractArray{T,1}, azimuth::AbstractArray{T,1}, mode, dz, hx, hy, px, py, TZ) where {T}
    nz, nh, npx, npy = size(m,1), size(m,2), length(px), length(py)

    compute_shift = slantstack_shift_theta_phi_geologic
    
    d .= 0
    mtap = TZ * m
    @threads for ipx = 1:npx
        holder = zeros(T, nz)
        for ipy = 1:npy
            num = tan.(dip).^2 .* sin.(azimuth .- py[ipy]) .* cos.(azimuth .- py[ipy])
            denom = sqrt.(1 .+ (tan.(dip) .* sin.(azimuth .- py[ipy])).^2)
            for ih = 1:nh
                for iz = 1:nz
                    shift = compute_shift(iz, ih, ih, ipx, ipy, hx, hy, px, py, dz, num, denom)
                    if abs(shift) < nz 
                        WaveFD.shiftforward!(WaveFD.shiftfilter(shift), holder, @view(mtap[:,ih]))
                        d[iz,ipx,ipy] += holder[iz]
                    end
                end
            end
        end
    end
    d
end

function JopSlantStackShiftSum3D_df_scalar′!(m::AbstractArray{T,2}, d::AbstractArray{T,3}; dip::Real, azimuth::Real, mode, dz, hx, hy, px, py, TZ, kwargs...) where {T}
    nz, nh, npx, npy = size(m,1), size(m,2), length(px), length(py)

    if mode == "time"
        compute_shift = slantstack_shift_px_py
    elseif dip == 0
        compute_shift = slantstack_shift_theta_phi
    else
        compute_shift = slantstack_shift_theta_phi_geologic
    end

    mtap = zeros(T, nz, nh)
    @threads for ih = 1:nh
        acc = zeros(T, nz)
        holder = zeros(T, nz)
        for ipx = 1:npx
            for ipy = 1:npy
                num = tan(dip)^2 *sin(azimuth - py[ipy])*cos(azimuth - py[ipy])
                denom = sqrt(1 + (tan(dip)*sin(azimuth - py[ipy]))^2)
                shift = compute_shift(1, ih, ih, ipx, ipy, hx, hy, px, py, dz, num, denom)
                if abs(shift) < nz
                    WaveFD.shiftadjoint!(WaveFD.shiftfilter(shift), holder, @view(d[:,ipx,ipy]))
                    acc .+= holder
                end
            end
        end
        mtap[:,ih] .= acc
    end
    m = TZ' * mtap
    m
end

function JopSlantStackShiftSum3D_df_vector′!(m::AbstractArray{T,2}, d::AbstractArray{T,3}; dip::AbstractArray{T,1}, azimuth::AbstractArray{T,1}, mode, dz, hx, hy, px, py, TZ, kwargs...) where {T}
    nz, nh, npx, npy = size(m,1), size(m,2), length(px), length(py)

    compute_shift = slantstack_shift_theta_phi_geologic

    mtap = zeros(T, nz, nh)
    @threads for ih = 1:nh
        acc = zeros(T, nz)
        holder = zeros(T, nz)
        for ipx = 1:npx
            for ipy = 1:npy
                num = tan.(dip).^2 .* sin.(azimuth .- py[ipy]) .* cos.(azimuth .- py[ipy])
                denom = sqrt.(1 .+ (tan.(dip) .* sin.(azimuth .- py[ipy])).^2)
                for iz = 1:nz
                    shift = compute_shift(iz, ih, ih, ipx, ipy, hx, hy, px, py, dz, num, denom)
                    if abs(shift) < nz
                        fill!(holder, zero(T))
                        holder[iz] = d[iz,ipx,ipy]
                        WaveFD.shiftadjoint!(WaveFD.shiftfilter(shift), acc, holder)
                        mtap[:,ih] .+= acc
                    end
                end
            end
        end
    end
    m = TZ' * mtap
    m
end

@inline function slantstack_shift_px_py(iz::Int64, ihx::Int64, ihy::Int64, ipx::Int64, ipy::Int64, hx, hy, px, py, dz, num::Real, denom::Real)
    # for time domain, the shift is given by (hx*px + hy*py) / dz
    shift = + (hx[ihx] * px[ipx] + hy[ihy] * py[ipy]) / dz
    shift
end

@inline function slantstack_shift_theta_phi(iz::Int64, ihx::Int64, ihy::Int64, ipx::Int64, ipy::Int64, hx, hy, px, py, dz, num::Real, denom::Real)
    # for depth domain ignoring geologic dip, the shift is given by (-hx*tan(θ)*cos(ϕ) - hy*tan(θ)*sin(ϕ)) / dz
    shift = + (hx[ihx] * px[ipx] * cos(py[ipy]) + hy[ihy] * px[ipx] * sin(py[ipy])) / dz
    shift
end

@inline function slantstack_shift_theta_phi_geologic(iz::Int64, ihx::Int64, ihy::Int64, ipx::Int64, ipy::Int64, hx, hy, px, py, dz, num::Real, denom::Real)
    # for depth domain accounting for geologic dip, the shift is given by (-hx*tan(θ)*(cos(ϕ)/denom + sin(ϕ)*num/denom) - hy*tan(θ)*(sin(ϕ)/denom - cos(ϕ)*num/denom)) / dz
    shift = + hx[ihx] * px[ipx] * (cos(py[ipy]) / denom + sin(py[ipy]) * num / denom) + hy[ihy] * px[ipx] * (sin(py[ipy]) / denom - cos(py[ipy]) * num / denom)
    shift / dz
end

@inline function slantstack_shift_theta_phi_geologic(iz::Int64, ihx::Int64, ihy::Int64, ipx::Int64, ipy::Int64, hx, hy, px, py, dz, num::AbstractArray{T,1}, denom::AbstractArray{T,1}) where {T}
    # for depth domain accounting for geologic dip, the shift is given by (-hx*tan(θ)*(cos(ϕ)/denom + sin(ϕ)*num/denom) - hy*tan(θ)*(sin(ϕ)/denom - cos(ϕ)*num/denom)) / dz
    shift = + hx[ihx] * px[ipx] * (cos(py[ipy]) / denom[iz] + sin(py[ipy]) * num[iz] / denom[iz]) + hy[ihy] * px[ipx] * (sin(py[ipy]) / denom[iz] - cos(py[ipy]) * num[iz] / denom[iz])
    shift / dz
end