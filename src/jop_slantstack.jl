"""
    A = JopSlantStack(dom[; dz=1.0, dh=1.0, h0=0.0, ...])

where `A` is the 2D slant-stack operator mapping for `z-h` to `z-θ` (depth mode) or `t-h` to `tau-p` (time mode).
The domain of the operator is `nz` x `nh` with precision T, `dz` is the depth spacing (or time interval),
`dh` is the (half) offset spacing, and `h0` is the origin of the (half) offset axis.  The additional named optional arguments
along with their default values are,

* `mode="depth` - choose between "depth" and "time" to specify if the input domain is `z-h` or `t-h`
* `theta=-45:1.0:45` - range of incidence angles used when `mode="depth"`
* `p=range(-dz/dh,dz/dh,128)` - ray parameter sampling used when `mode="time"`
* `padz=0.0,padh=0.0` - fractional padding in depth and offset to apply before applying the Fourier transform
* `taperz=(0,0)` - beginning and end taper (fractional) in the z-direction (or t-direction) before transforming from `z-h` to `kz-kh` or `t-h` to `f-kh`
* `taperh=(0,0)` - beginning and end taper (fractional) in the h-direction before transforming from `z-h` to `kz-kh` or `t-h` to `f-kh`
* `taperkz=(0,0)` - beginning and end taper (fractional) in the kz-direction (or frequency) before sampling
* `taperkh=(0,0)` - beginning and end taper (fractional) in the kh-direction before sampling

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
    TK = JopTaper(JetSpace(Complex{eltype(dom)},div(nzfft,2)+1, nhfft), (1,2), (taperkz[1], taperkh[1]), (taperkz[2], taperkh[2]), mode=(:normal,:fftshift))

    JopLn(dom = dom, rng = JetSpace(T, nz, mode == "time" ? length(p) : length(tant)), df! = JopSlantStack_df!, df′! = JopSlantStack_df′!,
        s = (;mode, nzfft, nhfft, kz, kh, tant, p, h0, TX, TK))
end
export JopSlantStack

function JopSlantStack_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; mode, nzfft, nhfft, kz, kh, tant, p, h0, TX, TK, kwargs...) where {T}
    nh, np, dh = size(m,2), mode == "time" ? length(p) : length(tant), abs(kh[2]-kh[1])
    nz = size(d, 1)

    mpad = zeros(T, nzfft, nhfft)
    mpad[1:nz,1:nh] = TX*m

    M = TK*rfft(mpad)

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

    _d = brfft(D, nzfft, 1)
    d .= _d[1:nz,1:np] ./ nzfft
end

function JopSlantStack_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; mode, nzfft, nhfft, kz, kh, tant, p, h0, TX, TK, kwargs...) where {T}
    nh, np, dh = size(m,2), mode == "time" ? length(p) : length(tant), abs(kh[2]-kh[1])
    nz = size(d, 1)

    dpad = zeros(T, nzfft, np)
    dpad[1:nz,:] = d

    D = (rfft(dpad, 1) ./ nzfft)

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

    m .= TX * (brfft(TK * M, nzfft)[1:nz,1:nh])
end

@inline function slantstack_compute_kh_from_kz(ikz::Int64, ip::Int64, tant, p, kz, kh, nhfft)
    _kh = -kz[ikz]*tant[ip]

    # check if wavenumber is outside Nyquist
    ((_kh > kh[div(nhfft+1,2)]) || (_kh < kh[div(nhfft+1,2)+1])) && return -1, -1, _kh

    ikh_m1 = floor(Int64, _kh/kh[2]) + 1
    ikh_p1 = ceil(Int64, _kh/kh[2]) + 1

    ikh_m1 = ikh_m1 < 1 ? nhfft + ikh_m1 : ikh_m1
    ikh_p1 = ikh_p1 < 1 ? nhfft + ikh_p1 : ikh_p1

    ikh_m1, ikh_p1, _kh
end

@inline function slantstack_compute_kh_from_frequency(iω::Int64, ip::Int64, tant, p, ω, kh, nhfft)
    _kh = p[ip]*ω[iω]

    # check if wavenumber is outside Nyquist
    ((_kh > kh[div(nhfft+1,2)]) || (_kh < kh[div(nhfft+1,2)+1])) && return -1, -1, _kh

    ikh_m1 = floor(Int64, _kh/kh[2]) + 1
    ikh_p1 = ceil(Int64, _kh/kh[2]) + 1

    ikh_m1 = ikh_m1 < 1 ? nhfft + ikh_m1 : ikh_m1
    ikh_p1 = ikh_p1 < 1 ? nhfft + ikh_p1 : ikh_p1

    ikh_m1, ikh_p1, _kh
end

"""
    A = JopSlantStack3D(dom[; dz=1.0, dhx=1.0, dhy=1.0, hx0=0.0, hy0=0.0, ...])

where `A` is the 3D slant-stack operator mapping for `z-hx-hy` to `z-θ-ϕ` (depth mode) or `t-hx-hy` to `tau-px-py` (time mode).
The domain of the operator is typically `nz` x `nhx` x `nhy` with precision T, `dz` is the depth spacing (or time interval),
`dhx` and `dhy` are the (half) offset spacings, and `hx0` and `hy0` are the origins of the (half) offset axes.
The domain can also be `nz` x `ny` x `nx` x `nhx` x `nhy` where the `y` and `x` dimensions are passive.
The additional named optional arguments along with their default values are,

* `mode="depth` - choose between "depth" and "time" to specify if the input domain is `z-hx-hy` or `t-hx-hy`.
* `theta=-45:1.0:45` - range of incidence angles used when `mode="depth"`.
* `phi=0:45.0:135` - range of azimuth angles used when `mode="depth"`.
* `dip=0.0` - geologic dip (degrees) used when `mode="depth"`
* `azimuth=0.0` - geologic azimuth (degrees) used when `mode="depth"`
* `px=range(-dz/dhx,dz/dhx,64)` - ray parameter sampling used when `mode="time"`
* `py=range(-dz/dhy,dz/dhy,64)` - ray parameter sampling used when `mode="time"`
* `padz=0.0,padhx=0.0,padhy=0.0` - fractional padding in depth and offset to apply before applying the Fourier transform
* `taperz=(0,0)` - beginning and end taper (fractional) in the z-direction (or t-direction) before transforming from `z-hx-hy` to `kz-khx-khy` or `t-hx-hy` to `f-khx-khy`
* `taperhx=(0,0)` - beginning and end taper (fractional) in the hx-direction before transforming from `z-hx-hy` to `kz-khx-khy` or `t-hx-hy` to `f-khx-khy`
* `taperhy=(0,0)` - beginning and end taper (fractional) in the hy-direction before transforming from `z-hx-hy` to `kz-khx-khy` or `t-hx-hy` to `f-khx-khy`
* `taperkz=(0,0)` - beginning and end taper (fractional) in the kz-direction (or frequency) before sampling
* `taperkhx=(0,0)` - beginning and end taper (fractional) in the khx-direction before sampling
* `taperkhy=(0,0)` - beginning and end taper (fractional) in the khy-direction before sampling

# Notes

* If the mode is "time", then `padz` is the padding for the time dimension, `dz` is the time sampling interval, `taperz` is the taper for the time dimension and `tapkerkz` is the taper for frequency.
* For mode="depth", typically `theta` needs to cover both positive and negative angles and must be < 90 degrees.
* For mode="depth", the incidence angle depends on the geologic `dip` and `azimuth` (see 3-D Seismic Imaging by Prof. Biondo Biondi, Chapter 6). If not provided, this dependency is ignored.
* Typically, geologic `dip` ∈ [0, 90] degrees and `azimuth` ∈ [0, 360] degrees, which is different from `theta` and `phi` ranges.
* For mode="depth", only scalar `dip` and `azimuth` are supported, which means the same geologic angle is applied across the entire depth. This is a simplification and may not be accurate for complex geologies.
"""
function JopSlantStack3D(
        dom::JetAbstractSpace{T};
        theta = collect(-45.0:1.0:45.0),
        phi = collect(0.0:45.0:135.0),
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
        dip = 0.0,
        azimuth = 0.0,
        mode = "depth") where {T}
    mode ∈ ("depth", "time") || error("expected mode to be either 'depth' or 'time', got mode=$(mode)")
    dz < 0.0 && error("expected dz>0.0, got dz=$(dz)")
    dhx < 0.0 && error("expected dhx>0.0, got dhx=$(dhx)")
    dhy < 0.0 && error("expected dhy>0.0, got dhy=$(dhy)")
    mode == "depth" && any(abs.(theta) .>= 90.0) && error("for mode='depth', all theta values must be < 90 degrees in absolute value")

    ndim = length(size(dom))
    if ndim == 3
        nz,nhx,nhy = size(dom)
        active_dims = (1,2,3)
    elseif ndim == 5
        nz,ny,nx,nhx,nhy = size(dom)
        active_dims = (1,4,5)
    else
        error("unsupported domain size: $(size(dom))")
    end

    # kz
    nzfft = nextprod([2,3,5,7], round(Int, nz*(1 + padz)))
    kz = rfftfreq(nzfft, 2*pi / dz)

    # khx
    nhxfft = nextprod([2,3,5,7], round(Int, nhx*(1+padhx)))
    khx = fftfreq(nhxfft, 2*pi / dhx)

    # khy
    nhyfft = nextprod([2,3,5,7], round(Int, nhy*(1+padhy)))
    khy = fftfreq(nhyfft, 2*pi / dhy)

    # tan(theta), phi - used for mode=="depth"
    tant = @. tan(deg2rad(theta))
    phi = @. deg2rad(phi)

    dip = deg2rad(dip)
    azimuth = deg2rad(azimuth)

    # p - used for both modes
    if mode == "depth"
        px = - tant
        py = phi
    elseif px === nothing || py === nothing
        (px === nothing) && (px = collect(range(-dz/dhx, dz/dhx; length=64)))
        (py === nothing) && (py = collect(range(-dz/dhy, dz/dhy; length=64)))
    end

    # conversions
    kz,khx,khy,tant,phi,px,py = map(x->convert(Array{T,1}, x), (kz,khx,khy,tant,phi,px,py))
    nzfft,nhxfft,nhyfft = map(x->convert(Int64, x), (nzfft,nhxfft,nhyfft))
    hx0 = T(hx0)
    hy0 = T(hy0)
    dip = T(dip)
    azimuth = T(azimuth)

    # tapers
    TX = JopTaper(dom, active_dims, (taperz[1],taperhx[1],taperhy[1]), (taperz[2],taperhx[2],taperhy[2]))
    TK = JopTaper(ndim == 3 ? JetSpace(Complex{eltype(dom)}, div(nzfft,2)+1, nhxfft, nhyfft) : JetSpace(Complex{eltype(dom)}, div(nzfft,2)+1, ny, nx, nhxfft, nhyfft), active_dims, (taperkz[1], taperkhx[1], taperkhy[1]), (taperkz[2], taperkhx[2], taperkhy[2]), mode=(:normal,:fftshift,:fftshift))

    JopLn(dom = dom, rng = ndim == 3 ? JetSpace(T, nz, length(px), length(py)) : JetSpace(T, nz, ny, nx, length(px), length(py)), df! = JopSlantStack3D_df!, df′! = JopSlantStack3D_df′!,
        s = (;mode, nzfft, nhxfft, nhyfft, kz, khx, khy, px, py, hx0, hy0, dip, azimuth, TX, TK))
end
export JopSlantStack3D

function JopSlantStack3D_df!(d::AbstractArray{T,3}, m::AbstractArray{T,3}; mode, nzfft, nhxfft, nhyfft, kz, khx, khy, px, py, hx0, hy0, dip, azimuth, TX, TK, kwargs...) where {T}
    nhx, nhy, npx, npy, dhx, dhy = size(m,2), size(m,3), length(px), length(py), abs(khx[2]-khx[1]), abs(khy[2]-khy[1])
    nz = size(d, 1)

    mpad = zeros(T, nzfft, nhxfft, nhyfft)
    mpad[1:nz,1:nhx,1:nhy] = TX*m
    M = TK*rfft(mpad)

    if mode == "time"
        compute_kh = slantstack_khxy_from_frequency
    elseif dip == 0
        compute_kh = slantstack_khxy_from_kz
    else
        compute_kh = slantstack_khxy_from_kz_geologic
    end

    is_out_of_bounds = (ikhx_m1, ikhx_p1, ikhy_m1, ikhy_p1)->(ikhx_m1 < 1 || ikhx_p1 > nhxfft || ikhy_m1 < 1 || ikhy_p1 > nhyfft)
    
    D = zeros(eltype(M), size(M,1), npx, npy)
    @threads for ip = 1:npx*npy
        ipy = div(ip-1, npx) + 1
        ipx = mod(ip-1, npx) + 1
        factor = sqrt( (1 + tan(dip)^2) / (1 + tan(dip)^2 * cos(azimuth - py[ipy])^2) )
        for ikz = 1:div(nzfft,2)+1
            ikhx_m1, ikhx_p1, _khx, ikhy_m1, ikhy_p1, _khy = compute_kh(ikz, ipx, ipy, px, py, kz, khx, khy, nhxfft, nhyfft, factor)

            is_out_of_bounds(ikhx_m1, ikhx_p1, ikhy_m1, ikhy_p1) && continue

            if (ikhx_m1 == ikhx_p1) && (ikhy_m1 == ikhy_p1)
                D[ikz,ipx,ipy] = M[ikz,ikhx_m1,ikhy_m1]*exp(-im*(khx[ikhx_m1]*hx0 + khy[ikhy_m1]*hy0))
                continue
            end

            ax_m1 = (khx[ikhx_p1] - _khx)/dhx
            ax_p1 = 1 - ax_m1
            ay_m1 = (khy[ikhy_p1] - _khy)/dhy
            ay_p1 = 1 - ay_m1

            d_p1_p1 = M[ikz,ikhx_p1,ikhy_p1]*exp(-im*(khx[ikhx_p1]*hx0 + khy[ikhy_p1]*hy0))
            d_p1_m1 = M[ikz,ikhx_p1,ikhy_m1]*exp(-im*(khx[ikhx_p1]*hx0 + khy[ikhy_m1]*hy0))
            d_m1_p1 = M[ikz,ikhx_m1,ikhy_p1]*exp(-im*(khx[ikhx_m1]*hx0 + khy[ikhy_p1]*hy0))
            d_m1_m1 = M[ikz,ikhx_m1,ikhy_m1]*exp(-im*(khx[ikhx_m1]*hx0 + khy[ikhy_m1]*hy0))

            D[ikz,ipx,ipy] = ax_m1*ay_m1*d_m1_m1 + ax_m1*ay_p1*d_m1_p1 + ax_p1*ay_m1*d_p1_m1 + ax_p1*ay_p1*d_p1_p1
        end
    end

    _d = brfft(D, nzfft, 1)
    d .= _d[1:nz,1:npx,1:npy] ./ nzfft
end

function JopSlantStack3D_df!(d::AbstractArray{T,5}, m::AbstractArray{T,5}; mode, nzfft, nhxfft, nhyfft, kz, khx, khy, px, py, hx0, hy0, dip, azimuth, TX, TK, kwargs...) where {T}
    ny, nx, nhx, nhy, npx, npy, dhx, dhy = size(m,2), size(m,3), size(m,4), size(m,5), length(px), length(py), abs(khx[2]-khx[1]), abs(khy[2]-khy[1])
    nz = size(d, 1)

    mpad = zeros(T, nzfft, ny, nx, nhxfft, nhyfft)
    mpad[1:nz,1:ny,1:nx,1:nhx,1:nhy] = TX*m
    M = TK*rfft(mpad, (1, 4, 5))

    if mode == "time"
        compute_kh = slantstack_khxy_from_frequency
    elseif dip == 0
        compute_kh = slantstack_khxy_from_kz
    else
        compute_kh = slantstack_khxy_from_kz_geologic
    end

    is_out_of_bounds = (ikhx_m1, ikhx_p1, ikhy_m1, ikhy_p1)->(ikhx_m1 < 1 || ikhx_p1 > nhxfft || ikhy_m1 < 1 || ikhy_p1 > nhyfft)

    D = zeros(eltype(M), size(M,1), ny, nx, npx, npy)
    @threads for ip = 1:npx*npy
        ipy = div(ip-1, npx) + 1
        ipx = mod(ip-1, npx) + 1
        factor = sqrt( (1 + tan(dip)^2) / (1 + tan(dip)^2 * cos(azimuth - py[ipy])^2) )
        for ikz = 1:div(nzfft,2)+1
            ikhx_m1, ikhx_p1, _khx, ikhy_m1, ikhy_p1, _khy = compute_kh(ikz, ipx, ipy, px, py, kz, khx, khy, nhxfft, nhyfft, factor)

            is_out_of_bounds(ikhx_m1, ikhx_p1, ikhy_m1, ikhy_p1) && continue

            if (ikhx_m1 == ikhx_p1) && (ikhy_m1 == ikhy_p1)
                @views D[ikz,:,:,ipx,ipy] .= M[ikz,:,:,ikhx_m1,ikhy_m1] .* exp(-im*(khx[ikhx_m1]*hx0 + khy[ikhy_m1]*hy0))
                continue
            end

            ax_m1 = (khx[ikhx_p1] - _khx)/dhx
            ax_p1 = 1 - ax_m1
            ay_m1 = (khy[ikhy_p1] - _khy)/dhy
            ay_p1 = 1 - ay_m1

            @views D[ikz,:,:,ipx,ipy] .= (ax_m1*ay_m1*exp(-im*(khx[ikhx_m1]*hx0 + khy[ikhy_m1]*hy0))) .* M[ikz,:,:,ikhx_m1,ikhy_m1] .+ 
                                         (ax_m1*ay_p1*exp(-im*(khx[ikhx_m1]*hx0 + khy[ikhy_p1]*hy0))) .* M[ikz,:,:,ikhx_m1,ikhy_p1] .+ 
                                         (ax_p1*ay_m1*exp(-im*(khx[ikhx_p1]*hx0 + khy[ikhy_m1]*hy0))) .* M[ikz,:,:,ikhx_p1,ikhy_m1] .+ 
                                         (ax_p1*ay_p1*exp(-im*(khx[ikhx_p1]*hx0 + khy[ikhy_p1]*hy0))) .* M[ikz,:,:,ikhx_p1,ikhy_p1]
        end
    end

    _d = brfft(D, nzfft, 1)
    d .= _d[1:nz,:,:,1:npx,1:npy] ./ nzfft
end

function JopSlantStack3D_df′!(m::AbstractArray{T,3}, d::AbstractArray{T,3}; mode, nzfft, nhxfft, nhyfft, kz, khx, khy, px, py, hx0, hy0, dip, azimuth, TX, TK, kwargs...) where {T}
    nhx, nhy, npx, npy, dhx, dhy = size(m,2), size(m,3), length(px), length(py), abs(khx[2]-khx[1]), abs(khy[2]-khy[1])
    nz = size(d, 1)

    dpad = zeros(T, nzfft, npx, npy)
    dpad[1:nz,:,:] = d

    D = (rfft(dpad, 1) ./ nzfft)

    if mode == "time"
        compute_kh = slantstack_khxy_from_frequency
    elseif dip == 0
        compute_kh = slantstack_khxy_from_kz
    else
        compute_kh = slantstack_khxy_from_kz_geologic
    end

    is_out_of_bounds = (ikhx_m1, ikhx_p1, ikhy_m1, ikhy_p1)->(ikhx_m1 < 1 || ikhx_p1 > nhxfft || ikhy_m1 < 1 || ikhy_p1 > nhyfft)

    M = zeros(Complex{T}, div(nzfft,2)+1, nhxfft, nhyfft)
    for ip = 1:npx*npy
        ipy = div(ip-1, npx) + 1
        ipx = mod(ip-1, npx) + 1
        factor = sqrt( (1 + tan(dip)^2) / (1 + tan(dip)^2 * cos(azimuth - py[ipy])^2) )
        for ikz = 1:div(nzfft,2)+1
            ikhx_m1, ikhx_p1, _khx, ikhy_m1, ikhy_p1, _khy = compute_kh(ikz, ipx, ipy, px, py, kz, khx, khy, nhxfft, nhyfft, factor)

            is_out_of_bounds(ikhx_m1, ikhx_p1, ikhy_m1, ikhy_p1) && continue

            if (ikhx_m1 == ikhx_p1) && (ikhy_m1 == ikhy_p1)
                M[ikz,ikhx_p1,ikhy_p1] += D[ikz,ipx,ipy]*exp(im*(khx[ikhx_p1]*hx0 + khy[ikhy_p1]*hy0))
                continue
            end

            ax_m1 = (khx[ikhx_p1] - _khx)/dhx
            ax_p1 = 1 - ax_m1
            ay_m1 = (khy[ikhy_p1] - _khy)/dhy
            ay_p1 = 1 - ay_m1

            m_p1_p1 = D[ikz,ipx,ipy]*exp(im*(khx[ikhx_p1]*hx0 + khy[ikhy_p1]*hy0))
            m_p1_m1 = D[ikz,ipx,ipy]*exp(im*(khx[ikhx_p1]*hx0 + khy[ikhy_m1]*hy0))
            m_m1_p1 = D[ikz,ipx,ipy]*exp(im*(khx[ikhx_m1]*hx0 + khy[ikhy_p1]*hy0))
            m_m1_m1 = D[ikz,ipx,ipy]*exp(im*(khx[ikhx_m1]*hx0 + khy[ikhy_m1]*hy0))

            M[ikz,ikhx_p1,ikhy_p1] += ax_p1*ay_p1*m_p1_p1
            M[ikz,ikhx_p1,ikhy_m1] += ax_p1*ay_m1*m_p1_m1
            M[ikz,ikhx_m1,ikhy_p1] += ax_m1*ay_p1*m_m1_p1
            M[ikz,ikhx_m1,ikhy_m1] += ax_m1*ay_m1*m_m1_m1
        end
    end

    m .= TX * (brfft(TK * M, nzfft)[1:nz,1:nhx,1:nhy])
end

function JopSlantStack3D_df′!(m::AbstractArray{T,5}, d::AbstractArray{T,5}; mode, nzfft, nhxfft, nhyfft, kz, khx, khy, px, py, hx0, hy0, dip, azimuth, TX, TK, kwargs...) where {T}
    ny, nx, nhx, nhy, npx, npy, dhx, dhy = size(m,2), size(m,3), size(m,4), size(m,5), length(px), length(py), abs(khx[2]-khx[1]), abs(khy[2]-khy[1])
    nz = size(d, 1)

    dpad = zeros(T, nzfft, ny, nx, npx, npy)
    dpad[1:nz,:,:,:,:] = d

    D = (rfft(dpad, 1) ./ nzfft)

    if mode == "time"
        compute_kh = slantstack_khxy_from_frequency
    elseif dip == 0
        compute_kh = slantstack_khxy_from_kz
    else
        compute_kh = slantstack_khxy_from_kz_geologic
    end

    is_out_of_bounds = (ikhx_m1, ikhx_p1, ikhy_m1, ikhy_p1)->(ikhx_m1 < 1 || ikhx_p1 > nhxfft || ikhy_m1 < 1 || ikhy_p1 > nhyfft)

    M = zeros(Complex{T}, div(nzfft,2)+1, ny, nx, nhxfft, nhyfft)
    for ip = 1:npx*npy
        ipy = div(ip-1, npx) + 1
        ipx = mod(ip-1, npx) + 1
        factor = sqrt( (1 + tan(dip)^2) / (1 + tan(dip)^2 * cos(azimuth - py[ipy])^2) )
        for ikz = 1:div(nzfft,2)+1
            ikhx_m1, ikhx_p1, _khx, ikhy_m1, ikhy_p1, _khy = compute_kh(ikz, ipx, ipy, px, py, kz, khx, khy, nhxfft, nhyfft, factor)

            is_out_of_bounds(ikhx_m1, ikhx_p1, ikhy_m1, ikhy_p1) && continue

            if (ikhx_m1 == ikhx_p1) && (ikhy_m1 == ikhy_p1)
                @views M[ikz,:,:,ikhx_p1,ikhy_p1] .+= D[ikz,:,:,ipx,ipy] .* exp(im*(khx[ikhx_p1]*hx0 + khy[ikhy_p1]*hy0))
                continue
            end

            ax_m1 = (khx[ikhx_p1] - _khx)/dhx
            ax_p1 = 1 - ax_m1
            ay_m1 = (khy[ikhy_p1] - _khy)/dhy
            ay_p1 = 1 - ay_m1

            @views M[ikz,:,:,ikhx_p1,ikhy_p1] .+= (ax_p1*ay_p1*exp(im*(khx[ikhx_p1]*hx0 + khy[ikhy_p1]*hy0))) .* D[ikz,:,:,ipx,ipy]
            @views M[ikz,:,:,ikhx_p1,ikhy_m1] .+= (ax_p1*ay_m1*exp(im*(khx[ikhx_p1]*hx0 + khy[ikhy_m1]*hy0))) .* D[ikz,:,:,ipx,ipy]
            @views M[ikz,:,:,ikhx_m1,ikhy_p1] .+= (ax_m1*ay_p1*exp(im*(khx[ikhx_m1]*hx0 + khy[ikhy_p1]*hy0))) .* D[ikz,:,:,ipx,ipy]
            @views M[ikz,:,:,ikhx_m1,ikhy_m1] .+= (ax_m1*ay_m1*exp(im*(khx[ikhx_m1]*hx0 + khy[ikhy_m1]*hy0))) .* D[ikz,:,:,ipx,ipy]
        end
    end

    m .= TX * (brfft(TK * M, nzfft, (1,4,5))[1:nz,:,:,1:nhx,1:nhy])
end

@inline function slantstack_khxy_from_kz(ikz::Int64, ipx::Int64, ipy::Int64, px, py, kz, khx, khy, nhxfft, nhyfft, factor)
    _khx = kz[ikz]*px[ipx]*cos(py[ipy])
    _khy = kz[ikz]*px[ipx]*sin(py[ipy])

    # check if wavenumber is outside Nyquist
    ((_khx > khx[div(nhxfft+1,2)]) || (_khx < khx[div(nhxfft+1,2)+1]) || (_khy > khy[div(nhyfft+1,2)]) || (_khy < khy[div(nhyfft+1,2)+1])) && return -1, -1, _khx, -1, -1, _khy

    ikhx_m1 = floor(Int64, _khx/khx[2]) + 1
    ikhx_p1 = ceil(Int64, _khx/khx[2]) + 1
    ikhy_m1 = floor(Int64, _khy/khy[2]) + 1
    ikhy_p1 = ceil(Int64, _khy/khy[2]) + 1

    ikhx_m1 = ikhx_m1 < 1 ? nhxfft + ikhx_m1 : ikhx_m1
    ikhx_p1 = ikhx_p1 < 1 ? nhxfft + ikhx_p1 : ikhx_p1
    ikhy_m1 = ikhy_m1 < 1 ? nhyfft + ikhy_m1 : ikhy_m1
    ikhy_p1 = ikhy_p1 < 1 ? nhyfft + ikhy_p1 : ikhy_p1

    ikhx_m1, ikhx_p1, _khx, ikhy_m1, ikhy_p1, _khy
end

@inline function slantstack_khxy_from_kz_geologic(ikz::Int64, ipx::Int64, ipy::Int64, px, py, kz, khx, khy, nhxfft, nhyfft, factor)
    _khx = kz[ikz]*px[ipx]*cos(py[ipy])*factor
    _khy = kz[ikz]*px[ipx]*sin(py[ipy])*factor

    # check if wavenumber is outside Nyquist
    ((_khx > khx[div(nhxfft+1,2)]) || (_khx < khx[div(nhxfft+1,2)+1]) || (_khy > khy[div(nhyfft+1,2)]) || (_khy < khy[div(nhyfft+1,2)+1])) && return -1, -1, _khx, -1, -1, _khy

    ikhx_m1 = floor(Int64, _khx/khx[2]) + 1
    ikhx_p1 = ceil(Int64, _khx/khx[2]) + 1
    ikhy_m1 = floor(Int64, _khy/khy[2]) + 1
    ikhy_p1 = ceil(Int64, _khy/khy[2]) + 1

    ikhx_m1 = ikhx_m1 < 1 ? nhxfft + ikhx_m1 : ikhx_m1
    ikhx_p1 = ikhx_p1 < 1 ? nhxfft + ikhx_p1 : ikhx_p1
    ikhy_m1 = ikhy_m1 < 1 ? nhyfft + ikhy_m1 : ikhy_m1
    ikhy_p1 = ikhy_p1 < 1 ? nhyfft + ikhy_p1 : ikhy_p1

    ikhx_m1, ikhx_p1, _khx, ikhy_m1, ikhy_p1, _khy
end

@inline function slantstack_khxy_from_frequency(iω::Int64, ipx::Int64, ipy::Int64, px, py, ω, khx, khy, nhxfft, nhyfft, factor)
    _khx = px[ipx]*ω[iω]
    _khy = py[ipy]*ω[iω]

    # check if wavenumber is outside Nyquist
    ((_khx > khx[div(nhxfft+1,2)]) || (_khx < khx[div(nhxfft+1,2)+1]) || (_khy > khy[div(nhyfft+1,2)]) || (_khy < khy[div(nhyfft+1,2)+1])) && return -1, -1, _khx, -1, -1, _khy

    ikhx_m1 = floor(Int64, _khx/khx[2]) + 1
    ikhx_p1 = ceil(Int64, _khx/khx[2]) + 1
    ikhy_m1 = floor(Int64, _khy/khy[2]) + 1
    ikhy_p1 = ceil(Int64, _khy/khy[2]) + 1

    ikhx_m1 = ikhx_m1 < 1 ? nhxfft + ikhx_m1 : ikhx_m1
    ikhx_p1 = ikhx_p1 < 1 ? nhxfft + ikhx_p1 : ikhx_p1
    ikhy_m1 = ikhy_m1 < 1 ? nhyfft + ikhy_m1 : ikhy_m1
    ikhy_p1 = ikhy_p1 < 1 ? nhyfft + ikhy_p1 : ikhy_p1

    ikhx_m1, ikhx_p1, _khx, ikhy_m1, ikhy_p1, _khy
end

"""
    A = JopSlantStackShiftSum(dom[; dz=1.0, ...])

where `A` is the 2D slant-stack operator mapping for `z-h` to `z-θ` (depth mode) or `t-h` to `tau-p` (time mode).
The slant stacking is performed directly on the input domain by shifting and summing along the offset axis.
The domain of the operator is `nz` x `nh` with precision T, `dz` is the depth spacing (or time interval),
`h`, if provided, is the array of (half) offsets (can be irregular). If not provided, a regular grid is assumed with same sampling as dz.
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
    holder = zeros(T, nz, Threads.maxthreadid())
    @threads for ip = 1:np
        for ih = 1:nh
            shift = + h[ih] * p[ip] / dz
            if abs(shift) < nz
                _shift_forward!(@view(holder[:,Threads.threadid()]), @view(mtap[:,ih]), shift)
                @views d[:,ip] .+= holder[:,Threads.threadid()]
            end
        end
    end
    d
end

function JopSlantStackShiftSum_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; dz, h, p, TZ, kwargs...) where {T}
    nz, nh, np = size(m,1), size(m,2), length(p)

    mtap = zeros(T, nz, nh)
    holder = zeros(T, nz, Threads.maxthreadid())
    @threads for ih = 1:nh
        for ip = 1:np
            shift = + h[ih] * p[ip] / dz
            if abs(shift) < nz
                _shift_adjoint!(@view(holder[:,Threads.threadid()]), @view(d[:,ip]), shift)
                @views mtap[:,ih] .+= holder[:,Threads.threadid()]
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
`hx` and `hy`, if provided, are the arrays of (half) offsets (can be irregular). If not provided, a regular grid is assumed with same sampling as dz.
The additional named optional arguments along with their default values are,

* `mode="depth` - choose between "depth" and "time" to specify if the input domain is `z-hx-hy` or `t-hx-hy`
* `theta=-45:1.0:45` - range of incidence angles used when `mode="depth"`
* `phi=0:45.0:135` - range of image azimuth angles used when `mode="depth"`
* `dip=0.0` - geologic dip used when `mode="depth"`
* `azimuth=0.0` - geologic azimuth used when `mode="depth"`
* `px=range(-dz/max(diff(hx)),dz/max(diff(hx)),64)` - ray parameter sampling used when `mode="time"`
* `py=range(-dz/max(diff(hy)),dz/max(diff(hy)),64)` - ray parameter sampling used when `mode="time"`
* `taperz=(0,0)` - beginning and end taper (fractional) in the z-direction (or t-direction) before shifting and summing

# Notes

* If the mode is "time", then `dz` is the time sampling interval, `taperz` is the taper for the time dimension.
* For mode="depth", typically `theta` needs to cover both positive and negative angles and must be < 90 degrees.
* For mode="depth", the incidence angle depends on the geologic `dip` and `azimuth` (see 3-D Seismic Imaging by Prof. Biondo Biondi, Chapter 6). If not provided, this dependency is ignored.
* Typically, geologic `dip` ∈ [0, 90] degrees and `azimuth` ∈ [0, 360] degrees, which is different from `theta` and `phi` ranges.
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
    dhxmax = length(hx) > 1 ? maximum(diff(sort(hx))) : 1.0
    dhymax = length(hy) > 1 ? maximum(diff(sort(hy))) : 1.0
    mode == "depth" && any(abs.(theta) .>= 90.0) && error("for mode='depth', theta values must be < 90 degrees in absolute value")
    mode == "depth" && (any(phi .> 180.0) || any(phi .< 0.0)) && error("for mode='depth', phi values must be between 0 and 180 degrees")

    nz = size(dom, 1)
    nhx = size(dom, 2)
    nhy = size(dom, ndy)

    # tan(theta), phi - used for mode=="depth"
    tant = @. tan(deg2rad(theta))
    phi = @. deg2rad(phi)

    dip = @. deg2rad(dip)
    azimuth = @. deg2rad(azimuth)

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

function JopSlantStackShiftSum3D_df!(d::AbstractArray{T,3}, m::AbstractArray{T,3}; dip, azimuth, mode, dz, hx, hy, px, py, TZ, kwargs...) where {T}
    if isa(dip, Array)
        JopSlantStackShiftSum3D_df_vector!(d, m; dip, azimuth, mode, dz, hx, hy, px, py, TZ)
    else
        JopSlantStackShiftSum3D_df_scalar!(d, m; dip, azimuth, mode, dz, hx, hy, px, py, TZ)
    end
end

function JopSlantStackShiftSum3D_df′!(m::AbstractArray{T,3}, d::AbstractArray{T,3}; dip, azimuth, mode, dz, hx, hy, px, py, TZ, kwargs...) where {T}
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
    holder = zeros(T, nz, Threads.maxthreadid())
    @threads for ip = 1:npx*npy
        ipy = div(ip-1, npx) + 1
        ipx = mod(ip-1, npx) + 1
        factor = sqrt( (1 + tan(dip)^2) / (1 + tan(dip)^2 * cos(azimuth - py[ipy])^2) )
        for ih = 1:nh
            shift = compute_shift(1, ih, ih, ipx, ipy, hx, hy, px, py, dz, factor)
            if abs(shift) < nz
                _shift_forward!(@view(holder[:,Threads.threadid()]), @view(mtap[:,ih]), shift)
                @views d[:,ipx,ipy] .+= holder[:,Threads.threadid()]
            end
        end
    end
    d
end

function JopSlantStackShiftSum3D_df_scalar!(d::AbstractArray{T,3}, m::AbstractArray{T,3}; dip::Real, azimuth::Real, mode, dz, hx, hy, px, py, TZ) where {T}
    nz, nhx, nhy, npx, npy = size(m,1), size(m,2), size(m,3), length(px), length(py)

    if mode == "time"
        compute_shift = slantstack_shift_px_py
    elseif dip == 0
        compute_shift = slantstack_shift_theta_phi
    else
        compute_shift = slantstack_shift_theta_phi_geologic
    end

    d .= 0
    mtap = TZ * m
    holder = zeros(T, nz, Threads.maxthreadid())
    @threads for ip = 1:npx*npy
        ipy = div(ip-1, npx) + 1
        ipx = mod(ip-1, npx) + 1
        factor = sqrt( (1 + tan(dip)^2) / (1 + tan(dip)^2 * cos(azimuth - py[ipy])^2) )
        for ihx = 1:nhx
            for ihy = 1:nhy
                shift = compute_shift(1, ihx, ihy, ipx, ipy, hx, hy, px, py, dz, factor)
                if abs(shift) < nz
                    _shift_forward!(@view(holder[:,Threads.threadid()]), @view(mtap[:,ihx,ihy]), shift)
                    @views d[:,ipx,ipy] .+= holder[:,Threads.threadid()]
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
    @threads for ip = 1:npx*npy
        ipy = div(ip-1, npx) + 1
        ipx = mod(ip-1, npx) + 1
        factor = sqrt.( (1 .+ tan.(dip).^2) ./ (1 .+ tan.(dip).^2 .* cos.(azimuth .- py[ipy]).^2) )
        for ih = 1:nh
            for iz = 1:nz
                shift = compute_shift(iz, ih, ih, ipx, ipy, hx, hy, px, py, dz, factor)
                if abs(shift) < nz 
                    # holder=_shift_forward(@view(mtap[:,ih]), shift)
                    # d[iz,ipx,ipy] += holder[iz]
                    d[iz,ipx,ipy] += _interpolate_forward(@view(mtap[:,ih]), iz - shift)
                end
            end
        end
    end
    d
end

function JopSlantStackShiftSum3D_df_vector!(d::AbstractArray{T,3}, m::AbstractArray{T,3}; dip::AbstractArray{T,1}, azimuth::AbstractArray{T,1}, mode, dz, hx, hy, px, py, TZ) where {T}
    nz, nhx, nhy, npx, npy = size(m,1), size(m,2), size(m,3), length(px), length(py)

    compute_shift = slantstack_shift_theta_phi_geologic
    
    d .= 0
    mtap = TZ * m
    @threads for ip = 1:npx*npy
        ipy = div(ip-1, npx) + 1
        ipx = mod(ip-1, npx) + 1
        factor = sqrt.( (1 .+ tan.(dip).^2) ./ (1 .+ tan.(dip).^2 .* cos.(azimuth .- py[ipy]).^2) )
        for ihx = 1:nhx
            for ihy = 1:nhy
                for iz = 1:nz
                    shift = compute_shift(iz, ihx, ihy, ipx, ipy, hx, hy, px, py, dz, factor)
                    if abs(shift) < nz 
                        d[iz,ipx,ipy] += _interpolate_forward(@view(mtap[:,ihx,ihy]), iz - shift)
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
    acc = zeros(T, nz, Threads.maxthreadid())
    holder = zeros(T, nz, Threads.maxthreadid())
    @threads for ih = 1:nh
        fill!(@view(acc[:, Threads.threadid()]), zero(T))
        for ipx = 1:npx
            for ipy = 1:npy
                factor = sqrt( (1 + tan(dip)^2) / (1 + tan(dip)^2 * cos(azimuth - py[ipy])^2) )
                shift = compute_shift(1, ih, ih, ipx, ipy, hx, hy, px, py, dz, factor)
                if abs(shift) < nz
                    _shift_adjoint!(@view(holder[:,Threads.threadid()]), @view(d[:,ipx,ipy]), shift)
                    @views acc[:, Threads.threadid()] .+= holder[:,Threads.threadid()]
                end
            end
        end
        @views mtap[:,ih] .= acc[:, Threads.threadid()]
    end
    m = TZ' * mtap
    m
end

function JopSlantStackShiftSum3D_df_scalar′!(m::AbstractArray{T,3}, d::AbstractArray{T,3}; dip::Real, azimuth::Real, mode, dz, hx, hy, px, py, TZ, kwargs...) where {T}
    nz, nhx, nhy, npx, npy = size(m,1), size(m,2), size(m,3), length(px), length(py)

    if mode == "time"
        compute_shift = slantstack_shift_px_py
    elseif dip == 0
        compute_shift = slantstack_shift_theta_phi
    else
        compute_shift = slantstack_shift_theta_phi_geologic
    end

    mtap = zeros(T, nz, nhx, nhy)
    acc = zeros(T, nz, Threads.maxthreadid())
    holder = zeros(T, nz, Threads.maxthreadid())
    @threads for ih = 1:nhx*nhy
        ihy = div(ih-1, nhx) + 1
        ihx = mod(ih-1, nhx) + 1
        fill!(@view(acc[:, Threads.threadid()]), zero(T))
        for ipx = 1:npx
            for ipy = 1:npy
                factor = sqrt( (1 + tan(dip)^2) / (1 + tan(dip)^2 * cos(azimuth - py[ipy])^2) )
                shift = compute_shift(1, ihx, ihy, ipx, ipy, hx, hy, px, py, dz, factor)
                if abs(shift) < nz
                    _shift_adjoint!(@view(holder[:, Threads.threadid()]), @view(d[:,ipx,ipy]), shift)
                    @views acc[:, Threads.threadid()] .+= holder[:, Threads.threadid()]
                end
            end
        end
        @views mtap[:,ihx,ihy] .= acc[:, Threads.threadid()]
    end
    m = TZ' * mtap
    m
end

function JopSlantStackShiftSum3D_df_vector′!(m::AbstractArray{T,2}, d::AbstractArray{T,3}; dip::AbstractArray{T,1}, azimuth::AbstractArray{T,1}, mode, dz, hx, hy, px, py, TZ, kwargs...) where {T}
    nz, nh, npx, npy = size(m,1), size(m,2), length(px), length(py)

    compute_shift = slantstack_shift_theta_phi_geologic

    mtap = zeros(T, nz, nh)
    @threads for ih = 1:nh
        for ipx = 1:npx
            for ipy = 1:npy
                factor = sqrt.( (1 .+ tan.(dip).^2) ./ (1 .+ tan.(dip).^2 .* cos.(azimuth .- py[ipy]).^2) )
                for iz = 1:nz
                    shift = compute_shift(iz, ih, ih, ipx, ipy, hx, hy, px, py, dz, factor)
                    if abs(shift) < nz
                        _interpolate_adjoint!(@view(mtap[:,ih]), d[iz,ipx,ipy], iz - shift)
                    end
                end
            end
        end
    end
    m = TZ' * mtap
    m
end

function JopSlantStackShiftSum3D_df_vector′!(m::AbstractArray{T,3}, d::AbstractArray{T,3}; dip::AbstractArray{T,1}, azimuth::AbstractArray{T,1}, mode, dz, hx, hy, px, py, TZ, kwargs...) where {T}
    nz, nhx, nhy, npx, npy = size(m,1), size(m,2), size(m,3), length(px), length(py)

    compute_shift = slantstack_shift_theta_phi_geologic

    mtap = zeros(T, nz, nhx, nhy)
    @threads for ih = 1:nhx*nhy
        ihy = div(ih-1, nhx) + 1
        ihx = mod(ih-1, nhx) + 1
        for ipx = 1:npx
            for ipy = 1:npy
                factor = sqrt.( (1 .+ tan.(dip).^2) ./ (1 .+ tan.(dip).^2 .* cos.(azimuth .- py[ipy]).^2) )
                for iz = 1:nz
                    shift = compute_shift(iz, ihx, ihy, ipx, ipy, hx, hy, px, py, dz, factor)
                    if abs(shift) < nz
                        _interpolate_adjoint!(@view(mtap[:,ihx,ihy]), d[iz,ipx,ipy], iz - shift)
                    end
                end
            end
        end
    end
    m = TZ' * mtap
    m
end

@inline function slantstack_shift_px_py(iz::Int64, ihx::Int64, ihy::Int64, ipx::Int64, ipy::Int64, hx, hy, px, py, dz, factor::Real)
    # for time domain, the shift is given by (hx*px + hy*py) / dz
    shift = + hx[ihx] * px[ipx] + hy[ihy] * py[ipy]
    shift / dz
end

@inline function slantstack_shift_theta_phi(iz::Int64, ihx::Int64, ihy::Int64, ipx::Int64, ipy::Int64, hx, hy, px, py, dz, factor::Real)
    # for depth domain ignoring geologic dip, the shift is given by (-hx*tan(θ)*cos(ϕ) - hy*tan(θ)*sin(ϕ)) / dz
    shift = + hx[ihx] * px[ipx] * cos(py[ipy]) + hy[ihy] * px[ipx] * sin(py[ipy])
    shift / dz
end

@inline function slantstack_shift_theta_phi_geologic(iz::Int64, ihx::Int64, ihy::Int64, ipx::Int64, ipy::Int64, hx, hy, px, py, dz, factor::Real)
    # for depth domain accounting for geologic dip, the shift is given by (-hx*tan(θ)*cos(ϕ)*factor - hy*tan(θ)*sin(ϕ)*factor) / dz
    shift = + hx[ihx] * px[ipx] * cos(py[ipy]) + hy[ihy] * px[ipx] * sin(py[ipy])
    shift * factor / dz
end

@inline function slantstack_shift_theta_phi_geologic(iz::Int64, ihx::Int64, ihy::Int64, ipx::Int64, ipy::Int64, hx, hy, px, py, dz, factor::AbstractArray{T,1}) where {T}
    # for depth domain accounting for geologic dip, the shift is given by (-hx*tan(θ)*cos(ϕ)*factor - hy*tan(θ)*sin(ϕ)*factor) / dz
    shift = + hx[ihx] * px[ipx] * cos(py[ipy]) + hy[ihy] * px[ipx] * sin(py[ipy])
    shift * factor[iz] / dz
end

# helper functions to build a compact sinc kernel for forward and adjoint shifting/interpolation in 1D

function _shift_zeropad(x::AbstractVector{T}, s::Int) where {T}
    y = zeros(T, length(x))
    n = length(x)
    src_lo = max(1, 1 - s)
    src_hi = min(n, n - s)
    if src_lo <= src_hi
        dst_lo = src_lo + s
        dst_hi = src_hi + s
        @views y[dst_lo:dst_hi] .= x[src_lo:src_hi]
    end
    return y
end

function _sinc_kernel(δ::Real, N::Int)
    n = -(N÷2):(N÷2)
    h = sinc.(n .- δ)
    h .* hann(length(h))
end

function _shift_forward!(y::AbstractArray{T,1}, x::AbstractArray{T,1}, shift::Real, N::Int = 7) where {T}
    ishift = round(Int, shift)
    frac = shift - ishift
    if frac == 0.0
        y .= _shift_zeropad(x, ishift)
        return nothing
    end
    h = _sinc_kernel(frac, N)
    L = length(h) ÷ 2
    x_int = _shift_zeropad(x, ishift)
    x_frac = conv(x_int, h)
    y .= x_frac[L+1 : L+length(x)]
end

function _shift_adjoint!(y::AbstractArray{T,1}, x::AbstractArray{T,1}, shift::Real, N::Int = 7) where {T}
    ishift = round(Int, shift)
    frac = shift - ishift
    if frac == 0.0
        y .= _shift_zeropad(x, -ishift)
        return nothing
    end
    h = reverse(_sinc_kernel(frac, N))
    n = length(x)
    m = length(h)
    L = m ÷ 2
    x_pad = zeros(eltype(x), n+m-1)
    x_pad[L+1 : L+n] .= x
    x_frac = conv(x_pad, h)
    y .= _shift_zeropad(@view(x_frac[m : m+n-1]), -ishift)
end

function _interpolate_forward(x::AbstractArray{T,1}, loc::Real, N::Int = 7) where {T}
    iloc = round(Int, loc)
    frac = loc - iloc
    if frac == 0.0 && 1 <= iloc <= length(x)
        return x[iloc]
    end
    
    h = _sinc_kernel(frac, N)
    L = length(h) ÷ 2
    
    # Apply sinc interpolation at fractional location using direct sum
    result = zero(T)
    n = -(N÷2)
    for weight in h
        idx = iloc + n
        if 1 <= idx <= length(x)
            result += x[idx] * weight
        end
        n += 1
    end
    result
end

function _interpolate_adjoint!(x::AbstractArray{T,1}, val::T, loc::Real, N::Int = 7) where {T}
    iloc = round(Int, loc)
    frac = loc - iloc
    
    if frac == 0.0 && 1 <= iloc <= length(x)
        x[iloc] += val
        return nothing
    end
    
    h = _sinc_kernel(frac, N)
    n = -(N÷2)
    for weight in h
        idx = iloc + n
        if 1 <= idx <= length(x)
            x[idx] += val * weight
        end
        n += 1
    end
end