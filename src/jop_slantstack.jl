"""
    A = JopSlantStack(dom[; dz=10.0, dh=10.0, h0=-1000.0, ...])

where `A` is the 2D slant-stack operator mapping for `z-h` to `tau-p`.  The domain of the operator
is `nz` x `nh` with precision T, `dz` is the depth spacing, `dh` is the half-offset spacing, and `h0` is the
origin of the half-offset axis.  The additional named optional arguments along with their default
values are,

* `theta=-60:1.0:60` - range of opening angles.  The ray parameter is: p=sin(theta/2)/c
* `padz=0.0,padh=0.0` - fractional padding in depth and offset to apply before applying the Fourier transfrom
* `taperz=(0,0)` - beginning and end taper in the z-direction before transforming from `z-h` to `kz-kh`
* `taperh=(0,0)` - beginning and end taper in the h-direction before transforming from `z-h` to `kz-kh`
* `taperkz=(0,0)` - beginning and end taper in the kz-direction before transforming from `kz-kh` to `z-h`
* `taperkh=(0,0)` - beginning and end taper in the kh-direction before transforming from `kz-kh` to `z-h`

# Notes
* It should be possible to extend this operator to 3D

* If your domain is t-h rather than z-h, you can still use this operator
"""
function JopSlantStack(
        dom::JetAbstractSpace{T};
        theta = collect(-60.0:1.0:60.0),
        dz = -1.0,
        dh = -1.0,
        h0 = NaN,
        padz = 0.0,
        padh = 0.0,
        taperz = (0,0),
        taperh = (0,0),
        taperkz = (0,0),
        taperkh = (0,0)) where {T}
    dz < 0.0 && error("expected dz>0.0, got dz=$(dz)")
    dh < 0.0 && error("expected dh>0.0, got dh=$(dh)")
    isnan(h0) && error("expected finite h0, got h0=$(h0)")

    nz,nh = size(dom)

    # kz
    nzfft = nextprod([2,3,5,7], round(Int, nz*(1 + padz)))
    kn = pi/dz
    dk = kn/nzfft
    kz = dk*[0:div(nzfft,2)+1;]

    # kh
    nhfft = nextprod([2,3,5,7], round(Int, nh*(1+padh)))
    kn = pi/dh
    dk = kn/nhfft
    local kh
    if rem(nhfft,2) == 0
        kh = 2*dk*[ [0:div(nhfft,2);] ; [-div(nhfft,2)+1:1:-1;] ] # factor of 2 is for kg+ks with kg=ks -- i.e. go from (kg,ks)->kh
    else
        kh = 2*dk*[ [0:div(nhfft,2);] ; [-div(nhfft,2)+0:1:-1;] ] # factor of 2 is for kg+ks with kg=ks -- i.e. go from (kg,ks)->kh
    end

    # c*p
    cp = sin.(.5*theta*pi/180) # factor of .5 is to make theta the opening angle -- i.e. go from (theta_s, theta_g)->theta

    # conversions
    kz,kh,cp = map(x->convert(Array{T,1}, x), (kz,kh,cp))
    nzfft,nhfft = map(x->convert(Int64, x), (nzfft,nhfft))
    h0 = T(h0)

    # tapers
    TX = JopTaper(dom, (1,2), (taperz[1],taperh[1]), (taperz[2],taperh[2]))
    TK = JopTaper(JetSpace(Complex{eltype(dom)},div(nzfft,2)+1,length(cp)), (1,2), (taperkz[1], taperkh[1]), (taperkz[2], taperkh[2]), mode=(:normal,:fftshift))

    JopLn(dom = dom, rng = JetSpace(T, nz, length(cp)), df! = JopSlantStack_df!, df′! = JopSlantStack_df′!,
        s = (nzfft=nzfft, nhfft=nhfft, kz=kz, kh=kh, cp=cp, h0=h0, TX=TX, TK=TK))
end
export JopSlantStack

function JopSlantStack_df!(d::AbstractArray{T,2}, m::AbstractArray{T,2}; nzfft, nhfft, kz, kh, cp, h0, TX, TK, kwargs...) where {T}
    nz, nh, np, dh = size(m)..., length(cp), abs(kh[2]-kh[1])

    mpad = zeros(T, nzfft, nhfft)
    mpad[1:nz,1:nh] = TX*m

    M = rfft(mpad)
    dtmp = similar(d)

    D = zeros(eltype(M), size(M,1), np)
    for ikz = 1:div(nzfft,2)+1, ip = 1:np
        ikh_m1, ikh_p1, _kh = slantstack_compute_kh(ikz, ip, cp, kz, kh, nhfft)

        ikh_m1 < 1 && continue

        if ikh_m1 == ikh_p1
            D[ikz,ip] = M[ikz,ikh_p1]*exp(-im*kh[ikh_p1]*h0)
            continue
        end

        local d_p1, a_p1
        if 1 <= ikh_p1 <= nhfft
            d_p1 = M[ikz,ikh_p1]*exp(-im*kh[ikh_p1]*h0)
            a_p1 = abs(kh[ikh_p1] - _kh)/dh
        else
            a_p1 = 0.0
        end

        local d_m1, a_m1
        if 1 <= ikh_m1 <= nhfft
            d_m1 = M[ikz,ikh_m1]*exp(-im*kh[ikh_m1]*h0)
            a_m1 = abs(_kh - kh[ikh_m1])/dh
        else
            a_m1 = 0.0
        end

        D[ikz,ip] = a_m1*d_m1 + a_p1*d_p1
    end

    d .= brfft(TK*D, nzfft, 1)[1:nz,1:np] ./ nzfft
end

function JopSlantStack_df′!(m::AbstractArray{T,2}, d::AbstractArray{T,2}; nzfft, nhfft, kz, kh, cp, h0, TX, TK, kwargs...) where {T}
    nz, nh, np, dh = size(m)..., length(cp), abs(kh[2]-kh[1])

    dpad = zeros(T, nzfft, np)
    dpad[1:nz,:] = d

    D = TK * (rfft(dpad, 1) ./ nzfft)

    M = zeros(Complex{T}, div(nzfft,2)+1, nhfft)
    for ikz = 1:div(nzfft,2)+1, ip = 1:np
        ikh_m1, ikh_p1, _kh = slantstack_compute_kh(ikz, ip, cp, kz, kh, nhfft)

        ikh_m1 < 1 && continue

        if ikh_m1 == ikh_p1
            M[ikz,ikh_p1] += D[ikz,ip]*exp(im*kh[ikh_p1]*h0)
            continue
        end

        if 1 <= ikh_p1 <= nhfft
            m_p1 = D[ikz,ip]*exp(im*kh[ikh_p1]*h0)
            a_p1 = (kh[ikh_p1] - _kh)/dh
            M[ikz,ikh_p1] += a_p1*m_p1
        end

        if 1 <= ikh_m1 <= nhfft
            m_m1 = D[ikz,ip]*exp(im*kh[ikh_m1]*h0)
            a_m1 = (_kh - kh[ikh_m1])/dh
            M[ikz,ikh_m1] += a_m1*m_m1
        end
    end
    m .= TX * (brfft(M, nzfft)[1:nz,1:nh])
end

@inline function slantstack_compute_kh(ikz::Int64, ip::Int64, cp, kz, kh, nhfft)
    _kh = 1/(1-cp[ip]^2) - 1
    _kh < 0 && return -1,-1,1.0
    _kh = 2*kz[ikz]*sqrt(_kh)

    ikh_m1 = floor(Int64, _kh/kh[2]) + 1
    ikh_p1 = ceil(Int64, _kh/kh[2]) + 1
    ikh_m1 = ikh_m1 < 1 ? nhfft + ikh_m1 : ikh_m1
    ikh_p1 = ikh_p1 < 1 ? nhfft + ikh_p1 : ikh_p1

    ikh_m1, ikh_p1, _kh
end
