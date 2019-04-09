"""
    A = JopFdct(N1, N2[; nbscales=-1, nbangles_coarse=16, ac=2])

`A` is the 2D curvelet transform operator, operating on domain that contains
as two-dimensional array: `size(domain(A))=(N1,N2)`.  The optional arguments and
their default values are:

* `nbscales=-1` number of curvelet scales, if set to `-1`, then a heuristic is used
to determine the number of scales.

* `nbangles_coarse=16` number of angles at the coarsest scales

* `ac=2` if set to `2`, then the wavelets at the finest scale.  If set to `1`,
use curvelets at the finset scale.
"""
function JopFdct(N1::Int, N2::Int; nbscales=-1, nbangles_coarse=16, ac=2, t::Type{T}=Complex{Float64}) where {T}
    p = Fdct(N1, N2, nbscales=nbscales, nbangles_coarse=nbangles_coarse, ac=ac)
    n = ncurvelets(p)
    if t == Float64
        n *= 2
    end
    JopLn(dom = JetSpace(Float64, N1, N2), rng = JetSpace(T, n), df! = JopFdct_df!, df′! = JopFdct_df′!, s = (p=p,))
end
export JopFdct

JopFdct_df!(d::Vector{ComplexF64}, m::Matrix{Float64}; p, kwargs...) = begin fdct!(d, p, m); d end
JopFdct_df′!(m::Matrix{Float64}, d::Vector{ComplexF64}; p, kwargs...) = begin ifdct!(m, p, d); m end

function JopFdct_df!(d::Vector{Float64}, m::Matrix{Float64}; p, kwargs...)
    _d = unsafe_wrap(Array, convert(Ptr{ComplexF64}, pointer(d)), (div(length(d),2),), own=false)
    fdct!(_d, p, m)
    d
end

function JopFdct_df′!(m::Matrix{Float64}, d::Vector{Float64}; p, kwargs...)
    _d = unsafe_wrap(Array, convert(Ptr{ComplexF64}, pointer(d)), (div(length(d),2),), own=false)
    ifdct!(m, p, _d)
    m
end
