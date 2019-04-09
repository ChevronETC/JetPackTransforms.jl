"""
    A = JopDct(T, n...[;dims=()])

where `A` is a type II discrete cosine transfrom (DCT), T is the element type of
the domain and range if A, and n is the size of A.  If `dims` is not set, then `A`
is the N-dimensional DCT and where `N=length(n)`.  Otherwise, `A` is a DCT
over the dimensions specified by `dims`.

# Examples

## 1D
```julia
A = JopDct(Float64, 128)
m = rand(domain(A))
d = A*m
```

## 2D
```julia
A = JopDct(Float64, 128, 64)
m = rand(domain(A))
d = A*m
```

## 2D transform over 3D array
```julia
A = JopDct(Float64,128,64,32;dims=(1,2))
m = rand(domain(A))
d = A*m
```

# Notes
* the adjoint pair is achieved by application `sqrt(2/N)` and `sqrt(2)` factors.  See:
    * https://en.wikipedia.org/wiki/Discrete_cosine_transform
    * http://www.fftw.org/fftw3_doc/1d-Real_002deven-DFTs-_0028DCTs_0029.html#g_t1d-Real_002deven-DFTs-_0028DCTs_0029
"""
function JopDct(T, n...; dims=())
    dims = dims == () ? ntuple(i->i, length(n)) : dims
    JopLn(df! = JopDct_df!, df′! = JopDct_df′!, dom = JetSpace(T, n), rng = JetSpace(T, n), s = (dims=dims,))
end
export JopDct

function JopDct_df!(d::AbstractArray{T}, m::AbstractArray{T}; dims, kwargs...) where {T}
    d .= m
    dct!(d, dims)
    sc1, sc = dctscale(dims, size(m), T)
    d[1] *= sc1
    d .*= sc
end

function JopDct_df′!(m::AbstractArray{T}, d::AbstractArray{T}; dims, kwargs...) where {T}
    m .= d
    sc1, sc = dctscale(dims, size(m), T)
    m[1] *= sc1
    idct!(m, dims)
    m .*= sc
end

dctscale(dims, n, ::Type{T}) where {T} = length(dims)*sqrt(T(2)), T(sqrt(1.0/mapreduce(dim->n[dim], *, dims)))
