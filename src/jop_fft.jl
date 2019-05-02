"""
    A = JopFft(T, n...[; dims=()])

where `A` is a Fourier transform, the domain of A is (T,n).  If `dims`
is not set, then `A` is an N-dimensional Fourier transform and where `N=ndims(dom)`.
Otherwise, `A` is a Fourier transform over the dimensions specifed by `dims`.

For a complex-to-complex transform: `range(A)::JetSpace`.  For a real-to-complex
transform: `range(A)::JetSpaceSymmetric`.  Normally, JopFft is responsible for constructing
range(A).  In the event that you need to manually construct this space, we provide
a convenience method:

    R = symspace(JopLn{JopFft_df!}, T, symdim, n...)

where `T` is the precision (generally `Float32` or `Float64`), `n...` is
the dimensions of the domain of `A`, and `symdim` is the first dimension
begin Fourier transformed.

# Examples

## 1D, real to complex
```julia
A = JopFft(Float64,128)
m = rand(domain(A))
d = A*m
```
## 1D, complex to complex
```julia
A = JopFft(ComplexF64,128)
m = rand(domain(A))
d = A*m
```

## 2D, real to complex
```julia
A = JopFft(Float64,128,256)
m = rand(domain(A))
d = A*m
```

## 3D, real to complex, transforming over only the first dimension
```julia
A = JopFft(Float64,128,256; dims=(1,))
m = rand(domain(A))
d = A*m
```
"""
function JopFft(::Type{T},n::Vararg{Int,N};dims=()) where {T<:Real,N}
    dims = isempty(dims) ? ntuple(i->i, N) : dims
    JopLn(dom = JetSpace(T,n), rng = symspace(typeof(JopFft_df!),T,dims[1],n...), df! = JopFft_df!, df′! = JopFft_df′!, s = (dims=dims,))
end
function JopFft(::Type{T},n::Vararg{Int,N};dims=()) where {T<:Complex,N}
    dims = isempty(dims) ? ntuple(i->i, N) : dims
    JopLn(dom = JetSpace(T,n), rng = JetSpace(T,n), df! = JopFft_df!, df′! = JopFft_df′!, s = (dims=dims,))
end
export JopFft

function JopFft_df!(d::AbstractArray{Complex{T}}, m::AbstractArray{T}; dims, kwargs...) where {T<:Real}
    P = plan_rfft(m, dims)
    JopFft_mul!(parent(d), P, m, dims)
    d
end

function JopFft_df!(d::AbstractArray{T}, m::AbstractArray{T}; dims, kwargs...) where {T<:Complex}
    P = plan_fft(m, dims)
    JopFft_mul!(d, P, m, dims)
    d
end

function JopFft_mul!(d::AbstractArray{T}, P::FFTW.FFTWPlan, m::AbstractArray, dims) where {T}
    mul!(d, P, m)
    sc = fftscale(T, size(m), dims)
    d .*= sc
end

function JopFft_df′!(m::AbstractArray{T}, d::AbstractArray; dims, kwargs...) where {T<:Real}
    P = plan_brfft(parent(d), size(m, dims[1]), dims)
    JopFft_mul_adjoint!(m, P, copy(parent(d)), dims) # in-place version of brfft modifies the input buffer
    m
end

function JopFft_df′!(m::AbstractArray{T}, d::AbstractArray; dims, kwargs...) where {T<:Complex}
    P = plan_bfft(d, dims)
    JopFft_mul_adjoint!(m, P, d, dims)
    m
end

function JopFft_mul_adjoint!(m::AbstractArray{T}, P::FFTW.FFTWPlan, d::AbstractArray, dims) where {T}
    mul!(m, P, d)
    sc = fftscale(T, size(m), dims)
    m .*= sc
end

fftscale(::Type{T}, n, dims) where {T} = T(1.0/sqrt(mapreduce(dim->n[dim], *, dims)))

function _JopFft_symspace_map(idim, I, symdim, n)
    if idim == symdim
        return 2 + n[idim] - I[idim]
    else
        return I[idim]
    end
end

function JopFft_symspace_map(I, symdim::Int, n::NTuple{N,Int}, _n::NTuple{N,Int}) where {N}
    if I[symdim] > _n[symdim]
        J = ntuple(idim->_JopFft_symspace_map(idim, I, symdim, n), N)
        return CartesianIndex(J)
    else
        return CartesianIndex(I)
    end
end

function Jets.symspace(JopFft, ::Type{T}, symdim::Int, n::Vararg{Int,N}) where {T,N}
    _n = ntuple(idim->idim == symdim ? div(n[idim], 2) + 1 : n[idim], N)
    Jets.JetSSpace(Complex{T}, n, _n, I->JopFft_symspace_map(I, symdim, n, _n))
end
