"""
    A = JopDwt(T, n...[, wt=wavelet(WT.haar, WT.Lifting)])

`A` is the N-D (N=1,2 or 3) wavelet transform operator, operating on the domain
of type `T` and dimensions `n`.  The optional argument `wt` is the wavelet type.
The supported wavelet types are `WT.haar` and `WT.db` (Daubechies).

# Notes
* For 2 and 3 dimensional transforms the domain must be square.  In other-words, `size(dom,1)==size(dom,2)==size(dom,3)`.
* If your domain is rectangular, please consider using 1D wavelet transforms in combination wih the kronecker product (`JopKron`).
* The wavelet transform is provide by the Wavelets.jl package: `http://github.com/JuliaDSP/Wavelets.jl`
* You may try other wavelet classes supported by `Wavelets.jl`; however, these have yet to be tested for correctness with respect to the
transpose.

# Example
```julia
A = JopDwt(Float64, 64, 64; wt=wavelet(WT.db5))
d = A*rand(domain(A))
```
"""
function JopDwt(::Type{T}, n::Vararg{N}; wt=wavelet(WT.haar, WT.Lifting)) where {T,N}
    @assert length(n) <= 3
    map(i->@assert(n[i]==n[1]), 2:length(n))
    @assert n[1] == nextpow(2, n[1])
    wt.name != "haar" && startswith(wt.name, "db") != true && error("JopDwt only supports Haar and Daubechies wavelet functions")
    JopLn(dom = JetSpace(T,n), rng = JetSpace(T,n), df! = JopDwt_df!, df′! = JopDwt_df′!, s = (wt=wt,))
end
export JopDwt

function JopDwt_df!(d::AbstractArray, m::AbstractArray; wt, kwargs...)
    # dwt! does not work without copy for all wt
    # if ok with copy and gc
    copyto!(d, dwt(m, wt))
    #copyto!(d, m)
    #dwt!(d, op.wt, maxtransformlevels(m))
end

function JopDwt_df′!(m::AbstractArray, d::AbstractArray; wt, kwargs...)
    # idwt! does not work without copy for all wt
    # if ok with copy and gc
    copyto!(m, idwt(d, wt))
    #copyto!(m, d)
    #idwt!(m, op.wt, maxtransformlevels(d))
end
