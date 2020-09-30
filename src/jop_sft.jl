"""
    op = JopSft(dom,freqs,dt)

Polychromatic slow Fourier transforms for a specified list of frequencies. Tranform
along the fast dimension of `dom::JetSpace{<:Real}`.

Notes:
    - it is expected that the domain is 2D real of size [nt,ntrace]
    - the range will be 2D complex of size [length(freqs),ntrace]
    - regarding the factor of (2/n): the factor "1/n" is from the Fourier transform, and the "2" is from Hermittian symmetry.
"""
function JopSft(dom::JetSpace{T}, freqs::Vector, dt) where {T<:Real}
    @assert length(size(dom)) == 2
    nf = length(freqs)
    n2 = size(dom,2)
    JopLn(dom = dom, rng = JetSpace(Complex{T}, nf, n2), df! = JopSft_df!, df′! = JopSft_df′!, s = (freqs=freqs, dt=dt))
end
export JopSft

function JopSft_df!(rngvec::AbstractArray{Complex{T},N}, domvec::AbstractArray{T,N}; freqs, dt, kwargs...) where {T,N}
    nf = length(freqs)
    n1 = size(domvec,1)
    _domvec = reshape(domvec, n1, :)
    _rngvec = reshape(rngvec, nf, :)
    @assert size(_domvec,2) == size(rngvec,2)
    n2 = size(_domvec,2)

    _rngvec .= 0
    for kfreq = 1:nf
        for k2 = 1:n2
            for k1 = 1:n1
                t = dt * (k1 - 1)
                phs = exp(-im * 2 * pi * freqs[kfreq] * t)
                _rngvec[kfreq,k2] += _domvec[k1,k2] * phs
            end
        end
    end

    _rngvec .*= (2/n1)
end

function JopSft_df′!(domvec::AbstractArray{T,N}, rngvec::AbstractArray{Complex{T},N}; freqs, dt, kwargs...) where {T,N}
    nf = length(freqs)
    n1 = size(domvec,1)
    _domvec = reshape(domvec, n1, :)
    _rngvec = reshape(rngvec, nf, :)
    @assert size(_domvec,2) == size(rngvec,2)
    n2 = size(_domvec,2)

    _domvec .= 0
    for kfreq = 1:nf
        for k2 = 1:n2
            for k1 = 1:n1
                t = dt * (k1 - 1)
                phs = exp(-im * 2 * pi * freqs[kfreq] * t)
                _domvec[k1,k2] += real(_rngvec[kfreq,k2] * conj(phs))
            end
        end
    end

    _domvec .*= (2/n1)
end
