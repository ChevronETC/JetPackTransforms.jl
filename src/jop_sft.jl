"""
    op = JopSft(dom,freqs,dt)

Polychromatic slow Fourier transforms for a specified list of frequencies. Tranform
along the fast dimension of `dom::JetSpace{<:Real}`.

Note: regarding the factor of (2/n):
    the factor "1/n" is from the Fourier transform
    I think the "2" is from Hermittian symmetry, but not entirely sure
"""
function JopSft(dom::JetSpace{T}, freqs::Vector, dt) where {T<:Real}
    JopLn(dom = dom, rng = JetSpace(Complex{T}, size(dom)), df! = JopSft_df!, df′! = JopSft_df′!, s = (freqs=freqs, dt=dt))
end
export JopSft

function JopSft_df!(rngvec::AbstractArray{Complex{T},N}, domvec::AbstractArray{T,N}; freqs, dt, kwargs...) where {T,N}
    nfreq = length(freqs)
    n1 = size(domvec,1)
    _domvec = reshape(domvec, n1, :)
    _rngvec = reshape(rngvec, nfreq, :)
    n2 = size(_domvec,2)

    _rngvec .= 0
    for kfreq = 1:nfreq
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
    nfreq = length(freqs)
    n1 = size(domvec,1)
    _domvec = reshape(domvec, n1, :)
    _rngvec = reshape(rngvec, nfreq, :)
    n2 = size(_domvec,2)

    _domvec .= 0
    for kfreq = 1:nfreq
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
