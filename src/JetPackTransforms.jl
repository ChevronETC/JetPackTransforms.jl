module JetPackTransforms

using CurveLab, FFTW, JetPack, Jets, LinearAlgebra, Wavelets

include("jop_dct.jl")
include("jop_dwt.jl")
include("jop_fdct.jl")
include("jop_fft.jl")
include("jop_sft.jl")
include("jop_slantstack.jl") # requires JopTaper from JetPack

end
