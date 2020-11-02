# set random seed to promote repeatability in CI unit tests
using Random
Random.seed!(101)

for file in (
        "jop_dct.jl",
        "jop_dwt.jl",
        "jop_fft.jl",
        "jop_sft.jl",
        "jop_slantstack.jl")
    include(file)
end
