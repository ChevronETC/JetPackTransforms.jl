| **Documentation**                                                               |
|:-------------------------------------------------------------------------------:|
| [![](https://img.shields.io/badge/docs-dev-blue.svg)](https://chevronetc.github.io/JetPackTransforms.jl/dev/) |
# JetPackTransforms.jl

This package contains transform operators for Jets.jl that depend on FFTW.jl,
CurveLab.jl and Wavelets.jl.

# Jet transforms operator pack

* JetPackTransforms.JopDct  - N-dimensional discrete cosine transform
* JetPackTransforms.JopDwt  - N-dimensional discrete wavelet transform
* JetPackTransforms.JopFdct - 2D discrete discrite curvelet transform
* JetPackTransforms.JopFft  - N-dimensional Fast Fourier transform
* JetPackTransforms.JopSft  - N-dimensional Slow Fourier transform
* JetPackTransforms.JopSlantStack - 2D slant stack transform