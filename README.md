
Jets.jl Transforms operator pack

- [`JetPackTransforms.JopDct`](README.md#JetPackTransforms.JopDct)
- [`JetPackTransforms.JopDwt`](README.md#JetPackTransforms.JopDwt)
- [`JetPackTransforms.JopFdct`](README.md#JetPackTransforms.JopFdct)
- [`JetPackTransforms.JopFft`](README.md#JetPackTransforms.JopFft)
- [`JetPackTransforms.JopSft`](README.md#JetPackTransforms.JopSft)
- [`JetPackTransforms.JopSlantStack`](README.md#JetPackTransforms.JopSlantStack)

<a id='JetPackTransforms.JopDct' href='#JetPackTransforms.JopDct'>#</a>
**`JetPackTransforms.JopDct`** &mdash; *Function*.



```julia
A = JopDct(T, n...[;dims=()])
```

where `A` is a type II discrete cosine transfrom (DCT), T is the element type of the domain and range if A, and n is the size of A.  If `dims` is not set, then `A` is the N-dimensional DCT and where `N=length(n)`.  Otherwise, `A` is a DCT over the dimensions specified by `dims`.

**Examples**

**1D**

```julia
A = JopDct(Float64, 128)
m = rand(domain(A))
d = A*m
```

**2D**

```julia
A = JopDct(Float64, 128, 64)
m = rand(domain(A))
d = A*m
```

**2D transform over 3D array**

```julia
A = JopDct(Float64,128,64,32;dims=(1,2))
m = rand(domain(A))
d = A*m
```

**Notes**

  * the adjoint pair is achieved by application `sqrt(2/N)` and `sqrt(2)` factors.  See:

      * https://en.wikipedia.org/wiki/Discrete*cosine*transform
      * http://www.fftw.org/fftw3*doc/1d-Real*002deven-DFTs-*0028DCTs*0029.html#g*t1d-Real*002deven-DFTs-*0028DCTs*0029

<a id='JetPackTransforms.JopDwt' href='#JetPackTransforms.JopDwt'>#</a>
**`JetPackTransforms.JopDwt`** &mdash; *Function*.



```julia
A = JopDwt(T, n...[, wt=wavelet(WT.haar, WT.Lifting)])
```

`A` is the N-D (N=1,2 or 3) wavelet transform operator, operating on the domain of type `T` and dimensions `n`.  The optional argument `wt` is the wavelet type. The supported wavelet types are `WT.haar` and `WT.db` (Daubechies).

**Notes**

  * For 2 and 3 dimensional transforms the domain must be square.  In other-words, `size(dom,1)==size(dom,2)==size(dom,3)`.
  * If your domain is rectangular, please consider using 1D wavelet transforms in combination wih the kronecker product (`JopKron`).
  * The wavelet transform is provide by the Wavelets.jl package: `http://github.com/JuliaDSP/Wavelets.jl`
  * You may try other wavelet classes supported by `Wavelets.jl`; however, these have yet to be tested for correctness with respect to the

transpose.

**Example**

```julia
A = JopDwt(Float64, 64, 64; wt=wavelet(WT.db5))
d = A*rand(domain(A))
```

<a id='JetPackTransforms.JopFdct' href='#JetPackTransforms.JopFdct'>#</a>
**`JetPackTransforms.JopFdct`** &mdash; *Function*.



```julia
A = JopFdct(N1, N2[; nbscales=-1, nbangles_coarse=16, ac=2])
```

`A` is the 2D curvelet transform operator, operating on domain that contains as two-dimensional array: `size(domain(A))=(N1,N2)`.  The optional arguments and their default values are:

  * `nbscales=-1` number of curvelet scales, if set to `-1`, then a heuristic is used

to determine the number of scales.

  * `nbangles_coarse=16` number of angles at the coarsest scales
  * `ac=2` if set to `2`, then the wavelets at the finest scale.  If set to `1`,

use curvelets at the finset scale.

<a id='JetPackTransforms.JopFft' href='#JetPackTransforms.JopFft'>#</a>
**`JetPackTransforms.JopFft`** &mdash; *Function*.



```julia
A = JopFft(T, n...[; dims=()])
```

where `A` is a Fourier transform, the domain of A is (T,n).  If `dims` is not set, then `A` is an N-dimensional Fourier transform and where `N=ndims(dom)`. Otherwise, `A` is a Fourier transform over the dimensions specifed by `dims`.

For a complex-to-complex transform: `range(A)::JetSpace`.  For a real-to-complex transform: `range(A)::JetSpaceSymmetric`.  Normally, JopFft is responsible for constructing range(A).  In the event that you need to manually construct this space, we provide a convenience method:

```
R = symspace(JopLn{JopFft_df!}, T, symdim, n...)
```

where `T` is the precision (generally `Float32` or `Float64`), `n...` is the dimensions of the domain of `A`, and `symdim` is the first dimension begin Fourier transformed.

**Examples**

**1D, real to complex**

```julia
A = JopFft(Float64,128)
m = rand(domain(A))
d = A*m
```

**1D, complex to complex**

```julia
A = JopFft(ComplexF64,128)
m = rand(domain(A))
d = A*m
```

**2D, real to complex**

```julia
A = JopFft(Float64,128,256)
m = rand(domain(A))
d = A*m
```

**3D, real to complex, transforming over only the first dimension**

```julia
A = JopFft(Float64,128,256; dims=(1,))
m = rand(domain(A))
d = A*m
```

<a id='JetPackTransforms.JopSft' href='#JetPackTransforms.JopSft'>#</a>
**`JetPackTransforms.JopSft`** &mdash; *Function*.



```julia
op = JopSft(dom,freqs,dt)
```

Polychromatic slow Fourier transforms for a specified list of frequencies. Tranform along the fast dimension of `dom::JetSpace{<:Real}`.

Note: regarding the factor of (2/n):     the factor "1/n" is from the Fourier transform     I think the "2" is from Hermittian symmetry, but not entirely sure

<a id='JetPackTransforms.JopSlantStack' href='#JetPackTransforms.JopSlantStack'>#</a>
**`JetPackTransforms.JopSlantStack`** &mdash; *Function*.



```julia
A = JopSlantStack(dom[; dz=10.0, dh=10.0, h0=-1000.0, ...])
```

where `A` is the 2D slant-stack operator mapping for `z-h` to `tau-p`.  The domain of the operator is `nz` x `nh` with precision T, `dz` is the depth spacing, `dh` is the half-offset spacing, and `h0` is the origin of the half-offset axis.  The additional named optional arguments along with their default values are,

  * `theta=-60:1.0:60` - range of opening angles.  The ray parameter is: p=sin(theta/2)/c
  * `padz=0.0,padh=0.0` - padding in depth and offset to apply before applying the Fourier transfrom
  * `taperz=(0,0)` - beginning and end taper in the z-direction before transforming from `z-h` to `kz-kh`
  * `taperh=(0,0)` - beginning and end taper in the h-direction before transforming from `z-h` to `kz-kh`
  * `taperkz=(0,0)` - beginning and end taper in the kz-direction before transforming from `kz-kh` to `z-h`
  * `taperkh=(0,0)` - beginning and end taper in the kh-direction before transforming from `kz-kh` to `z-h`

**Notes**

  * It should be possible to extend this operator to 3D
  * If your domain is t-h rather than z-h, you can still use this operator

