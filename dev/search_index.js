var documenterSearchIndex = {"docs":
[{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [JetPackTransforms]\nOrder = [:function]","category":"page"},{"location":"reference/#JetPackTransforms.JopDct-Tuple{Any,Vararg{Any,N} where N}","page":"Reference","title":"JetPackTransforms.JopDct","text":"A = JopDct(T, n...[;dims=()])\n\nwhere A is a type II discrete cosine transfrom (DCT), T is the element type of the domain and range if A, and n is the size of A.  If dims is not set, then A is the N-dimensional DCT and where N=length(n).  Otherwise, A is a DCT over the dimensions specified by dims.\n\nExamples\n\n1D\n\nA = JopDct(Float64, 128)\nm = rand(domain(A))\nd = A*m\n\n2D\n\nA = JopDct(Float64, 128, 64)\nm = rand(domain(A))\nd = A*m\n\n2D transform over 3D array\n\nA = JopDct(Float64,128,64,32;dims=(1,2))\nm = rand(domain(A))\nd = A*m\n\nNotes\n\nthe adjoint pair is achieved by application sqrt(2/N) and sqrt(2) factors.  See:\nhttps://en.wikipedia.org/wiki/Discretecosinetransform\nhttp://www.fftw.org/fftw3doc/1d-Real002deven-DFTs-0028DCTs0029.html#gt1d-Real002deven-DFTs-0028DCTs0029\n\n\n\n\n\n","category":"method"},{"location":"reference/#JetPackTransforms.JopDwt-Union{Tuple{Jets.JetSpace}, Tuple{N}, Tuple{T}} where N where T","page":"Reference","title":"JetPackTransforms.JopDwt","text":"A = JopDwt(T, n...[, wt=wavelet(WT.haar, WT.Lifting)])\n\nA is the N-D (N=1,2 or 3) wavelet transform operator, operating on the domain of type T and dimensions n.  The optional argument wt is the wavelet type. The supported wavelet types are WT.haar and WT.db (Daubechies).\n\nNotes\n\nFor 2 and 3 dimensional transforms the domain must be square.  In other-words, size(dom,1)==size(dom,2)==size(dom,3).\nIf your domain is rectangular, please consider using 1D wavelet transforms in combination wih the kronecker product (JopKron).\nThe wavelet transform is provide by the Wavelets.jl package: http://github.com/JuliaDSP/Wavelets.jl\nYou may try other wavelet classes supported by Wavelets.jl; however, these have yet to be tested for correctness with respect to the\n\ntranspose.\n\nExample\n\nA = JopDwt(Float64, 64, 64; wt=wavelet(WT.db5))\nd = A*rand(domain(A))\n\n\n\n\n\n","category":"method"},{"location":"reference/#JetPackTransforms.JopFft-Union{Tuple{N}, Tuple{T}, Tuple{Type{T},Vararg{Int64,N}}} where N where T<:Real","page":"Reference","title":"JetPackTransforms.JopFft","text":"A = JopFft(T, n...[; dims=()])\n\nwhere A is a Fourier transform, the domain of A is (T,n).  If dims is not set, then A is an N-dimensional Fourier transform and where N=ndims(dom). Otherwise, A is a Fourier transform over the dimensions specifed by dims.\n\nFor a complex-to-complex transform: range(A)::JetSpace.  For a real-to-complex transform: range(A)::JetSpaceSymmetric.  Normally, JopFft is responsible for constructing range(A).  In the event that you need to manually construct this space, we provide a convenience method:\n\nR = symspace(JopLn{JopFft_df!}, T, symdim, n...)\n\nwhere T is the precision (generally Float32 or Float64), n... is the dimensions of the domain of A, and symdim is the first dimension begin Fourier transformed.\n\nExamples\n\n1D, real to complex\n\nA = JopFft(Float64,128)\nm = rand(domain(A))\nd = A*m\n\n1D, complex to complex\n\nA = JopFft(ComplexF64,128)\nm = rand(domain(A))\nd = A*m\n\n2D, real to complex\n\nA = JopFft(Float64,128,256)\nm = rand(domain(A))\nd = A*m\n\n3D, real to complex, transforming over only the first dimension\n\nA = JopFft(Float64,128,256; dims=(1,))\nm = rand(domain(A))\nd = A*m\n\n\n\n\n\n","category":"method"},{"location":"reference/#JetPackTransforms.JopSft-Union{Tuple{T}, Tuple{Jets.JetSpace{T,N} where N,Array{T,1} where T,Any}} where T<:Real","page":"Reference","title":"JetPackTransforms.JopSft","text":"op = JopSft(dom,freqs,dt)\n\nPolychromatic slow Fourier transforms for a specified list of frequencies. Tranform along the fast dimension of dom::JetSpace{<:Real}.\n\nNotes:     - it is expected that the domain is 2D real of size [nt,ntrace]     - the range will be 2D complex of size [length(freqs),ntrace]     - regarding the factor of (2/n): the factor \"1/n\" is from the Fourier transform, I think the \"2\" is from Hermittian symmetry, but not entirely sure\n\n\n\n\n\n","category":"method"},{"location":"reference/#JetPackTransforms.JopSlantStack-Union{Tuple{Jets.JetAbstractSpace{T,N} where N}, Tuple{T}} where T","page":"Reference","title":"JetPackTransforms.JopSlantStack","text":"A = JopSlantStack(dom[; dz=10.0, dh=10.0, h0=-1000.0, ...])\n\nwhere A is the 2D slant-stack operator mapping for z-h to tau-p.  The domain of the operator is nz x nh with precision T, dz is the depth spacing, dh is the half-offset spacing, and h0 is the origin of the half-offset axis.  The additional named optional arguments along with their default values are,\n\ntheta=-60:1.0:60 - range of opening angles.  The ray parameter is: p=sin(theta/2)/c\npadz=0.0,padh=0.0 - padding in depth and offset to apply before applying the Fourier transfrom\ntaperz=(0,0) - beginning and end taper in the z-direction before transforming from z-h to kz-kh\ntaperh=(0,0) - beginning and end taper in the h-direction before transforming from z-h to kz-kh\ntaperkz=(0,0) - beginning and end taper in the kz-direction before transforming from kz-kh to z-h\ntaperkh=(0,0) - beginning and end taper in the kh-direction before transforming from kz-kh to z-h\n\nNotes\n\nIt should be possible to extend this operator to 3D\nIf your domain is t-h rather than z-h, you can still use this operator\n\n\n\n\n\n","category":"method"},{"location":"reference/#Index","page":"Reference","title":"Index","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"#JetPackTransforms.jl","page":"JetPackTransforms.jl","title":"JetPackTransforms.jl","text":"","category":"section"},{"location":"","page":"JetPackTransforms.jl","title":"JetPackTransforms.jl","text":"This package contains transform operators for Jets.jl. It depends on FFTW.jl and Wavelets.jl.","category":"page"}]
}
