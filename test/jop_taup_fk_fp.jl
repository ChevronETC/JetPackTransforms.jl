using JetPackTransforms, Jets, LinearAlgebra, PyPlot, Test

# @testset "JopTauP_FK_FP, shift test" for T in (Float32, Float64)
@testset "JopTauP_FK_FP, shift test" for T in (Float32, )
    Δz = 10.0
    Δx = 10.0
    z0 = 0.0
    x0 = 0.0

    A = JopTauP_FK_FP(JetSpace(T, 101, 101); Δz=Δz, Δx=Δx, z0=z0, x0=x0, np=101)
    m = zeros(domain(A))
    nz,nx = size(domain(A))
    nz2,nx2 = div(nz,2), div(nx,2)
    box = 5
    m[nz2-box:nx2+box, nx2-box:nx2+box] .= 1.0
    @show extrema(m)
    d = A * m
    @show extrema(m)
    @show extrema(d)
    @show extrema(m .- d)

    xmin, xmax = x0, x0+Δx*(nx-1)
    zmin, zmax = z0, z0+Δz*(nz-1)
    extentXZ = [xmin,xmax,zmax,zmin]

    figure(figsize=(16,12))

    subplot(2,2,1); 
    imshow(m,aspect="auto", cmap="seismic", extent=extentXZ, clim=[-1,+1]);
    colorbar(orientation="vertical", label="Magnitude", pad=0.02, fraction=0.05, shrink=1.0)
    xlabel("X"); ylabel("Z"); title("Input");

    subplot(2,2,2); 
    imshow(d,aspect="auto", cmap="seismic", extent=extentXZ, clim=[-1,+1]);
    colorbar(orientation="vertical", label="Magnitude", pad=0.02, fraction=0.05, shrink=1.0)
    xlabel("X"); ylabel("Z"); title("Shifted");

    subplot(2,2,3); 
    imshow(m .- d,aspect="auto", cmap="seismic", extent=extentXZ, clim=[-1,+1]);
    colorbar(orientation="vertical", label="Magnitude", pad=0.02, fraction=0.05, shrink=1.0)
    xlabel("X"); ylabel("Z"); title("Input - Shifted");

    tight_layout()
    filename = "image-taup-shift.png" 
    savefig(filename, dpi=150)
end

# @testset "JopTauP_FK_FP, dot product test" for T in (Float32, Float64)
#     A = JopTauP_FK_FP(JetSpace(T, 64, 128); Δz=10.0, Δx=10.0, z0=0.0, x0=0.0)
#     lhs, rhs = dot_product_test(A, rand(domain(A)), rand(range(A)))
#     dif = (lhs - rhs) / (lhs + rhs)
#     @show lhs
#     @show rhs
#     @show  dif
#     @test isapprox(lhs,rhs,rtol=1e-4)
# end

# @testset "JopTauP_FK_FP, correctness" begin
#     A = JopTauP_FK_FP(JetSpace(Float64, 64, 128); dz=10.0, dh=10.0, h0=-1000.0)
#     m = zeros(domain(A))
#     m[32,:] .= 1
#     d = A*m
#     v,i = findmax(d)
#     @test i[1] == 32
#     @test i[2] == findfirst(x->x≈0, state(A).cp)
# end
