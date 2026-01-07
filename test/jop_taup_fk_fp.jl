using JetPackTransforms, Jets, LinearAlgebra, PyPlot, Serialization, Test

# @testset "JopTauP_FK_FP, shift test" for T in (Float32, ), perturb in (1,2,3,4,5)
@testset "JopTauP_FK_FP, shift test" for T in (Float32, ), perturb in (5,)
    io = open("shot_gather.jls", "r")
    (data,t0,x0,Δt,Δx) = deserialize(io)
    close(io)
    nt,nx = size(data)
    @show t0,x0,Δt,Δx
    x0 = +3.0
    @show t0,x0,Δt,Δx

    A = JopTauP_FK_FP(JetSpace(T, nt, nx); t0=t0, x0=x0, Δt=Δt, Δx=Δx, padt=2, padx=2, 
        taperT=(0.05,0.05), taperX=(0.05,0.05), np=501, vmin=1000.0, interpsinc=1)
    m = zeros(domain(A))
    if perturb == 1
        m[div(nt,2)-1:div(nt,2)+1,div(nx,2)-1:div(nx,2)+1] .= 1
    elseif perturb == 2
        m[div(nt,3)-1:div(nt,3)+1,div(nx,2)-1:div(nx,2)+1] .= 1
    elseif perturb == 3
        m[div(nt,2)-1:div(nt,2)+1,div(nx,3)-1:div(nx,3)+1] .= 1
    elseif perturb == 4
        m[div(nt,3)-1:div(nt,3)+1,div(nx,3)-1:div(nx,3)+1] .= 1
    elseif perturb == 5
        m .= data
    end

    d = A * m
    m2 = A' * d

    xmin, xmax = x0, x0+Δx*(nx-1)
    tmin, tmax = t0, t0+Δt*(nt-1)
    extentXZ = [xmin,xmax,tmax,tmin]

    scale = 3

    figure(figsize=(16,8))

    subplot(1,3,1); 
    imshow(scale .* m ./ maximum(abs.(m)),aspect="auto", cmap="seismic", extent=extentXZ, clim=[-1,+1]);
    xlabel("X"); ylabel("Z"); title("Input T-X");

    subplot(1,3,2); 
    imshow(scale .* m2 ./ maximum(abs.(m2)),aspect="auto", cmap="seismic", extent=extentXZ, clim=[-1,+1]);
    xlabel("X"); ylabel("Z"); title("Round Trip T-X");

    subplot(1,3,3); 
    imshow(scale .* d ./ maximum(abs.(d)),aspect="auto", cmap="seismic", extent=extentXZ, clim=[-1,+1]);
    xlabel("X"); ylabel("Z"); title("Forward Tau-P");

    tight_layout()
    filename = "image-taup-shift-$(perturb).png" 
    savefig(filename, dpi=150)
end

@test_skip @testset "JopTauP_FK_FP, dot product test" for T in (Float32, Float64)
    A = JopTauP_FK_FP(JetSpace(T, 64, 128); Δt=0.005, Δx=10.0, t0=0.0, x0=0.0)
    lhs, rhs = dot_product_test(A, rand(domain(A)), rand(range(A)))
    dif = (lhs - rhs) / (lhs + rhs)
    @show lhs
    @show rhs
    @show  dif
    @test isapprox(lhs,rhs,rtol=1e-4)
end

# @testset "JopTauP_FK_FP, correctness" begin
#     A = JopTauP_FK_FP(JetSpace(Float64, 64, 128); dz=10.0, dh=10.0, h0=-1000.0)
#     m = zeros(domain(A))
#     m[32,:] .= 1
#     d = A*m
#     v,i = findmax(d)
#     @test i[1] == 32
#     @test i[2] == findfirst(x->x≈0, state(A).cp)
# end
