using JetPackTransforms, Jets, Test, LinearAlgebra

# depth mode
@testset "JetSlantStack, dot product test" for T in (Float32, Float64)
    A = JopSlantStack(JetSpace(T, 64, 128); dz=10.0, dh=10.0, h0=-1000.0)

    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test isapprox(lhs,rhs,rtol=1e-4)

    A = JopSlantStack(JetSpace(T, 64, 128); dz=10.0, dh=10.0, h0=-1000.0, taperz=(0.3,0.3), taperh=(0.3,0.3), taperkz=(0.3,0.3), taperkh=(0.3,0.3))
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test isapprox(lhs,rhs,rtol=1e-4)
end

@testset "JetSlantStack, correctness" for padz in (0.0, 0.2)
    A = JopSlantStack(JetSpace(Float64, 64, 128); dz=10.0, dh=10.0, h0=-1000.0, padz)
    m = zeros(domain(A))
    m[32,:] .= 1
    d = A*m
    v,i = findmax(d)
    @test i[1] == 32
    @test i[2] == findfirst(x->x≈0, state(A).tant)
end

# time mode
@testset "JetSlantStack, dot product test" for T in (Float32, Float64)
    A = JopSlantStack(JetSpace(T, 64, 128); dz=0.004, dh=10.0, h0=-1000.0, mode="time")

    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test isapprox(lhs,rhs,rtol=1e-4)

    A = JopSlantStack(JetSpace(T, 64, 128); dz=0.004, dh=10.0, h0=-1000.0, taperz=(0.3,0.3), taperh=(0.3,0.3), taperkz=(0.3,0.3), taperkh=(0.3,0.3), mode="time")
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test isapprox(lhs,rhs,rtol=1e-4)
end

@testset "JetSlantStack, correctness" for padz in (0.0, 0.2)
    A = JopSlantStack(JetSpace(Float64, 64, 128); dz=0.004, dh=10.0, h0=-1000.0, padz, mode="time")
    m = zeros(domain(A))
    m[32,:] .= 1
    d = A*m
    v,i = findmax(d)
    @test i[1] == 32
end

# depth-time parity test
@testset "JetSlantStack, parity" begin
    theta = collect(-45:0.5:45)
    # the ray parameters that would give the same results as theta is
    p = @. -tan(deg2rad(theta))
    
    Ad = JopSlantStack(JetSpace(Float64, 128, 129); mode="depth", theta=theta, p=p, dz=0.01, dh=0.01, h0=-0.64, padz=1, padh=1, taperkz = (0.5,0.5), taperkh = (0.5,0.5))
    At = JopSlantStack(JetSpace(Float64, 128, 129); mode="time" , theta=theta, p=p, dz=0.01, dh=0.01, h0=-0.64, padz=1, padh=1, taperkz = (0.5,0.5), taperkh = (0.5,0.5))
    m = zeros(domain(Ad))
    for i = 1:129
        m[div(i,4)+20,i] = 1.0
    end
    m[16,64] = 1
    m[64,96] = 1
    dd = Ad*m
    dt = At*m

    err_d = maximum(abs.(dd .- dt)) / maximum(abs.(dd))
    @show err_d
    @test err_d < 1e-7
end

@testset "JetSlantStackShiftSum, dot product test" for T in (Float32, Float64), mode in ("depth", "time")
    A = JopSlantStackShiftSum(JetSpace(T, 64, 128); dz=10.0, mode=mode)

    A = JopSlantStackShiftSum(JetSpace(T, 64, 128); dz=10.0, mode=mode, h = 10.0 .* rand(128) .- 5)
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test isapprox(lhs,rhs,rtol=1e-4)
end

@testset "JetSlantStackShiftSum, parity" begin
    theta = collect(-45:0.5:45)
    # the ray parameters that would give the same results as theta is
    p = @. -tan(deg2rad(theta))
    
    Ad = JopSlantStackShiftSum(JetSpace(Float64, 128, 129); mode="depth", theta=theta, p=p, dz=0.01, h=collect(-0.64:0.01:0.64))
    At = JopSlantStackShiftSum(JetSpace(Float64, 128, 129); mode="time" , theta=theta, p=p, dz=0.01, h=collect(-0.64:0.01:0.64))
    m = zeros(domain(Ad))
    for i = 1:129
        m[div(i,4)+20,i] = 1.0
    end
    m[16,64] = 1
    m[64,96] = 1
    dd = Ad*m
    dt = At*m

    err_d = maximum(abs.(dd .- dt)) / maximum(abs.(dd))

    @show err_d
    @test err_d < 1e-7
end

@testset "JetSlantStack vs JetSlantStackShiftSum, parity" begin
    theta = collect(-45:0.5:45)
    A1 = JopSlantStack(JetSpace(Float64, 128, 129); mode="depth", theta=theta, dz=0.01, h0=-0.64, dh=0.01)
    A2 = JopSlantStackShiftSum(JetSpace(Float64, 128, 129); mode="depth" , theta=theta, dz=0.01, h=collect(-0.64:0.01:0.64))
    m = zeros(domain(A1))
    for i = 1:129
        m[div(i,4)+20,i] = 1.0
    end
    m[16,64] = 1
    m[64,96] = 1
    d1 = A1*m
    d2 = A2*m

    # check cosine similarity of the results since the two operators are not exactly the same
    similarity = dot(d1, d2) / (norm(d1) * norm(d2))

    @show similarity
    @test similarity > 0.95
end

@testset "JetSlantStackShiftSum3D, dot product test, T = $(T), mode = $(mode), dip = $(dip)" for T in (Float32, Float64), mode in ("depth", "time"), (dip, azimuth) in ((0.0,0.0), (30.0, 60.0), ([0.0], [0.0]))
    if isa(dip, Array)
        dip = 90 .* rand(32)
        azimuth = 360 .* rand(32)
    end
    A = JopSlantStackShiftSum3D(JetSpace(T, 32, 64); dz=10.0, mode=mode, hx = 10.0 .* rand(64) .- 5, hy = 10.0 .* rand(64) .- 5,  dip=dip, azimuth=azimuth)
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test isapprox(lhs,rhs,rtol=1e-4)

    A = JopSlantStackShiftSum3D(JetSpace(T, 32, 8, 8); dz=10.0, mode=mode, hx = 10.0 .* rand(8) .- 5, hy = 10.0 .* rand(8) .- 5,  dip=dip, azimuth=azimuth)
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test isapprox(lhs,rhs,rtol=1e-4)
end

@testset "JetSlantStackShiftSum3D, time-depth parity" begin
    theta = collect(-45:0.5:45)
    phi = [0.0, 90.0]
    
    # the ray parameters that would give the same results as theta/phi are
    px = @. -tan(deg2rad(theta))
    py = [0.0]
    
    Ad = JopSlantStackShiftSum3D(JetSpace(Float64, 64, 129); mode="depth", theta=theta, phi = phi, px=px, py=py, dz=0.01, hx=collect(-0.64:0.01:0.64), hy=collect(-0.64:0.01:0.64))
    At1 = JopSlantStackShiftSum3D(JetSpace(Float64, 64, 129); mode="time" , theta=theta, phi = phi, px=px, py=py, dz=0.01, hx=collect(-0.64:0.01:0.64), hy=collect(-0.64:0.01:0.64))
    At2 = JopSlantStackShiftSum3D(JetSpace(Float64, 64, 129); mode="time" , theta=theta, phi = phi, px=py, py=px, dz=0.01, hx=collect(-0.64:0.01:0.64), hy=collect(-0.64:0.01:0.64))
    m = zeros(domain(Ad))
    for i = 1:129
        m[div(i,4)+20,i] = 1.0
    end
    m[16,64] = 1
    m[32,96] = 1
    dd = Ad*m
    dt1 = At1*m
    dt2 = At2*m

    err_d1 = maximum(abs.(dd[:,:,1] .- dt1[:,:,1])) / maximum(abs.(dd[:,:,1]))
    err_d2 = maximum(abs.(dd[:,:,2] .- dt2[:,1,:])) / maximum(abs.(dd[:,:,2]))

    @show err_d1, err_d2
    @test err_d1 < 1e-7
    @test err_d2 < 1e-7
end

@testset "JetSlantStackShiftSum3D, dip parity" begin
    theta = collect(-45:0.5:45)
    phi = collect(0.0:45.0:135.0)
    
    A1 = JopSlantStackShiftSum3D(JetSpace(Float64, 64, 129); mode="depth", theta=theta, phi = phi, dz=0.01, hx=collect(-0.64:0.01:0.64), hy=collect(-0.64:0.01:0.64), dip = 0.0, azimuth = 0.0)
    A2 = JopSlantStackShiftSum3D(JetSpace(Float64, 64, 129); mode="depth", theta=theta, phi = phi, dz=0.01, hx=collect(-0.64:0.01:0.64), hy=collect(-0.64:0.01:0.64), dip = zeros(64), azimuth = zeros(64))
    m = zeros(domain(A1))
    for i = 1:129
        m[div(i,4)+20,i] = 1.0
    end
    m[16,64] = 1
    m[32,96] = 1
    d1 = A1*m
    d2 = A2*m

    err_d = maximum(abs.(d1 .- d2)) / maximum(abs.(d1))

    @show err_d
    @test err_d < 1e-7

    A1 = JopSlantStackShiftSum3D(JetSpace(Float64, 64, 129); mode="depth", theta=theta, phi = phi, dz=0.01, hx=collect(-0.64:0.01:0.64), hy=collect(-0.64:0.01:0.64), dip = 30.0, azimuth = 45.0)
    A2 = JopSlantStackShiftSum3D(JetSpace(Float64, 64, 129); mode="depth", theta=theta, phi = phi, dz=0.01, hx=collect(-0.64:0.01:0.64), hy=collect(-0.64:0.01:0.64), dip = 30 .+ zeros(64), azimuth = 45 .+ zeros(64))

    d1 = A1*m
    d2 = A2*m

    err_d = maximum(abs.(d1 .- d2)) / maximum(abs.(d1))

    @show err_d
    @test err_d < 1e-7
end

@testset "JetSlantStackShiftSum3D, offsets parity" begin
    theta = collect(-45:5.0:45)
    phi = collect(0.0:45.0:135.0)

    hx_reg = collect(-0.6:0.1:0.6)
    hy_reg = collect(-0.3:0.1:0.2)
    nhx = length(hx_reg)
    nhy = length(hy_reg)
    nh = nhx * nhy
    hx_irreg = repeat(hx_reg, outer=nhy)
    hy_irreg = repeat(hy_reg, inner=nhx)
    
    A1 = JopSlantStackShiftSum3D(JetSpace(Float64, 64, nhx, nhy); mode="depth", theta=theta, phi = phi, dz=0.01, hx=hx_reg, hy=hy_reg, dip = 30.0, azimuth = 45.0)
    A2 = JopSlantStackShiftSum3D(JetSpace(Float64, 64, nh); mode="depth", theta=theta, phi = phi, dz=0.01, hx=hx_irreg, hy=hy_irreg, dip = 30.0, azimuth = 45.0)
    m1 = zeros(domain(A1))
    for i = 1:nhy
        m1[div(i,4)+20,:,i] .= 1.0
    end
    m1[16,3,2] = 1
    m1[32,2, 4] = -1
    m2 = reshape(m1, 64, nh)
    
    d1 = A1*m1
    d2 = A2*m2

    err_d = maximum(abs.(d1 .- d2)) / maximum(abs.(d1))

    @show err_d
    @test err_d < 1e-7
end