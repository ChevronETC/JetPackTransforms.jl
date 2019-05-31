using Jets, JetPackTransforms, Test, Wavelets
# warning! This only tests the default Haar wavelet
@testset "dwt" begin
    # 1D transform Haar
    println("1D Haar")
    A = JopDwt(Float64, 128)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # 2D transform Haar
    println("2D Haar")
    A = JopDwt(JetSpace(Float64, 128, 128))
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # 3D transform HAar
    println("3D Haar")
    A = JopDwt(Float64, 128, 128, 128)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # db1
    wt = wavelet(WT.db1)
    # 1D transform
    println("1D $(wt.name)")
    A = JopDwt(Float64, 128; wt=wt)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # 2D transform
    println("2D $(wt.name)")
    A = JopDwt(Float64, 128, 128; wt=wt)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # 3D transform
    println("3D $(wt.name)")
    A = JopDwt(Float64, 128, 128, 128; wt=wt)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # db6
    wt = wavelet(WT.db6)
    # 1D transform
    println("1D $(wt.name)")
    A = JopDwt(Float64, 128; wt=wt)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # 2D transform
    println("2D $(wt.name)")
    A = JopDwt(Float64, 128, 128; wt=wt)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # 3D transform
    println("3D $(wt.name)")
    A = JopDwt(Float64, 128, 128, 128; wt=wt)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # db10
    wt = wavelet(WT.db10)
    # 1D transform
    println("1D $(wt.name)")
    A = JopDwt(Float64, 128; wt=wt)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # 2D transform
    println("2D $(wt.name)")
    A = JopDwt(Float64, 128, 128; wt=wt)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)

    # 3D transform
    println("3D $(wt.name)")
    A = JopDwt(Float64, 128, 128, 128; wt=wt)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test isapprox(lhs, rhs, rtol=1e-7)
end
