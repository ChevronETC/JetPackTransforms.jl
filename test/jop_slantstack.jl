using JetPackTransforms, Jets, Test

@testset "JetSlantStack, dot product test" for T in (Float32, Float64)
    A = JopSlantStack(JetSpace(T, 64, 128); dz=10.0, dh=10.0, h0=-1000.0)

    m = rand(domain(A))
    d = A*m

    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test isapprox(lhs,rhs,rtol=1e-4)

    A = JopSlantStack(JetSpace(T, 64, 128); dz=10.0, dh=10.0, h0=-1000.0, taperz=(0.3,0.3), taperh=(0.3,0.3), taperkz=(0.3,0.3), taperkh=(0.3,0.3))
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test isapprox(lhs,rhs,rtol=1e-4)
end

@testset "JetSlantStack, correctness" begin
    A = JopSlantStack(JetSpace(Float64, 64, 128); dz=10.0, dh=10.0, h0=-1000.0)
    m = zeros(domain(A))
    m[32,:] .= 1
    d = A*m
    v,i = findmax(d)
    @test i[1] == 32
    @test i[2] == findfirst(x->x≈0, state(A).cp)
end
