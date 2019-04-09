using Jets, JetPackTransforms, Test

@testset "fdct" begin
    A = JopFdct(128, 256)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test lhs ≈ rhs rtol=1e-7

    A = JopFdct(128, 256, t=Float64)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test lhs ≈ rhs rtol=1e-7
end
