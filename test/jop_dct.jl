using FFTW, Jets, JetPackTransforms, Test

@testset "DCT, 1D" begin
    A = JopDct(Float64, 128)
    lhs, rhs = dot_product_test(A, rand(domain(A)), rand(range(A)))
    @test lhs ≈ rhs

    m = rand(domain(A))
    d = A*m
    d_check = dct(m)
    d_check[1] *= sqrt(2)
    d_check[:] *= sqrt(1/length(m))
    @test d ≈ d_check
end

@testset "DCT, 2D" begin
    A = JopDct(Float64, 128, 64)
    lhs, rhs = dot_product_test(A, rand(domain(A)), rand(range(A)))
    @test lhs ≈ rhs

    m = rand(domain(A))
    d = A*m
    d_check = dct(m)
    d_check[1] *= 2*sqrt(2)
    d_check[:] *= sqrt(1/length(m))
    @test d_check ≈ d

    A = JopDct(Float64, 128, 64; dims=(1,))
    lhs, rhs = dot_product_test(A, rand(domain(A)), rand(range(A)))
    @test lhs ≈ rhs

    m = rand(domain(A))
    d = A*m
    d_check = dct(m,1)
    d_check[1] *= sqrt(2)
    d_check[:] *= sqrt(1/size(m,1))
    @test d ≈ d_check

    A = JopDct(Float64, 128, 64; dims=(2,))
    lhs, rhs = dot_product_test(A, rand(domain(A)), rand(range(A)))
    @test lhs ≈ rhs

    m = rand(domain(A))
    d = A*m
    d_check = dct(m,2)
    d_check[1] *= sqrt(2)
    d_check .*= sqrt(1/size(m,2))
    @test d ≈ d_check
end

@testset "DCT, 3D" begin
    A = JopDct(Float64, 128, 64, 32)
    lhs, rhs = dot_product_test(A, rand(domain(A)), rand(range(A)))
    @test lhs ≈ rhs

    m = rand(domain(A))
    d = A*m
    d_check = dct(m)
    d_check[1] *= 3*sqrt(2)
    d_check[:] *= sqrt(1/length(m))
    @test d ≈ d_check

    for dims in ((1,2), (1,3), (2,1), (2,3), (1,), (2,), (3,))
        A = JopDct(Float64, 128, 64, 32; dims=dims)
        lhs, rhs = dot_product_test(A, rand(domain(A)), rand(range(A)))
        @test lhs ≈ rhs

        m = rand(domain(A))
        d = A*m
        d_check = dct(m,dims)
        d_check[1] *= length(dims)*sqrt(2)
        d_check[:] *= sqrt(1/mapreduce(dim->size(m,dim), *, dims))
        @test d ≈ d_check
    end
end
