using FFTW, Jets, JetPackTransforms, Test

@testset "fft, 1D, complex" begin
    n = 512
    m = rand(n) + im * rand(n)
    d = rand(n) + im * rand(n)
    A = JopFft(ComplexF64, n)

    lhs, rhs = dot_product_test(A, m, d)
    @test lhs ≈ rhs
    expected = fft(m) / sqrt(n)
    observed = A * m
    @test expected ≈ observed
    expected = bfft(d) / sqrt(n)
    observed = A' * d
    @test expected ≈ observed
end

@testset "fft, alternative constructor" begin
    R = JetSpace(Float64,512)
    A = JopFft(R)
    B = JopFft(Float64,512)
    m = rand(R)
    @test A*m ≈ B*m
end

@testset "fft, 1D, real" begin
    n = 512
    spDom = JetSpace(Float64, n)
    spRng = symspace(JopFft, Float64, 1, n)
    m = rand(spDom)
    d = rand(spRng)
    d[1] = real(d[1])
    d[length(spRng)] = d[1]
    A = JopFft(Float64, n)
    @test range(A) == spRng
    lhs, rhs = dot_product_test(A, m, d)
    @test lhs ≈ rhs
end

@testset "fft, 2D" begin
    n1, n2 = 128, 256
    A = JopFft(ComplexF64, n1, n2)
    m = rand(domain(A))
    d = rand(range(A))
    lhs, rhs = dot_product_test(A, m, d)
    @test lhs ≈ rhs
end

@testset "fft, 2D, real->complex" begin
    n1, n2 = 128, 256
    spDom = JetSpace(Float64, n1, n2)
    spRng = symspace(JopFft, Float64, 1, n1, n2)
    m = rand(spDom)
    d = rand(spRng)
    A = JopFft(Float64,n1,n2)
    @test spRng == range(A)
    lhs, rhs = dot_product_test(A, m, d)
    @test lhs ≈ rhs

    m = rand(spDom)
    m_roundtrip = A' * (A * m)
    @test m ≈ m_roundtrip

    d = A * m
    d_check = rfft(m) / sqrt(length(m))
    @test parent(d) ≈ d_check

    d_copy = copy(d)
    mm = A' * d
    @test d_copy ≈ d
    mm_check = brfft(parent(d), n1) / sqrt(length(m))
    @test mm_check ≈ mm
    @test mm_check ≈ m
    @test mm ≈ m
end

@testset "fft, 1D transform of 2D array" begin
    n1, n2 = 128, 256
    A = JopFft(ComplexF64, n1, n2; dims=(1,))
    m = rand(domain(A))
    d = A*m
    d_expected = similar(d)
    for i2 = 1:n2
        d_expected[:,i2] = fft(vec(m[:,i2])) / sqrt(n1)
    end
    @test d ≈ d_expected
    a = A'*d
    a_expected = similar(a)
    for i2 = 1:n2
        a_expected[:,i2] = bfft(vec(d[:,i2])) / sqrt(n1)
    end
    @test a ≈ a_expected
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test lhs ≈ rhs
end

@testset "fft, 1D transform of 2D array, real->complex" begin
    n1, n2 = 128, 256
    A = JopFft(Float64, n1, n2; dims=(1,))
    m = rand(domain(A))
    d = A*m
    d_expected = similar(parent(d))
    for i2 = 1:n2
        d_expected[:,i2] = rfft(vec(m[:,i2])) / sqrt(n1)
    end
    @test parent(d) ≈ d_expected
    a = A'*d
    a_expected = similar(a)
    for i2 = 1:n2
        a_expected[:,i2] = brfft(vec(parent(d)[:,i2]),n1) / sqrt(n1)
    end
    @test a ≈ a_expected
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test lhs ≈ rhs
end

@testset "fft, 1D transform of 2D array, 2nd dim" begin
    n1, n2 = 128, 256
    A = JopFft(ComplexF64, n1, n2; dims=(2,))
    m = rand(domain(A))
    d = A*m
    d_expected = similar(d)
    for i1 = 1:n1
        d_expected[i1,:] = fft(vec(m[i1,:])) / sqrt(n2)
    end
    @test d ≈ d_expected
    a = A'*d
    a_expected = similar(a)
    for i1 = 1:n1
        a_expected[i1,:] = bfft(vec(d_expected[i1,:])) / sqrt(n2)
    end
    @test a ≈ a_expected
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test lhs ≈ rhs
end

@testset "fft, 1D transform of 2D array, 2nd dim, real->complex" begin
    n1, n2 = 128, 256
    A = JopFft(Float64, n1, n2; dims=(2,))
    m = rand(domain(A))
    d = A*m

    d_expected = similar(parent(d))
    for i1 = 1:n1
        d_expected[i1,:] = rfft(vec(m[i1,:])) / sqrt(n2)
    end
    @test parent(d) ≈ d_expected
    a = A'*d
    a_expected = similar(a)
    for i1 = 1:n1
        a_expected[i1,:] = brfft(vec(d_expected[i1,:]), n2) / sqrt(n2)
    end
    @test a ≈ a_expected
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test lhs ≈ rhs
end

@testset "fft, 2D of 3D array, real to complex along first two dims" begin
    n1,n2,n3 = 128,256,10
    A = JopFft(Float64,n1,n2,n3;dims=(1,2))
    m = rand(domain(A))
    d = A*m
    d_expected = similar(parent(d))
    for i3=1:n3
        d_expected[:,:,i3] = rfft(m[:,:,i3]) / sqrt(n1*n2)
    end
    @test parent(d) ≈ d_expected
    lhs, rhs = dot_product_test(A,rand(domain(A)),rand(range(A)))
    @test lhs ≈ rhs
end
