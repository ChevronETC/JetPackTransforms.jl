using Jets, JetPackTransforms, LinearAlgebra, Printf, Test, FFTW

n1,n2 = 500,5
dt = 0.001

@testset "JopSft correctness test, T=$(T)" for T in (Float64,Float32)
    global n1,n2,dt
    f = fftfreq(n1) ./ dt
    freqs = T[f[5],f[9],f[13]]
    nfreq = length(freqs)
    dt = convert(T,dt)
    dom = JetSpace(T,n1,n2)
    rng = JetSpace(Complex{T},nfreq,n2)
    op = JopSft(dom,freqs,dt)
    x = zeros(domain(op))

    amp1 = 15.2873
    amp2 = 49.39874
    amp3 = 97.298743
    phs1 = 0.54321
    phs2 = 0.7654321
    phs3 = 1.87654321
    for k2 = 1:n2
        for k1 = 1:n1
            t = dt * (k1 - 1)
            x1 = amp1 * exp(im * (2 * pi * freqs[1] * t + phs1))
            x2 = amp2 * exp(im * (2 * pi * freqs[2] * t + phs2))
            x3 = amp3 * exp(im * (2 * pi * freqs[3] * t + phs3))
            x[k1,k2] = real(x1 + x2 + x3)
        end
    end

    y = op * x

    # actual amplitude and phase
    a1 = abs.(y[1,:])
    a2 = abs.(y[2,:])
    a3 = abs.(y[3,:])
    p1 = atan.(imag.(y[1,:]),real.(y[1,:]))
    p2 = atan.(imag.(y[2,:]),real.(y[2,:]))
    p3 = atan.(imag.(y[3,:]),real.(y[3,:]))

    rmsa1 = sqrt(norm(a1 .- amp1)^2/length(a1))
    rmsa2 = sqrt(norm(a2 .- amp2)^2/length(a2))
    rmsa3 = sqrt(norm(a3 .- amp3)^2/length(a3))

    rmsp1 = sqrt(norm(p1 .- phs1)^2/length(p1))
    rmsp2 = sqrt(norm(p2 .- phs2)^2/length(p2))
    rmsp3 = sqrt(norm(p3 .- phs3)^2/length(p3))

    @test rmsa1 < 1000 * eps(T)
    @test rmsa2 < 1000 * eps(T)
    @test rmsa3 < 1000 * eps(T)
    @test rmsp1 < 1000 * eps(T)
    @test rmsp2 < 1000 * eps(T)
    @test rmsp3 < 1000 * eps(T)
end

@testset "JopSft dot product test, T=$(T)" for T in (Float64,Float32)
    global n1,n2,dt
    f = fftfreq(n1) ./ dt
    freqs = T[f[5],f[9],f[13]]
    nfreq = length(freqs)
    dt = convert(T,dt)
    dom = JetSpace(T,n1,n2)
    rng = JetSpace(Complex{T},nfreq,n2)
    op = JopSft(dom,freqs,dt)
    x1 = rand(dom)
    y1 = rand(rng)
    y1 = op * rand(dom)
    lhs, rhs = dot_product_test(op, x1, y1)
    diff = abs((lhs - rhs) / (lhs + rhs))
    @test lhs ≈ rhs
end

@testset "JopSft linearity test, T=$(T)" for T in (Float64,Float32)
    global n1,n2,dt
    f = fftfreq(n1) ./ dt
    freqs = T[f[5],f[9],f[13]]
    nfreq = length(freqs)
    dt = convert(T,dt)
    dom = JetSpace(T,n1,n2)
    rng = JetSpace(Complex{T},nfreq,n2)
    op = JopSft(dom,freqs,dt)

    lhs,rhs = linearity_test(op)
    @test lhs ≈ rhs
end

nothing
