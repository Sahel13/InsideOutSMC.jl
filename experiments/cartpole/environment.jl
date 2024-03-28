module CartpoleEnvironment

    using Random
    using Distributions
    using LinearAlgebra
    using StatsBase

    using InsideOutSMC: StochasticDynamics
    using InsideOutSMC: IBISDynamics

    function drift_fn(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
    )
        l, mp, mc = p
        g = 9.81

        c, k = 1e-2, 1e-2
        d, v = 1e-2, 1e-2

        sin_q = sin(x[2])
        cos_q = cos(x[2])

        _, _, ds, dq = x

        dds = (
            only(u)
            - (c * x[3] + k * x[1])
            - (d * x[4] + v * x[2]) * cos_q / l
            + mp * sin_q * (l * x[4]^2 + g * cos_q)
        ) / (mc + mp * sin_q^2)
        ddq = (
            - only(u) * cos_q
            - mp * l * x[4]^2 * cos_q * sin_q
            - (mc + mp) * g * sin_q
            - (c * x[3] + k * x[1]) * cos_q
            - (d * x[4] + v * x[2]) * cos_q^2 / l
        ) / (l * mc + l * mp * sin_q^2)

        return [ds, dq, dds, ddq]
    end

    function drift_fn!(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
        xn::AbstractVector{Float64},
    )
        l, mp, mc = p
        g = 9.81

        c, k = 1e-2, 1e-2
        d, v = 1e-2, 1e-2

        sin_q = sin(x[2])
        cos_q = cos(x[2])

        xn[1] = x[3]
        xn[2] = x[4]
        xn[3] = (
            only(u)
            - (c * x[3] + k * x[1])
            - (d * x[4] + v * x[2]) * cos_q / l
            + mp * sin_q * (l * x[4]^2 + g * cos_q)
        ) / (mc + mp * sin_q^2)
        xn[4] = (
            - only(u) * cos_q
            - mp * l * x[4]^2 * cos_q * sin_q
            - (mc + mp) * g * sin_q
            - (c * x[3] + k * x[1]) * cos_q
            - (d * x[4] + v * x[2]) * cos_q^2 / l
        ) / (l * mc + l * mp * sin_q^2)
    end

    function diffusion_fn(
        args::AbstractVector{Float64}...,
    )::Vector{Float64}
        return [0.0, 0.0, 1e-1, 0.0]
    end

    xdim = 4
    udim = 1

    step_size = 0.05
    init_state = [0.0, 0.0, 0.0, 0.0, 0.0]

    true_params = [1.0, 1.0, 1.0]

    param_prior = MvLogNormal(
        MvNormal(
            [0.0, 0.0, 0.0],
            Diagonal([0.1, 0.1, 0.1].^2)
        )
    )

    function param_proposal(
        particles::AbstractMatrix{Float64},
        weights::AbstractVector{Float64},
        constant::Float64=1.0
    )::Matrix{Float64}
        log_particles = log.(particles)
        covar = cov(Matrix(log_particles), AnalyticWeights(weights), 2)
        eig_vals, eig_vecs = eigen(covar)
        sqrt_eig_vals = @. sqrt(max(eig_vals, 1e-8))
        sqrt_covar = eig_vecs * Diagonal(sqrt_eig_vals)
        log_particles .+= constant .* sqrt_covar * randn(size(log_particles))
        return exp.(log_particles)
    end

    dynamics = StochasticDynamics(
        xdim, udim,
        (x, u) -> drift_fn(true_params, x, u),
        diffusion_fn,
        step_size,
    )

    ibis_dynamics = IBISDynamics(
        xdim, udim,
        drift_fn!,
        diffusion_fn,
        step_size,
    )

    ctl_shift, ctl_scale = 0.0, 5.0

    function ctl_feature_fn(
        z::AbstractVector{Float64}
    )::Vector{Float64}
        return z[1:4]
    end

end