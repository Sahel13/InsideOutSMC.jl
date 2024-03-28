module PendulumEnvironment

    using Random
    using Distributions
    using LinearAlgebra
    using StatsBase

    using InsideOutSMC: StochasticDynamics
    using InsideOutSMC: IBISDynamics

    function drift_fn(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64}
    )::Vector{Float64}
        m, l = p
        g, d = 9.81, 1e-1

        q, dq = x
        ddq = - 3.0 * g / (2.0 * l) * sin(q)
        ddq += (only(u) - d * dq) * 3.0 / (m * l^2)
        return [dq, ddq]
    end

    function drift_fn!(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
        xn::AbstractVector{Float64},
    )
        m, l = p
        g, d = 9.81, 1e-1

        xn[1] = x[2]
        xn[2] = (
            - 3.0 * g / (2.0 * l) * sin(x[1])
            + (only(u) - d * x[2]) * 3.0 / (m * l^2)
        )
    end

    function diffusion_fn(
        args::AbstractVector{Float64}...
    )::Vector{Float64}
        return [0.0, 1e-1]
    end

    xdim = 2
    udim = 1

    step_size = 0.05
    init_state = [0.0, 0.0, 0.0]

    true_params = [1.0, 1.0]  # m, l

    param_prior = MvLogNormal(
        MvNormal(
            [0.0, 0.0],
            Diagonal([0.1, 0.1].^2)
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

    ctl_shift, ctl_scale = 0.0, 1.0

    function ctl_feature_fn(
        z::AbstractVector{Float64}
    )::Vector{Float64}
        return z[1:2]
    end

end