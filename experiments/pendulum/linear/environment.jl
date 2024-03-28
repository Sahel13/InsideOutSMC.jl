module PendulumEnvironment

    using Random
    using Distributions
    using LinearAlgebra
    using StatsBase

    using InsideOutSMC: Gaussian
    using InsideOutSMC: StochasticDynamics
    using InsideOutSMC: RaoBlackwellDynamics
    using InsideOutSMC: IBISDynamics

    function feature_fn(
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
    )::Vector{Float64}
        q, dq = x
        return vcat(-sin(q), -dq, u)
    end

    function drift_fn(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
    )
        _, dq = x
        ddq = feature_fn(x, u)' * p
        return [dq, ddq]
    end

    function drift_fn!(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
        xn::AbstractVector{Float64},
    )
        xn[1] = x[2]
        xn[2] = feature_fn(x, u)' * p
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

    param_prior = Gaussian(
        [14.7, 0.0, 3.0],
        Diagonal([0.1, 0.01, 0.1])
    )

    function param_proposal(
        particles::AbstractMatrix{Float64},
        weights::AbstractVector{Float64},
        constant::Float64=1.0
    )::Matrix{Float64}
        covar = cov(Matrix(particles), AnalyticWeights(weights), 2)
        eig_vals, eig_vecs = eigen(covar)
        sqrt_eig_vals = @. sqrt(max(eig_vals, 1e-8))
        sqrt_covar = eig_vecs * Diagonal(sqrt_eig_vals)
        return particles + constant .* sqrt_covar * randn(size(particles))
    end

    ibis_dynamics = IBISDynamics(
        xdim, udim,
        drift_fn!,
        diffusion_fn,
        step_size,
    )

    rb_dynamics = RaoBlackwellDynamics(
        xdim, udim,
        feature_fn,
        diffusion_fn,
        step_size
    )

    ctl_shift, ctl_scale = 0.0, 1.0

    function ctl_feature_fn(
        z::AbstractVector{Float64}
    )::Vector{Float64}
        return z[1:2]
    end

end