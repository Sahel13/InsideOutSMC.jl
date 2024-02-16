module PendulumEnvironment

    using LinearAlgebra: I
    using LinearAlgebra: Diagonal
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
        [10.0, 0.0, 5.0],
        1.0 * Matrix{Float64}(I, 3, 3)
    )

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

    ctl_shift, ctl_scale = 0.0, 2.5
end