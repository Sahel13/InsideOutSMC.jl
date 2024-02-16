module PendulumEnvironment

    using Distributions
    using LinearAlgebra: Diagonal
    using InsideOutSMC: StochasticDynamics
    using InsideOutSMC: RaoBlackwellDynamics
    using InsideOutSMC: IBISDynamics

    function drift_fn!(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
        xn::AbstractVector{Float64},
    )
        m, l = p
        g, d = 9.81, 1e-3

        xn[1] = x[2]
        xn[2] = (
            - 3.0 * g / (2.0 * l) * sin(x[1])
            + (only(u) - d * x[2]) * 3.0 / (m * l^2)
        )
    end

    function diffusion_fn(
        args::AbstractVector{Float64}...,
    )::Vector{Float64}
        return [0.0, 1e-1]
    end

    xdim = 2
    udim = 1

    step_size = 0.05
    init_state = [0.0, 0.0, 0.0]

    param_prior = MvLogNormal(
        MvNormal(
            [0.0, 0.0],
            Diagonal([0.1, 0.1].^2)
        )
    )

    ibis_dynamics = IBISDynamics(
        xdim, udim,
        drift_fn!,
        diffusion_fn,
        step_size,
    )

    ctl_shift, ctl_scale = 0.0, 2.5
end