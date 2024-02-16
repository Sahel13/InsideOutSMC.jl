module PendulumEnvironment

    using LinearAlgebra: Diagonal
    using InsideOutSMC: StochasticDynamics
    using InsideOutSMC: RaoBlackwellDynamics
    using InsideOutSMC: IBISDynamics

    xdim = 2
    udim = 1

    step_size = 0.05
    init_state = [0.0, 0.0, 0.0]

    m, l = 1.0, 1.0
    g, d = 9.81, 1e-3

    true_params = [m, l]

    rb_params = [
        3.0 * g / (2.0 * l),
        3.0 * d / (m * l^2),
        3.0 / (m * l^2),
    ]

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
        u::AbstractVector{Float64}
    )::Vector{Float64}
        m, l = p
        g, d = 9.81, 1e-3

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
        g, d = 9.81, 1e-3

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

    dynamics = StochasticDynamics(
        xdim, udim,
        (x, u) -> drift_fn(true_params, x, u),
        diffusion_fn,
        step_size,
    )

    rb_dynamics = RaoBlackwellDynamics(
        xdim, udim,
        feature_fn,
        diffusion_fn,
        step_size
    )

    ibis_dynamics = IBISDynamics(
        xdim, udim,
        drift_fn!,
        diffusion_fn,
        step_size,
    )

    ctl_shift, ctl_scale = 0.0, 2.5
end