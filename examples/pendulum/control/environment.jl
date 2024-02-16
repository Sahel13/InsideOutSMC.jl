module PendulumEnvironment

    using LinearAlgebra: Diagonal
    using InsideOutSMC: StochasticDynamics


    function reward_fn(
        z::AbstractVector{Float64}
    )::Float64
        q, dq = z[begin:2]
        u = z[3:end]

        function wrap_angle(q)
            return mod(q, (2.0 * pi))
        end

        Q = Diagonal([1e1, 1e-1])
        R = Diagonal([1e-3])
        g0 = Array{Float64}([pi, 0.0])

        xw = vcat(wrap_angle(q), dq)
        cost = (xw - g0)' * Q * (xw - g0)
        cost += u' * R * u
        return -0.5 * cost
    end


    xdim = 2
    udim = 1

    step_size = 0.05
    init_state = [0.0, 0.0, 0.0]

    true_params = [1.0, 1.0]

    function drift_fn(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
    )::Vector{Float64}
        m, l = p
        g, d = 9.81, 1e-3

        q, dq = x
        ddq = - 3.0 * g / (2.0 * l) * sin(q)
        ddq += (only(u) - d * dq) * 3.0 / (m * l^2)
        return [dq, ddq]
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
        step_size
    )

    ctl_shift, ctl_scale = 0.0, 2.5
end