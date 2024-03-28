module DoublePendulumEnvironment

    using Distributions
    using LinearAlgebra: Diagonal
    using InsideOutSMC: StochasticDynamics
    using InsideOutSMC: IBISDynamics

    function drift_fn(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
    )
        m1, m2, l1, l2 = p
        k1, k2 = 1e-1, 1e-1

        g = 9.81

        q1, q2, dq1, dq2 = x
        u1, u2 = u

        s1, c1 = sin(q1), cos(q1)
        s2, c2 = sin(q2), cos(q2)
        s12 = sin(q1 + q2)

        # inertia
        M = [
                (m1 + m2) * l1^2 + m2 * l2^2 + 2.0 * m2 * l1 * l2 * c2      m2 * l2^2 + m2 * l1 * l2 * c2;
                m2 * l2^2 + m2 * l1 * l2 * c2                               m2 * l2^2
        ]

        # Corliolis
        C = [
                0.0                                                         -m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2;
                0.5 * m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2                 -0.5 * m2 * l1 * l2 * dq1 * s2
        ]

        # gravity
        tau = -g .* [
                (m1 + m2) * l1 * s1 + m2 * l2 * s12,                        m2 * l2 * s12
        ]

        B = Diagonal([1.0, 1.0])

        u1 = u1 - k1 * dq1
        u2 = u2 - k2 * dq2
        u = [u1, u2]

        dq = [dq1, dq2]
        ddq = inv(M) * (tau + B * u - C * dq)
        return vcat(dq, ddq)
    end

    function drift_fn!(
        p::AbstractVector{Float64},
        x::AbstractVector{Float64},
        u::AbstractVector{Float64},
        xn::AbstractVector{Float64},
    )
        m1, m2, l1, l2 = p
        k1, k2 = 1e-1, 1e-1

        g = 9.81

        q1, q2, dq1, dq2 = x
        u1, u2 = u

        s1, c1 = sin(q1), cos(q1)
        s2, c2 = sin(q2), cos(q2)
        s12 = sin(q1 + q2)

        # inertia
        M = [
                (m1 + m2) * l1^2 + m2 * l2^2 + 2.0 * m2 * l1 * l2 * c2      m2 * l2^2 + m2 * l1 * l2 * c2;
                m2 * l2^2 + m2 * l1 * l2 * c2                               m2 * l2^2
        ]

        # Corliolis
        C = [
                0.0                                                         -m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2;
                0.5 * m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2                 -0.5 * m2 * l1 * l2 * dq1 * s2
        ]

        # gravity
        tau = -g .* [
                (m1 + m2) * l1 * s1 + m2 * l2 * s12,                        m2 * l2 * s12
        ]

        B = Diagonal([1.0, 1.0])

        u1 = u1 - k1 * dq1
        u2 = u2 - k2 * dq2
        u = [u1, u2]

        dq = [dq1, dq2]
        ddq = inv(M) * (tau + B * u - C * dq)

        xn[1] = dq[1]
        xn[2] = dq[2]
        xn[3] = ddq[1]
        xn[4] = ddq[2]
    end


    function diffusion_fn(
        args::AbstractVector{Float64}...,
    )::Vector{Float64}
        return [0.0, 0.0, 1e-1, 1e-1]
    end

    xdim = 4
    udim = 2

    step_size = 0.05
    init_state = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    true_params = [1.0, 1.0, 1.0, 1.0]

    param_prior = MvLogNormal(
        MvNormal(
            [0.0, 0.0, 0.0, 0.0],
            Diagonal([0.1, 0.1, 0.1, 0.1].^2)
        )
    )

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

    ctl_shift, ctl_scale = [0.0, 0.0], [4.0, 2.0]
end