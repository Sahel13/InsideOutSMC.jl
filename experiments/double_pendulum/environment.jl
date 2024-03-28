module DoublePendulumEnvironment

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

    function drift_fn_sans_matrices(
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

        # Mass
        I_1 = m1 * l1^2
        I_2 = m2 * l2^2

        M_1 = I_1 + I_2 + m2 * l1^2 + 2.0 * m2 * l1 * l2 * c2
        M_2 = I_2 + m2 * l1 * l2 * c2
        M_3 = M_2
        M_4 = I_2

        # Coriolis
        C_1 = 0.0
        C_2 = -m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2
        C_3 = 0.5 * m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2
        C_4 = -0.5 * m2 * l1 * l2 * dq1 * s2

        # Gravity
        tau_1 = -g * ((m1 + m2) * l1 * s1 + m2 * l2 * s12)
        tau_2 = -g * m2 * l2 * s12

        u1 = u1 - k1 * dq1
        u2 = u2 - k2 * dq2

        ddq2 = tau_2 + u2 - C_3 * dq1 - C_4 * dq2
        ddq2 -= M_3 / M_1 * (tau_1 + u1 - C_1 * dq1 - C_2 * dq2)
        ddq2 /= M_4 - M_3 * M_2 / M_1

        ddq1 = tau_1 + u1 - C_1 * dq1 - C_2 * dq2 - M_2 * ddq2
        ddq1 /= M_1
        return vcat(dq1, dq2, ddq1, ddq2)
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


    function drift_fn_sans_matrices!(
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

        # Mass
        I_1 = m1 * l1^2
        I_2 = m2 * l2^2

        M_1 = I_1 + I_2 + m2 * l1^2 + 2.0 * m2 * l1 * l2 * c2
        M_2 = I_2 + m2 * l1 * l2 * c2
        M_3 = M_2
        M_4 = I_2

        # Coriolis
        C_1 = 0.0
        C_2 = -m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2
        C_3 = 0.5 * m2 * l1 * l2 * (2.0 * dq1 + dq2) * s2
        C_4 = -0.5 * m2 * l1 * l2 * dq1 * s2

        # Gravity
        tau_1 = -g * ((m1 + m2) * l1 * s1 + m2 * l2 * s12)
        tau_2 = -g * m2 * l2 * s12

        u1 = u1 - k1 * dq1
        u2 = u2 - k2 * dq2

        xn[1] = dq1
        xn[2] = dq2

        ddq2 = tau_2 + u2 - C_3 * dq1 - C_4 * dq2
        ddq2 -= M_3 / M_1 * (tau_1 + u1 - C_1 * dq1 - C_2 * dq2)
        ddq2 /= M_4 - M_3 * M_2 / M_1

        ddq1 = tau_1 + u1 - C_1 * dq1 - C_2 * dq2 - M_2 * ddq2
        ddq1 /= M_1

        xn[3] = ddq1
        xn[4] = ddq2
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
        (x, u) -> drift_fn_sans_matrices(true_params, x, u),
        diffusion_fn,
        step_size,
    )

    ibis_dynamics = IBISDynamics(
        xdim, udim,
        drift_fn_sans_matrices!,
        diffusion_fn,
        step_size,
    )

    ctl_shift, ctl_scale = [0.0, 0.0], [4.0, 2.0]

    function ctl_feature_fn(
        z::AbstractVector{Float64}
    )::Vector{Float64}
        return z[1:4]
    end

end