using Base.Threads: nthreads, @threads, @spawn
using Base.Iterators: partition

using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using StatsBase

using Plots

Random.seed!(321)


xdim = 4
udim = 2
pdim = 4

step_size = 0.05
init_state = [pi/3.0, pi/3.0, 0.0, 0.0, 0.0, 0.0]

true_params = [1.0, 1.0, 1.0, 1.0]


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


rnd_policy = UniformStochasticPolicy([4.0, 2.0])

ibis_dynamics = IBISDynamics(
    xdim, udim,
    drift_fn!,
    diffusion_fn,
    step_size
)

ibis_loop = IBISClosedLoop(
    ibis_dynamics,
    rnd_policy
)

nb_steps = 50
nb_trajectories = 100

trajectories = Array{Float64}(undef, xdim+udim, nb_steps+1, nb_trajectories)
trajectories[:, 1, :] .= init_state

@inbounds @views for t = 1:nb_steps
    ibis_conditional_closedloop_sample!(
        ibis_loop,
        repeat(true_params, 1, nb_trajectories),
        view(trajectories, :, t, :),
        view(trajectories, :, t+1, :)
    )
end
# plot(trajectories[:, :, 1]')


# function param_proposal(
#     particles::AbstractMatrix{Float64},
#     prop_stddev::AbstractMatrix{Float64}=Diagonal([0.01, 0.01])
# )::Matrix{Float64}
#     log_particles = log.(particles)
#     log_particles .+= prop_stddev * randn(size(log_particles))
#     return exp.(log_particles)
# end


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


nb_particles = 128
param_prior = MvLogNormal(
    MvNormal(
        [0.0, 0.0, 0.0, 0.0],
        Diagonal([0.1, 0.1, 0.1, 0.1].^2)
    )
)

nb_moves = 3

xdim = ibis_loop.dyn.xdim
udim = ibis_loop.dyn.udim

scratch = Array{Float64,3}(undef, xdim, nb_particles, nb_trajectories)
state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
param_struct = IBISParamStruct(param_prior, nb_steps, nb_particles, nb_trajectories, scratch)
state_struct.trajectories .= reshape(trajectories, xdim+udim, nb_steps + 1, nb_trajectories)

chunk_size = 10
ranges = partition(1:nb_trajectories, chunk_size)

trajectory_chunks = [view(trajectories, :, :, range) for range in ranges]
param_struct_views = [view_struct(param_struct, range) for range in ranges]

@time begin
    tasks = map(zip(trajectory_chunks, param_struct_views)) do (traj_chunk, struct_view)
        @spawn begin
            batch_ibis!(
                traj_chunk,
                ibis_loop.dyn,
                param_prior,
                param_proposal,
                nb_moves,
                struct_view
            );
        end
    end
    fetch.(tasks);
end

num_tests_passed = 0
for n = 1:nb_trajectories
    params = @view param_struct.particles[:, end, :, n]
    param_weights = @view param_struct.weights[end, :, n]
    ibis_mean = sum(repeat(param_weights, 1, pdim)' .* params, dims=2)
    if norm(ibis_mean - true_params)^2 < 0.1
        global num_tests_passed += 1
    end
end

println("number of tests passed: ", num_tests_passed)