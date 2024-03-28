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
udim = 1
pdim = 3

step_size = 0.05
init_state = [0.0, pi/3.0, 0.0, 0.0, 0.0]

true_params = [1.0, 1.0, 1.0]


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


rnd_policy = UniformStochasticPolicy([5.0])

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
#     prop_stddev::AbstractMatrix{Float64}=Diagonal([0.01, 0.01, 0.01])
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
        [0.0, 0.0, 0.0],
        Diagonal([0.1, 0.1, 0.1].^2)
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