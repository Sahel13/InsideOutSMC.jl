using Base.Threads: nthreads, @threads, @spawn
using Base.Iterators: partition

using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using StatsBase

using Plots

Random.seed!(321)


xdim = 2
udim = 1
pdim = 3

step_size = 0.05
init_state = [pi/2.0, 0.0, 0.0]

m, l = 1.0, 1.0
g, d = 9.81, 1e-3

true_params = [
    3.0 * g / (2.0 * l),
    3.0 * d / (m * l^2),
    3.0 / (m * l^2),
]

param_prior = Gaussian(
    [10.0, 0.0, 5.0],
    1.0 * Matrix{Float64}(I, pdim, pdim)
)


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


rnd_policy = UniformStochasticPolicy([2.5])

ibis_dynamics = IBISDynamics(
    xdim, udim,
    drift_fn!,
    diffusion_fn,
    step_size
)

rb_dynamics = RaoBlackwellDynamics(
    xdim, udim,
    feature_fn,
    diffusion_fn,
    step_size,
)

ibis_loop = IBISClosedLoop(
    ibis_dynamics,
    rnd_policy
)

nb_steps = 50
nb_trajectories = 100

trajectories = Array{Float64}(undef, xdim+udim, nb_steps+1, nb_trajectories)
trajectories[:, 1, :] .= init_state

param_posteriors = Array{Gaussian}(undef, nb_steps+1, nb_trajectories)
for t in 1:nb_steps + 1
    for n in 1:nb_trajectories
        param_posteriors[t, n] = deepcopy(param_prior)
    end
end

for t = 1:nb_steps
    global param_posterior

    ibis_conditional_closedloop_sample!(
        ibis_loop,
        repeat(true_params, 1, nb_trajectories),
        view(trajectories, :, t, :),
        view(trajectories, :, t+1, :)
    )

    for n in 1:nb_trajectories
        param_posteriors[t+1, n] = rao_blackwell_dynamics_update(
            rb_dynamics,
            param_posteriors[t, n],
            trajectories[begin:xdim, t, n],
            trajectories[xdim+1:end, t+1, n],
            trajectories[begin:xdim, t+1, n]
        )
    end
end
# plot(trajectories[:, :, 1]')


function param_proposal(
    particles::AbstractMatrix{Float64},
    weights::AbstractVector{Float64},
    prop_stddev::Float64=0.1
)::Matrix{Float64}
    return particles .+ prop_stddev .* randn(size(particles))
end


# function param_proposal(
#     particles::AbstractMatrix{Float64},
#     weights::AbstractVector{Float64},
#     constant::Float64=10.0
# )::Matrix{Float64}
#     covar = cov(Matrix(particles), AnalyticWeights(weights), 2)
#     eig_vals, eig_vecs = eigen(covar)
#     sqrt_eig_vals = @. sqrt(max(eig_vals, 1e-8))
#     sqrt_covar = eig_vecs * Diagonal(sqrt_eig_vals)
#     return particles + constant .* sqrt_covar * randn(size(particles))
# end


nb_particles = 512
param_prior = MvNormal(
    param_prior.mean,
    param_prior.covar
)

nb_moves = 1

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

# @time batch_ibis!(
#     trajectories,
#     ibis_loop.dyn,
#     param_prior,
#     param_proposal,
#     nb_moves,
#     param_struct
# );

num_tests_passed = 0
for n = 1:nb_trajectories
    params = @view param_struct.particles[:, end, :, n]
    param_weights = @view param_struct.weights[end, :, n]
    ibis_mean = sum(repeat(param_weights, 1, pdim)' .* params, dims=2)
    rb_mean = param_posteriors[end, n].mean
    if norm(ibis_mean - rb_mean)^2 < 0.1
        global num_tests_passed += 1
    end
end

println("number of tests passed: ", num_tests_passed)