using Revise

using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsBase

import Zygote
import Flux
import Bijectors

include("environment.jl")

using .CartpoleEnvironment: xdim, udim
using .CartpoleEnvironment: init_state
using .CartpoleEnvironment: dynamics
using .CartpoleEnvironment: ibis_dynamics
using .CartpoleEnvironment: param_prior
using .CartpoleEnvironment: ctl_shift, ctl_scale

using Plots

# using ProfileView  # ProfileView.@profview
# using ProfileCanvas  # ProfileCanvas.@profview  # ProfileCanvas.@profview_allocs
# using BenchmarkTools


# function param_proposal(
#     particles::AbstractMatrix{Float64},
#     weights::AbstractVector{Float64},
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


function ctl_feature_fn(
    z::AbstractVector{Float64}
)::Vector{Float64}
    return z[1:4]
end


input_dim = 4
output_dim = 1
recur_size = 64
dense_size = 256

ctl_encoder_fn = Flux.f64(
    Flux.Chain(
        Flux.Dense(input_dim, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, recur_size),
        # Flux.LSTM(recur_size, recur_size),
        # Flux.LSTM(recur_size, recur_size),
        Flux.GRU(recur_size, recur_size),
        Flux.GRU(recur_size, recur_size),
    ),
)

ctl_mean_fn = Flux.f64(
    Flux.Chain(
        Flux.Dense(recur_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, output_dim),
    ),
)

ctl_log_std = @. log(sqrt([1.0]))

ctl_bijector = (
    Bijectors.Shift(ctl_shift)
    ∘ Bijectors.Scale(ctl_scale)
    ∘ Tanh()
)

learner_policy = StatefulHomoschedasticPolicy(
    ctl_feature_fn,
    ctl_encoder_fn,
    ctl_mean_fn,
    ctl_log_std,
    ctl_bijector,
)

evaluator_policy = StatefulHomoschedasticPolicy(
    ctl_feature_fn,
    ctl_encoder_fn,
    ctl_mean_fn,
    [-20.0],
    ctl_bijector,
)

learner_loop = IBISClosedLoop(
    ibis_dynamics, learner_policy
)

evaluator_loop = IBISClosedLoop(
    ibis_dynamics, evaluator_policy
)

action_penalty = 0.0
slew_rate_penalty = 0.1
tempering = 0.25

nb_steps = 50
nb_trajectories = 256
nb_particles = 1024

nb_ibis_moves = 3
nb_csmc_moves = 1

nb_iter = 15
opt_state = Flux.setup(Flux.Optimise.Adam(5e-4), learner_loop)
batch_size = 64

Flux.reset!(learner_loop.ctl)
state_struct, param_struct = smc_with_ibis_marginal_dynamics(
    nb_steps,
    nb_trajectories,
    nb_particles,
    init_state,
    learner_loop,
    param_prior,
    param_proposal,
    nb_ibis_moves,
    action_penalty,
    slew_rate_penalty,
    tempering,
)
idx = rand(Categorical(state_struct.weights))
reference = IBISReference(
    state_struct.trajectories[:, :, idx],
    param_struct.particles[:, :, :, idx],
    param_struct.weights[:, :, idx],
    param_struct.log_weights[:, :, idx],
    param_struct.log_likelihoods[:, :, idx]
)

learner_loop, _ = markovian_score_climbing_with_ibis_marginal_dynamics(
    nb_iter,
    opt_state,
    batch_size,
    nb_steps,
    nb_trajectories,
    nb_particles,
    init_state,
    learner_loop,
    evaluator_loop,
    param_prior,
    param_proposal,
    nb_ibis_moves,
    action_penalty,
    slew_rate_penalty,
    tempering,
    reference,
    nb_csmc_moves,
    true
)

trajectory = Array{Float64}(undef, xdim+udim, nb_steps + 1)
trajectory[:, 1] = init_state

Flux.reset!(learner_loop.ctl)
for t = 1:nb_steps
    state = trajectory[1:xdim, t]
    action = policy_mean(learner_loop.ctl, trajectory[:, t])
    next_state = dynamics_sample(dynamics, state, action)
    trajectory[:, t+1] = vcat(next_state, action)
end
plot(trajectory[1:end, :]')


using Printf: @printf

Flux.reset!(evaluator_loop.ctl)
state_struct, _ = smc_with_ibis_marginal_dynamics(
    nb_steps,
    16,
    16_000,
    init_state,
    evaluator_loop,
    param_prior,
    param_proposal,
    nb_ibis_moves,
    action_penalty,
    slew_rate_penalty,
    0.0,
)

@printf(
    "return: %0.4f\n",
    state_struct.cumulative_return' * state_struct.weights,
)
