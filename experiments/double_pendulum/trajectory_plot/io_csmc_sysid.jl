using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsBase

import Zygote
import Flux
import Bijectors

include("../environment.jl")

using .DoublePendulumEnvironment: xdim, udim
using .DoublePendulumEnvironment: init_state
using .DoublePendulumEnvironment: dynamics
using .DoublePendulumEnvironment: ibis_dynamics
using .DoublePendulumEnvironment: param_prior
using .DoublePendulumEnvironment: param_proposal
using .DoublePendulumEnvironment: ctl_shift, ctl_scale
using .DoublePendulumEnvironment: ctl_feature_fn

using JLD2


input_dim = 4
output_dim = 2
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

ctl_log_std = @. log(sqrt([1.0, 1.0]))

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
    [-20.0, -20.0],
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
nb_particles = 128

nb_ibis_moves = 3
nb_csmc_moves = 1

nb_iter = 25
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

jldsave("./experiments/double_pendulum/data/double_pendulum_ibis_csmc_ctl.jld2"; evaluator_loop.ctl)
