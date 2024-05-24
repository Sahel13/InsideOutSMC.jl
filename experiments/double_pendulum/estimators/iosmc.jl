using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using StatsBase

import Flux

using Printf: @printf

include("../environment.jl")

using .DoublePendulumEnvironment: xdim, udim
using .DoublePendulumEnvironment: step_size
using .DoublePendulumEnvironment: init_state
using .DoublePendulumEnvironment: ibis_dynamics
using .DoublePendulumEnvironment: param_prior
using .DoublePendulumEnvironment: param_proposal
using .DoublePendulumEnvironment: ctl_scale
using .DoublePendulumEnvironment: ctl_feature_fn

using JLD2


policy = UniformStochasticPolicy(ctl_scale)
# policy = PRBSStochasticPolicy(ctl_scale)
# policy = load("./experiments/double_pendulum/data/double_pendulum_ibis_csmc_ctl.jld2")["ctl"]

closedloop = IBISClosedLoop(
    ibis_dynamics, policy
)

nb_runs = 25

nb_steps = 50
nb_trajectories = 16
nb_particles = 1024

nb_ibis_moves = 3

action_penalty = 0.0
slew_rate_penalty = 0.0
tempering = 0.0

our_estimator = zeros(nb_runs)

for k in 1:nb_runs
    Flux.reset!(closedloop.ctl)
    state_struct, _ = smc_with_ibis_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        nb_particles,
        init_state,
        closedloop,
        param_prior,
        param_proposal,
        nb_ibis_moves,
        action_penalty,
        slew_rate_penalty,
        tempering,
    )
    our_estimator[k] = mean(state_struct.cumulative_return)
    @printf("iter: %i, EIG Estimate: %0.4f\n", k, our_estimator[k])
end

@printf("EIG Estimate: %0.4f ± %0.4f\n", mean(our_estimator), std(our_estimator))