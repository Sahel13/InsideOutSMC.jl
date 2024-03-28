using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using StatsBase

import Flux

using Printf: @printf

include("../environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: step_size
using .PendulumEnvironment: init_state
using .PendulumEnvironment: ibis_dynamics
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: param_proposal
using .PendulumEnvironment: ctl_scale
using .PendulumEnvironment: ctl_feature_fn

using JLD2


Random.seed!(1)

policy = UniformStochasticPolicy([ctl_scale])
# policy = PRBSStochasticPolicy([ctl_scale])
# policy = load("./experiments/pendulum/nonlinear/data/nonlinear_pendulum_ibis_csmc_ctl.jld2")["ctl"]

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
    @printf("iter: %i, Ours: %0.4f\n", k, our_estimator[k])
end

@printf("Ours: %0.4f ± %0.4f\n", mean(our_estimator), std(our_estimator))
