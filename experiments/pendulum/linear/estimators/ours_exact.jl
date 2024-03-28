using InsideOutSMC

using Random
using Distributions
using LinearAlgebra

import Flux

using Printf: @printf

include("../environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: step_size
using .PendulumEnvironment: init_state
using .PendulumEnvironment: rb_dynamics
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: ctl_scale
using .PendulumEnvironment: ctl_feature_fn

using JLD2


policy = UniformStochasticPolicy([ctl_scale])
# policy = PRBSStochasticPolicy([ctl_scale])
# policy = load("./experiments/pendulum/linear/data/linear_pendulum_rb_csmc_ctl.jld2")["ctl"]
# policy = load("./experiments/pendulum/linear/data/linear_pendulum_ibis_csmc_ctl.jld2")["ctl"]

closedloop = RaoBlackwellClosedLoop(
    rb_dynamics, policy
)

nb_runs = 25

nb_steps = 50
nb_trajectories = 16

action_penalty = 0.0
slew_rate_penalty = 0.0
tempering = 0.0

our_estimator = zeros(nb_runs)

for k in 1:nb_runs
    Flux.reset!(closedloop.ctl)
    state_struct, _ = smc_with_rao_blackwell_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        init_state,
        closedloop,
        param_prior,
        action_penalty,
        slew_rate_penalty,
        tempering,
    )
    our_estimator[k] = mean(state_struct.cumulative_return)
    @printf("iter: %i, Ours: %0.4f\n", k, our_estimator[k])
end

@printf("Ours: %0.4f ± %0.4f\n", mean(our_estimator), std(our_estimator))