using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics

import Flux

include("../environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: init_state
using .PendulumEnvironment: rb_dynamics
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: ctl_shift, ctl_scale

using Printf: @printf


prior_policy = UniformStochasticPolicy([ctl_scale])
policy_loop = RaoBlackwellClosedLoop(
    rb_dynamics, prior_policy
)

nb_runs = 25

nb_steps = 50
nb_trajectories = 16

nb_policy_particles = 256
action_penalty = 0.0
slew_rate_penalty = 0.1
tempering = 1.0

myopic_policy = MyopicAdaptiveRaoBlackwellPolicy(
    nb_policy_particles,
    policy_loop,
    action_penalty,
    slew_rate_penalty,
    tempering
)

adaptive_loop = RaoBlackwellAdaptiveLoop(
    rb_dynamics, myopic_policy
)

our_estimator = zeros(nb_runs)

for k in 1:nb_runs
    Random.seed!(k)
    state_struct, _ = myopic_smc_with_rao_blackwell_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        init_state,
        adaptive_loop,
        param_prior,
    )
    our_estimator[k] = mean(state_struct.cumulative_return)
    @printf("iter: %i, EIG Estimate: %0.4f\n", k, our_estimator[k])
end

@printf("EIG Estimate: %0.4f ± %0.4f\n", mean(our_estimator), std(our_estimator))