using Revise

using InsideOutSMC

using Random
using Distributions
using LinearAlgebra

using Printf: @printf

include("environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: step_size
using .PendulumEnvironment: init_state
using .PendulumEnvironment: rb_dynamics
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: ctl_scale

Random.seed!(1)

rnd_policy = UniformStochasticPolicy([ctl_scale])
rnd_closedloop = RaoBlackwellClosedLoop(
    rb_dynamics, rnd_policy
)

nb_runs = 5
nb_steps = 50
nb_trajectories = 16

action_penalty = 0.0
slew_rate_penalty = 0.0
tempering = 0.0

our_estimator = zeros(nb_runs)

for k in 1:nb_runs
    state_struct, param_struct = smc_with_rao_blackwell_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        init_state,
        rnd_closedloop,
        param_prior,
        action_penalty,
        slew_rate_penalty,
        tempering,
    )
    our_estimator[k] = state_struct.cumulative_return' * state_struct.weights
    @printf("iter: %i, Ours: %0.4f\n", k, our_estimator[k])
end

@printf("Ours: %0.4f ± %0.4f\n", mean(our_estimator), std(our_estimator))