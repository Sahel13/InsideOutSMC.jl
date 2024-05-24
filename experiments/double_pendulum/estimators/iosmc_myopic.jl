using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsBase

import Flux

include("../environment.jl")

using .DoublePendulumEnvironment: xdim, udim
using .DoublePendulumEnvironment: init_state
using .DoublePendulumEnvironment: ibis_dynamics
using .DoublePendulumEnvironment: param_prior
using .DoublePendulumEnvironment: param_proposal
using .DoublePendulumEnvironment: ctl_shift, ctl_scale

using Printf: @printf


prior_policy = UniformStochasticPolicy(ctl_scale)
policy_loop = IBISClosedLoop(
    ibis_dynamics, prior_policy
)

nb_runs = 25

nb_steps = 50
nb_trajectories = 16
nb_ibis_particles = 1024

nb_ibis_moves = 3

nb_policy_particles = 256
action_penalty = 0.0
slew_rate_penalty = 0.1
tempering = 0.25

policy_scratch = Array{Float64}(
    undef, xdim, nb_ibis_particles, nb_policy_particles
)

myopic_policy = MyopicAdaptiveIBISPolicy(
    nb_policy_particles,
    policy_loop,
    action_penalty,
    slew_rate_penalty,
    tempering,
    policy_scratch
)

adaptive_loop = IBISAdaptiveLoop(
    ibis_dynamics, myopic_policy
)

our_estimator = zeros(nb_runs)

for k in 1:nb_runs
    Random.seed!(k)
    state_struct, _ = myopic_smc_with_ibis_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        nb_ibis_particles,
        init_state,
        adaptive_loop,
        param_prior,
        param_proposal,
        nb_ibis_moves,
    )
    our_estimator[k] = mean(state_struct.cumulative_return)
    @printf("iter: %i, EIG Estimate: %0.4f\n", k, our_estimator[k])
end

@printf("EIG Estimate: %0.4f ± %0.4f\n", mean(our_estimator), std(our_estimator))