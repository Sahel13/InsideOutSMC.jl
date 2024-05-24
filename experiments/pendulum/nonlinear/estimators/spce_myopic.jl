using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsBase

include("../environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: init_state
using .PendulumEnvironment: ibis_dynamics
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: param_proposal
using .PendulumEnvironment: ctl_shift, ctl_scale

using Printf: @printf


prior_policy = UniformStochasticPolicy([ctl_scale])
policy_loop = IBISClosedLoop(
    ibis_dynamics, prior_policy
)

nb_runs = 25

nb_steps = 50
nb_outer_samples = 16
nb_inner_samples = 1_000_000

nb_ibis_particles = 1024
nb_ibis_moves = 3

nb_policy_particles = 256
action_penalty = 0.0
slew_rate_penalty = 0.2
tempering = 1.0

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

their_estimator = zeros(nb_runs)

for k in 1:nb_runs
    Random.seed!(k)
    spce = compute_sPCE_for_myopic_adaptive_policy(
        adaptive_loop,
        param_prior,
        init_state,
        nb_steps,
        nb_outer_samples,
        nb_inner_samples,
        nb_ibis_particles,
        param_proposal,
        nb_ibis_moves,
    )
    their_estimator[k] = spce
    @printf("iter: %i, sPCE: %0.4f\n", k, their_estimator[k])
end

@printf("sPCE: %0.4f ± %0.4f\n", mean(their_estimator), std(their_estimator))