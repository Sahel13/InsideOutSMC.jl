using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsBase

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
nb_outer_samples = 16
nb_inner_samples = 1_000_000

nb_policy_particles = 256
action_penalty = 0.0
slew_rate_penalty = 0.1
tempering = 1.0

myopic_policy = MyopicAdaptiveRaoBlackwellPolicy(
    nb_policy_particles,
    policy_loop,
    action_penalty,
    slew_rate_penalty,
    tempering,
)

adaptive_loop = RaoBlackwellAdaptiveLoop(
    rb_dynamics, myopic_policy
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
    )
    their_estimator[k] = spce
    @printf("iter: %i, Theirs: %0.4f\n", k, their_estimator[k])
end

@printf("Theirs: %0.4f ± %0.4f\n", mean(their_estimator), std(their_estimator))