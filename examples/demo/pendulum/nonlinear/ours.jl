using Revise

using Random
using Distributions
using LinearAlgebra

using InsideOutSMC

using Printf: @printf

include("environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: step_size
using .PendulumEnvironment: init_state
using .PendulumEnvironment: ibis_dynamics
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: ctl_scale


function param_proposal(
    particles::AbstractMatrix{Float64},
    prop_stddev::Float64=0.1
)::Matrix{Float64}
    return particles .+ prop_stddev .* randn(size(particles))
end


rnd_policy = UniformStochasticPolicy([ctl_scale])
rnd_closedloop = IBISClosedLoop(
    ibis_dynamics, rnd_policy
)

nb_runs = 5
nb_steps = 50
nb_trajectories = 16
nb_particles = 4096

nb_ibis_moves = 3

action_penalty = 0.0
slew_rate_penalty = 0.0
tempering = 0.0

our_estimator = zeros(nb_runs)

for k in 1:nb_runs
    state_struct, _ = smc_with_ibis_marginal_dynamics(
        nb_steps,
        nb_trajectories,
        nb_particles,
        init_state,
        rnd_closedloop,
        param_prior,
        param_proposal,
        nb_ibis_moves,
        action_penalty,
        slew_rate_penalty,
        tempering,
    )
    our_estimator[k] = state_struct.cumulative_return' * state_struct.weights
    @printf("iter: %i, Ours: %0.4f\n", k, our_estimator[k])
end

@printf("Ours: %0.4f ± %0.4f\n", mean(our_estimator), std(our_estimator))