using InsideOutSMC

using Random
using Distributions
using Statistics
using LinearAlgebra
using LogExpFunctions

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

using InsideOutSMC: compute_sPCE

using JLD2


Random.seed!(1)

policy = UniformStochasticPolicy(ctl_scale)
# policy = PRBSStochasticPolicy(ctl_scale)
# policy = load("./experiments/double_pendulum/data/double_pendulum_ibis_csmc_ctl.jld2")["ctl"]

closedloop = IBISClosedLoop(
    ibis_dynamics, policy
)

nb_runs = 25

nb_steps = 50
nb_outer_samples = 16
nb_inner_samples = 1_000_000

their_estimator = zeros(nb_runs)

for k in 1:nb_runs
    Flux.reset!(closedloop.ctl)
    spce = compute_sPCE(
        closedloop,
        param_prior,
        init_state,
        nb_steps,
        nb_outer_samples,
        nb_inner_samples
    )
    their_estimator[k] = spce
    @printf("iter: %i, Theirs: %0.4f\n", k, their_estimator[k])
end

@printf("Theirs: %0.4f ± %0.4f\n", mean(their_estimator), std(their_estimator))