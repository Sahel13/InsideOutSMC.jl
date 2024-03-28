using InsideOutSMC

using Random
using Distributions
using Statistics
using LinearAlgebra
using LogExpFunctions

import Flux

using Printf: @printf

include("../environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: step_size
using .PendulumEnvironment: init_state
using .PendulumEnvironment: ibis_dynamics
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: ctl_scale
using .PendulumEnvironment: ctl_feature_fn

using JLD2


Random.seed!(1)

_param_prior = MvNormal(
    param_prior.mean,
    param_prior.covar
)

policy = UniformStochasticPolicy([ctl_scale])
# policy = PRBSStochasticPolicy([ctl_scale])
# policy = load("./experiments/pendulum/linear/data/linear_pendulum_rb_csmc_ctl.jld2")["ctl"]
# policy = load("./experiments/pendulum/linear/data/linear_pendulum_ibis_csmc_ctl.jld2")["ctl"]

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
        _param_prior,
        init_state,
        nb_steps,
        nb_outer_samples,
        nb_inner_samples
    )
    their_estimator[k] = spce
    @printf("iter: %i, Theirs: %0.4f\n", k, their_estimator[k])
end

@printf("Theirs: %0.4f ± %0.4f\n", mean(their_estimator), std(their_estimator))
