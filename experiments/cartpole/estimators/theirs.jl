using InsideOutSMC

using Random
using Distributions
using Statistics
using LinearAlgebra

import Flux

using Printf: @printf

include("../environment.jl")

using .CartpoleEnvironment: xdim, udim
using .CartpoleEnvironment: step_size
using .CartpoleEnvironment: init_state
using .CartpoleEnvironment: ibis_dynamics
using .CartpoleEnvironment: param_prior
using .CartpoleEnvironment: ctl_scale
using .CartpoleEnvironment: ctl_feature_fn

using InsideOutSMC: compute_sPCE

using JLD2


Random.seed!(1)

policy = UniformStochasticPolicy([ctl_scale])
# policy = PRBSStochasticPolicy([ctl_scale])
# policy = load("./experiments/cartpole/data/cartpole_ibis_csmc_ctl.jld2")["ctl"]

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