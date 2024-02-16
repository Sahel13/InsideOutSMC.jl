using Revise

using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics

import Zygote
import Flux
import Bijectors

include("common.jl")
include("environment.jl")

using .PendulumEnvironment: init_state
using .PendulumEnvironment: dynamics
using .PendulumEnvironment: rb_dynamics
using .PendulumEnvironment: ctl_shift, ctl_scale

using Plots
using Printf: @printf


Random.seed!(1)

param_prior = Gaussian(
    [10.0, 0.0, 5.0],
    1.0 * Matrix{Float64}(I, 3, 3)
)


function ctl_feature_fn(
    z::AbstractVector{Float64}
)::Vector{Float64}
    return z[1:2]
end


input_dim = 2
output_dim = 1
recur_size = 64
dense_size = 256

ctl_encoder_fn = Flux.f64(
    Flux.Chain(
        Flux.Dense(input_dim, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, recur_size),
        Flux.LSTM(recur_size, recur_size),
        Flux.LSTM(recur_size, recur_size),
        # Flux.GRU(recur_size, recur_size),
        # Flux.GRU(recur_size, recur_size),
    ),
)

ctl_mean_fn = Flux.f64(
    Flux.Chain(
        Flux.Dense(recur_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, output_dim),
    ),
)

ctl_log_std = @. log(sqrt([1.0]))

ctl_bijector = (
    Bijectors.Shift(ctl_shift)
    ∘ Bijectors.Scale(ctl_scale)
    ∘ Tanh()
)

learner_policy = StatefulHomoschedasticPolicy(
    ctl_feature_fn,
    ctl_encoder_fn,
    ctl_mean_fn,
    ctl_log_std,
    ctl_bijector,
)

evaluator_policy = StatefulHomoschedasticPolicy(
    ctl_feature_fn,
    ctl_encoder_fn,
    ctl_mean_fn,
    [-20.0],
    ctl_bijector,
)

random_policy = UniformStochasticPolicy([ctl_scale])

learner_loop = RaoBlackwellClosedLoop(
    rb_dynamics, learner_policy
)

evaluator_loop = RaoBlackwellClosedLoop(
    rb_dynamics, evaluator_policy
)

random_loop = RaoBlackwellClosedLoop(
    rb_dynamics, random_policy
)

action_penalty = 0.0
slew_rate_penalty = 0.0
tempering = 0.5

nb_steps = 50
nb_trajectories = 256

nb_iter = 15
opt_state = Flux.setup(Flux.Optimise.Adam(5e-4), learner_loop)
batch_size = 64

learner_loop, _ = score_climbing_with_rao_blackwell_marginal_dynamics(
    nb_iter,
    opt_state,
    batch_size,
    nb_steps,
    nb_trajectories,
    init_state,
    learner_loop,
    evaluator_loop,
    param_prior,
    action_penalty,
    slew_rate_penalty,
    tempering,
    true
)

# Evaluate policy
policy_dict = Dict(
    "random" => UniformStochasticPolicy([ctl_scale]),
    "learned" => learner_loop.ctl,
)

entropy_vals = compare_policies_with_rao_blackwell_updates(
    dynamics,
    rb_dynamics,
    policy_dict,
    param_prior,
    init_state,
    nb_steps,
    1000
)
display(entropy_vals)

# Sample a trajectory
trajectory, param_posterior = rollout_with_rao_blackwell_updates(
    dynamics,
    rb_dynamics,
    learner_loop.ctl,
    param_prior,
    init_state,
    nb_steps
)

prior_mvn = MvNormal(param_prior.mean, Symmetric(param_prior.covar))
posterior_mvn = MvNormal(param_posterior.mean, Symmetric(param_posterior.covar))
@printf("EIG: %0.4f\n", entropy(prior_mvn) - entropy(posterior_mvn))

plot(trajectory')