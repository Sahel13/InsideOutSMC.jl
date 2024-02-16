using Revise

using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics

include("environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: init_state
using .PendulumEnvironment: dynamics
using .PendulumEnvironment: rb_dynamics
using .PendulumEnvironment: ctl_shift, ctl_scale

using Printf: @printf


function ctl_feature_fn(
    z::AbstractVector{Float64}
)
    return z[1:2]
end

param_prior = Gaussian(
    [10.0, 0.5, 5.0],
    Diagonal([16.0, 1.0, 3.0])
)

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
    ),
)

ctl_mean_fn = Flux.f64(
    Flux.Chain(
        Flux.Dense(recur_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, output_dim),
    ),
)

ctl_log_std = @. log(sqrt([0.5]))

ctl_bijector = (
    Bijectors.Shift(ctl_shift)
    ∘ Bijectors.Scale(ctl_scale)
    ∘ Tanh()
)

policy = StatefulHomoschedasticPolicy(
    ctl_feature_fn,
    ctl_encoder_fn,
    ctl_mean_fn,
    ctl_log_std,
    ctl_bijector,
)

learner = RaoBlackwellClosedLoop(
    rb_dynamics, policy
)

horizon = 25
nb_steps = 50
nb_trajectories = 512

action_penalty = 1e-3
slew_rate_penalty = 0.0
tempering = 0.5

Random.seed!(1)

cond_covar = rao_blackwell_conditional_dynamics_covar(rb_dynamics)
dist = MvNormal(zeros(xdim), cond_covar)
dynamics_noise = rand(dist, nb_steps)

# Unconditioned policy
trajectory = Array{Float64}(undef, xdim+udim, nb_steps + 1)
trajectory[:, 1] = init_state

param_posterior = deepcopy(param_prior)

Flux.reset!(policy)
for t = 1:nb_steps
    action = policy_mean(policy, trajectory[:, t])
    state = trajectory[1:xdim, t]
    next_state = dynamics_mean(dynamics, state, action) + dynamics_noise[:, t]
    trajectory[:, t+1] = vcat(next_state, action)
end

for t in 1:nb_steps
    global param_posterior

    param_posterior = rao_blackwell_dynamics_update(
        rb_dynamics,
        param_posterior,
        trajectory[1:xdim, t],
        trajectory[xdim+1:end, t+1],
        trajectory[1:xdim, t+1]
    )
end
prior_mvn = MvNormal(param_prior.mean, Symmetric(param_prior.covar))
posterior_mvn = MvNormal(param_posterior.mean, Symmetric(param_posterior.covar))
@printf("EIG with unconditioned actions: %0.4f\n", entropy(prior_mvn) - entropy(posterior_mvn))

# Conditioned policy
trajectory = Array{Float64}(undef, xdim+udim, nb_steps + 1)
trajectory[:, 1] = init_state

param_posterior = deepcopy(param_prior)

Flux.reset!(learner.ctl)
for t in 1:nb_steps
    global param_posterior
    global posterior_mvn

    local state_struct
    local idx
    local mvn

    state_struct, _ = smc_with_rao_blackwell_marginal_dynamics(
        min(horizon, (nb_steps + 1) - t),
        nb_trajectories,
        trajectory[:, t],
        learner,
        param_posterior,
        action_penalty,
        slew_rate_penalty,
        tempering,
    )
    idx = rand(Categorical(state_struct.weights))
    reference = state_struct.trajectories[:, :, idx]

    action = reference[xdim+1:end, 2]
    state = trajectory[1:xdim, t]
    next_state = dynamics_mean(dynamics, state, action) + dynamics_noise[:, t]
    trajectory[:, t+1] = vcat(next_state, action)

    param_posterior = rao_blackwell_dynamics_update(
        rb_dynamics,
        param_posterior,
        state,
        action,
        next_state
    )
    posterior_mvn = MvNormal(param_posterior.mean, Symmetric(param_posterior.covar))
    @printf("step: %i, entropy: %0.4f\n", t, entropy(posterior_mvn))
end
prior_mvn = MvNormal(param_prior.mean, Symmetric(param_prior.covar))
posterior_mvn = MvNormal(param_posterior.mean, Symmetric(param_posterior.covar))
@printf("EIG with conditioned actions: %0.4f\n", entropy(prior_mvn) - entropy(posterior_mvn))
