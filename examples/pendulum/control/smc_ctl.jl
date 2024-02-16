using Revise

using InsideOutSMC

using Random
using Distributions
using LinearAlgebra

import Zygote
import Flux
import Bijectors

include("environment.jl")

using .PendulumEnvironment: init_state
using .PendulumEnvironment: dynamics
using .PendulumEnvironment: reward_fn
using .PendulumEnvironment: ctl_shift, ctl_scale

using Plots


Random.seed!(1)

function ctl_feature_fn(
    z::AbstractVector{Float64}
)
    return vcat(
        sin(z[1]),
        cos(z[1]),
        z[2],
    )
end

input_dim = 3
output_dim = 1
recur_size = 32
dense_size = 256

ctl_encoder_fn = Flux.f64(
    Flux.Chain(
        Flux.Dense(input_dim, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, recur_size, Flux.relu),
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

ctl_log_std = @. log(sqrt([1.0]))

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

closedloop = ClosedLoop(
    dynamics, policy
)

tempering = 1e-1

nb_iter = 15
nb_steps = 100
nb_trajectories = 256

opt_state = Flux.setup(Flux.Optimise.Adam(1e-3), closedloop)
batch_size = 32
verbose = true

closedloop, samples = score_climbing(
    nb_iter,
    opt_state,
    batch_size,
    nb_steps,
    nb_trajectories,
    init_state,
    closedloop,
    reward_fn,
    tempering,
    verbose
)

Flux.reset!(closedloop.ctl)
trajectory = Array{Float64}(undef, 3, nb_steps + 1)
trajectory[:, 1] = init_state
for t = 1:nb_steps
    trajectory[:, t+1] = closedloop_mean(closedloop, trajectory[:, t])
end
plot(trajectory')