using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics

import Zygote
import Flux
import Bijectors

include("../environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: step_size
using .PendulumEnvironment: init_state
using .PendulumEnvironment: drift_fn
using .PendulumEnvironment: diffusion_fn
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: ctl_scale, ctl_shift
using .PendulumEnvironment: ctl_feature_fn

using JLD2
using DelimitedFiles
using Plots


ibis_policy = load("./experiments/pendulum/linear/data/linear_pendulum_ibis_csmc_ctl.jld2")["ctl"]
Flux.reset!(ibis_policy)

m, l = 1.0, 1.0
g, d = 9.81, 1e-3

true_params = [
    3.0 * g / (2.0 * l),
    3.0 * d / (m * l^2),
    3.0 / (m * l^2),
]

dynamics = StochasticDynamics(
    xdim, udim,
    (x, u) -> drift_fn(true_params, x, u),
    (x, u) -> diffusion_fn(true_params, x, u),
    step_size,
)

nb_steps = 50

trajectory = Array{Float64}(undef, xdim+udim, nb_steps + 1)
trajectory[:, 1] = init_state

Flux.reset!(ibis_policy)
for t = 1:nb_steps
    state = trajectory[1:xdim, t]
    action = policy_mean(ibis_policy, trajectory[:, t])
    next_state = dynamics_sample(dynamics, state, action)
    trajectory[:, t+1] = vcat(next_state, action)
end
plot(trajectory')

time_steps = 1:1:nb_steps+1
writedlm(
    "./experiments/pendulum/linear/data/linear_pendulum_ibis_csmc_trajectory.csv",
    hcat(time_steps, trajectory'), ','
)