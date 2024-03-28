using InsideOutSMC

using Random
using Distributions
using LinearAlgebra
using Statistics

import Zygote
import Flux
import Bijectors

include("../environment.jl")

using .DoublePendulumEnvironment: xdim, udim
using .DoublePendulumEnvironment: step_size
using .DoublePendulumEnvironment: init_state
using .DoublePendulumEnvironment: drift_fn
using .DoublePendulumEnvironment: diffusion_fn
using .DoublePendulumEnvironment: param_prior
using .DoublePendulumEnvironment: ctl_shift, ctl_scale
using .DoublePendulumEnvironment: ctl_feature_fn

using JLD2
using DelimitedFiles
using Plots


ibis_policy = load("./experiments/double_pendulum/data/double_pendulum_ibis_csmc_ctl.jld2")["ctl"]
Flux.reset!(ibis_policy)

true_params = [1.0, 1.0, 1.0, 1.0]

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
    "./experiments/double_pendulum/data/double_pendulum_ibis_csmc_trajectory.csv",
    hcat(time_steps, trajectory'), ','
)