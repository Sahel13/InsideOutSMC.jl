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
using .PendulumEnvironment: rb_dynamics
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: ctl_scale, ctl_shift

using JLD2
using DelimitedFiles
using Plots


function ctl_feature_fn(
    z::AbstractVector{Float64}
)::Vector{Float64}
    return z[1:2]
end


rnd_policy = UniformStochasticPolicy([ctl_scale])
prbs_policy = PRBSStochasticPolicy([ctl_scale])
rb_policy = load("./experiments/pendulum/linear/data/linear_pendulum_rb_csmc_ctl.jld2")["ctl"]
ibis_policy = load("./experiments/pendulum/linear/data/linear_pendulum_ibis_csmc_ctl.jld2")["ctl"]

Flux.reset!(rb_policy)
Flux.reset!(ibis_policy)

policy_dict = Dict(
    "rnd" => rnd_policy,
    "prbs" => prbs_policy,
    "rb" => rb_policy,
    "ibis" => ibis_policy,
)

nb_steps = 50
nb_evals = 1024

info_gain = Dict(key => Matrix{Float64}(undef, nb_steps, nb_evals) for key in keys(policy_dict))

info_gain_mean = Matrix{Float64}(undef, 4, nb_steps)
info_gain_std = Matrix{Float64}(undef, 4, nb_steps)

for k in 1:nb_evals
    params = rand(
        MvNormal(
            param_prior.mean,
            param_prior.covar
        )
    )

    dynamics = StochasticDynamics(
        xdim, udim,
        (x, u) -> drift_fn(params, x, u),
        (x, u) -> diffusion_fn(params, x, u),
        step_size,
    )

    dynamics_seed = rand(1:Int(1e9))
    Random.seed!(dynamics_seed)

    cond_covar = rao_blackwell_conditional_dynamics_covar(rb_dynamics)
    dist = MvNormal(zeros(xdim), cond_covar)
    dynamics_noise = rand(dist, nb_steps)

    policy_seed = rand(1:Int(1e9))

    for (policy_name, policy_obj) in pairs(policy_dict)
        Random.seed!(policy_seed)

        trajectory = Array{Float64}(undef, xdim+udim, nb_steps + 1)
        trajectory[:, 1] = init_state

        Flux.reset!(policy_obj)
        for t = 1:nb_steps
            state = trajectory[1:xdim, t]
            action = policy_mean(policy_obj, trajectory[:, t])
            next_state = dynamics_mean(dynamics, state, action) + dynamics_noise[:, t]
            trajectory[:, t+1] = vcat(next_state, action)
        end

        param_posterior = deepcopy(param_prior)
        for t in 1:nb_steps
            param_posterior = rao_blackwell_dynamics_update(
                rb_dynamics,
                param_posterior,
                trajectory[1:xdim, t],
                trajectory[xdim+1:end, t+1],
                trajectory[1:xdim, t+1]
            )

            prior_mvn = MvNormal(param_prior.mean, Symmetric(param_prior.covar))
            posterior_mvn = MvNormal(param_posterior.mean, Symmetric(param_posterior.covar))

            info_gain[policy_name][t, k] = entropy(prior_mvn) - entropy(posterior_mvn)
        end
    end
end

info_gain_stats = Dict(key => [] for key in keys(policy_dict))
for (key, val) in pairs(info_gain)
    info_gain_stats[key] = [mean(val, dims=2)[:, 1], std(val, dims=2)[:, 1]]
end

info_gain_mean[1, :] = info_gain_stats["rnd"][1]
info_gain_mean[2, :] = info_gain_stats["prbs"][1]
info_gain_mean[3, :] = info_gain_stats["rb"][1]
info_gain_mean[4, :] = info_gain_stats["ibis"][1]

info_gain_std[1, :] = info_gain_stats["rnd"][2]
info_gain_std[2, :] = info_gain_stats["prbs"][2]
info_gain_std[3, :] = info_gain_stats["rb"][2]
info_gain_std[4, :] = info_gain_stats["ibis"][2]

time_steps = 5:5:50

plot(
    time_steps,
    info_gain_mean[1, :][1:5:end],
    yerr=info_gain_std[1, :][1:5:end],
    seriescolor=:deepskyblue3,
    label="RND", marker=:circle
)

plot!(
    time_steps,
    info_gain_mean[2, :][1:5:end],
    yerr=info_gain_std[2, :][1:5:end],
    seriescolor=:red,
    label="PRBS", marker=:circle
)

plot!(
    time_steps,
    info_gain_mean[3, :][1:5:end],
    yerr=info_gain_std[3, :][1:5:end],
    seriescolor=:chocolate2,
    label="RB",
    marker=:circle
)

plot!(
    time_steps,
    info_gain_mean[4, :][1:5:end],
    yerr=info_gain_std[4, :][1:5:end],
    seriescolor=:mediumseagreen,
    label="IBIS",
    marker=:circle
)

writedlm(
    "./experiments/pendulum/linear/data/linear_pendulum_info_gain_rnd.csv",
    hcat(time_steps, info_gain_mean[1, :][1:5:end], info_gain_std[1, :][1:5:end]), ','
)

writedlm(
    "./experiments/pendulum/linear/data/linear_pendulum_info_gain_prbs.csv",
    hcat(time_steps, info_gain_mean[2, :][1:5:end], info_gain_std[2, :][1:5:end]), ','
)

writedlm(
    "./experiments/pendulum/linear/data/linear_pendulum_info_gain_rb.csv",
    hcat(time_steps, info_gain_mean[3, :][1:5:end], info_gain_std[3, :][1:5:end]), ','
)

writedlm(
    "./experiments/pendulum/linear/data/linear_pendulum_info_gain_ibis.csv",
    hcat(time_steps, info_gain_mean[4, :][1:5:end], info_gain_std[4, :][1:5:end]), ','
)