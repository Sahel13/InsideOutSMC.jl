using Revise

using InsideOutSMC

using Random
using Distributions
using Statistics
using LinearAlgebra
using LogExpFunctions

using Printf: @printf

include("environment.jl")

using .PendulumEnvironment: xdim, udim
using .PendulumEnvironment: step_size
using .PendulumEnvironment: init_state
using .PendulumEnvironment: ibis_dynamics
using .PendulumEnvironment: param_prior
using .PendulumEnvironment: ctl_scale


function compute_sPCE(
    closedloop::IBISClosedLoop,
    param_prior::Distributions.MultivariateDistribution,
    init_state::Vector{Float64},
    nb_steps::Int,
    nb_outer_samples::Int,
    nb_inner_samples::Int,
)
    xdim = closedloop.dyn.xdim
    udim = closedloop.dyn.udim

    # Generate outer and inner samples.
    outer_param_samples = rand(param_prior, nb_outer_samples)
    inner_param_samples = rand(param_prior, nb_inner_samples, nb_outer_samples)

    # Generate trajectories.
    trajectory_samples = Array{Float64,3}(undef, xdim+udim, nb_outer_samples, nb_steps + 1)
    trajectory_samples[:, :, 1] .= init_state

    @inbounds @views for t = 1:nb_steps
        ibis_conditional_closedloop_sample!(
            closedloop,
            outer_param_samples,
            view(trajectory_samples, :, :, t),
            view(trajectory_samples, :, :, t + 1)
        )
    end

    # Compute the denominator of the integrand.
    scratch_matrix = zeros(xdim, nb_inner_samples + 1)
    trajectory_logpdf = zeros(nb_inner_samples + 1, nb_outer_samples)
    integrand = Vector{Float64}(undef, nb_outer_samples)

    for n in 1:nb_outer_samples
        regularizing_samples = hcat(
            outer_param_samples[:, n],
            reduce(hcat, inner_param_samples[:, n])
        )
        for t = 1:nb_steps
            trajectory_logpdf[:, n] += ibis_conditional_dynamics_logpdf(
                closedloop.dyn,
                regularizing_samples,
                trajectory_samples[1:xdim, n, t],        # state
                trajectory_samples[xdim+1:end, n, t+1],  # action
                trajectory_samples[1:xdim, n, t+1],      # next state
                scratch_matrix
            )
        end
        integrand[n] = (
            trajectory_logpdf[1, n]
            - logsumexp(trajectory_logpdf[:, n])
            + log(nb_inner_samples + 1)
        )
    end
    return mean(integrand)
end


Random.seed!(1)

_param_prior = MvNormal(
    param_prior.mean,
    param_prior.covar
)

rnd_policy = UniformStochasticPolicy([ctl_scale])
rnd_closedloop = IBISClosedLoop(
    ibis_dynamics, rnd_policy
)

nb_runs = 5
nb_steps = 50
nb_outer_samples = 16
nb_inner_samples = 50_000

their_estimator = zeros(nb_runs)

for k in 1:nb_runs
    spce = compute_sPCE(
        rnd_closedloop,
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
