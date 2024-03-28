using Base.Threads: nthreads, @threads, @spawn
using Base.Iterators: partition

using Random
using Distributions
using LinearAlgebra
using LogExpFunctions

import Zygote
import Flux


function compute_sPCE_for_myopic_adaptive_policy(
    adaptive_loop::RaoBlackwellAdaptiveLoop,
    param_prior::Gaussian,
    init_state::Vector{Float64},
    nb_steps::Int,
    nb_outer_samples::Int,
    nb_inner_samples::Int,
)
    xdim = adaptive_loop.dyn.xdim
    udim = adaptive_loop.dyn.udim

    _param_prior = MvNormal(param_prior.mean, param_prior.covar)

    # Generate outer and inner samples
    outer_param_samples = rand(_param_prior, nb_outer_samples)
    inner_param_samples = rand(_param_prior, nb_inner_samples, nb_outer_samples)

    # Generate trajectories
    trajectory_samples = Array{Float64,3}(undef, xdim+udim, nb_steps + 1, nb_outer_samples)
    trajectory_samples[:, 1, :] .= init_state

    # Param struct
    param_struct = RaoBlackwellParamStruct(param_prior, nb_steps, nb_outer_samples)

    @inbounds @views for t = 1:nb_steps
        # sample from adaptive policy
        trajectory_samples[xdim+1:xdim+udim, t+1, :] = adaptive_policy_sample(
            adaptive_loop.ctl,
            param_struct.distributions[t, :],
            trajectory_samples[:, t, :]
        )

        # sample from conditional dynamics
        rao_blackwell_conditional_dynamics_sample!(
            adaptive_loop.dyn,
            outer_param_samples,
            view(trajectory_samples, 1:xdim, t, :),              # state
            view(trajectory_samples, xdim+1:xdim+udim, t+1, :),  # action
            view(trajectory_samples, 1:xdim, t+1, :)             # next state
        )

        # closed-form param update
        @views @inbounds for n = 1:nb_outer_samples
            q = param_struct.distributions[t, n]
            x = trajectory_samples[1:xdim, t, n]
            u = trajectory_samples[xdim+1:end, t+1, n]
            xn = trajectory_samples[1:xdim, t+1, n]
            qn = param_struct.distributions[t+1, n]

            rao_blackwell_dynamics_update!(adaptive_loop.dyn, q, x, u, xn, qn)
        end
    end

    # Compute the denominator of the integrand
    scratch_matrix = zeros(xdim, nb_inner_samples + 1)
    trajectory_logpdf = zeros(nb_inner_samples + 1, nb_outer_samples)
    integrand = Vector{Float64}(undef, nb_outer_samples)

    for n in 1:nb_outer_samples
        regularizing_samples = hcat(
            outer_param_samples[:, n],
            reduce(hcat, inner_param_samples[:, n])
        )
        for t = 1:nb_steps
            trajectory_logpdf[:, n] += rao_blackwell_conditional_dynamics_logpdf(
                adaptive_loop.dyn,
                regularizing_samples,
                trajectory_samples[1:xdim, t, n],        # state
                trajectory_samples[xdim+1:end, t+1, n],  # action
                trajectory_samples[1:xdim, t+1, n],      # next state
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


function compute_sPCE_for_myopic_adaptive_policy(
    adaptive_loop::IBISAdaptiveLoop,
    param_prior::MultivariateDistribution,
    init_state::Vector{Float64},
    nb_steps::Int,
    nb_outer_samples::Int,
    nb_inner_samples::Int,
    nb_ibis_particles::Int,
    ibis_proposal::Function,
    nb_ibis_moves::Int,
    nb_ibis_threads::Int = 2,
)
    xdim = adaptive_loop.dyn.xdim
    udim = adaptive_loop.dyn.udim

    # Generate outer and inner samples
    outer_param_samples = rand(param_prior, nb_outer_samples)
    inner_param_samples = rand(param_prior, nb_inner_samples, nb_outer_samples)

    # Generate trajectories
    trajectory_samples = Array{Float64,3}(undef, xdim+udim, nb_steps + 1, nb_outer_samples)
    trajectory_samples[:, 1, :] .= init_state

    # IBIS related objects
    ibis_scratch = Array{Float64}(undef, xdim, nb_ibis_particles, nb_outer_samples)
    ibis_struct = IBISParamStruct(param_prior, nb_steps, nb_ibis_particles, nb_outer_samples, ibis_scratch)

    chunk_size = round(Int, nb_outer_samples / nb_ibis_threads)
    ranges = partition(1:nb_outer_samples, chunk_size)

    trajectory_views = [view(trajectory_samples, :, :, range) for range in ranges]
    ibis_struct_views = [view_struct(ibis_struct, range) for range in ranges]

    @inbounds @views for t = 1:nb_steps
        # sample from adaptive policy
        trajectory_samples[xdim+1:xdim+udim, t+1, :] = adaptive_policy_sample(
            adaptive_loop.ctl,
            view(ibis_struct.particles, :, t, :, :),
            view(ibis_struct.weights, t, :, :),
            view(ibis_struct.log_weights, t, :, :),
            view(trajectory_samples, :, t, :)
        )

        # sample from conditional dynamics
        ibis_conditional_dynamics_sample!(
            adaptive_loop.dyn,
            outer_param_samples,
            view(trajectory_samples, 1:xdim, t, :),              # state
            view(trajectory_samples, xdim+1:xdim+udim, t+1, :),  # action
            view(trajectory_samples, 1:xdim, t+1, :)             # next state
        )

        # run IBIS outside
        tasks = map(zip(trajectory_views, ibis_struct_views)) do (traj_view, struct_view)
            @spawn begin
                batch_ibis_step!(
                    t,
                    traj_view,
                    adaptive_loop.dyn,
                    param_prior,
                    ibis_proposal,
                    nb_ibis_moves,
                    struct_view
                );
            end
        end
        fetch.(tasks);
    end

    # Compute the denominator of the integrand
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
                adaptive_loop.dyn,
                regularizing_samples,
                trajectory_samples[1:xdim, t, n],        # state
                trajectory_samples[xdim+1:end, t+1, n],  # action
                trajectory_samples[1:xdim, t+1, n],      # next state
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


function compute_sPCE(
    closedloop::IBISClosedLoop,
    param_prior::MultivariateDistribution,
    init_state::Vector{Float64},
    nb_steps::Int,
    nb_outer_samples::Int,
    nb_inner_samples::Int,
)
    xdim = closedloop.dyn.xdim
    udim = closedloop.dyn.udim

    # Generate outer and inner samples
    outer_param_samples = rand(param_prior, nb_outer_samples)
    inner_param_samples = rand(param_prior, nb_inner_samples, nb_outer_samples)

    # Generate trajectories
    trajectory_samples = Array{Float64,3}(undef, xdim+udim, nb_steps + 1, nb_outer_samples)
    trajectory_samples[:, 1, :] .= init_state

    @inbounds @views for t = 1:nb_steps
        ibis_conditional_closedloop_sample!(
            closedloop,
            outer_param_samples,
            view(trajectory_samples, :, t, :),
            view(trajectory_samples, :, t+1, :)
        )
    end

    # Compute the denominator of the integrand
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
                trajectory_samples[1:xdim, t, n],        # state
                trajectory_samples[xdim+1:end, t+1, n],  # action
                trajectory_samples[1:xdim, t+1, n],      # next state
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


function policy_gradient_objective(
    samples::AbstractArray{Float64},
    closedloop::Union{ClosedLoop, IBISClosedLoop, RaoBlackwellClosedLoop},
)
    Flux.reset!(closedloop.ctl)

    _, nb_steps_p_1, nb_samples = size(samples)
    nb_steps = nb_steps_p_1 - 1

    xdim = closedloop.dyn.xdim

    ll = 0.0
    for t = 1:nb_steps
        ll += sum(
            policy_logpdf(
                closedloop.ctl,
                samples[:, t, :],
                samples[xdim+1:end, t+1, :]
            )
        )
    end
    return ll / nb_samples
end


function maximization!(
    opt_state::NamedTuple,
    samples::AbstractArray{Float64},
    closedloop::Union{ClosedLoop, IBISClosedLoop, RaoBlackwellClosedLoop},
)
    nll, grad = Zygote.withgradient(closedloop) do f
        -1.0 * policy_gradient_objective(samples, f)
    end
    _, closedloop = Flux.update!(opt_state, closedloop, grad[1])
    return nll, closedloop
end


function ibis_info_gain_increment(
    id::IBISDynamics,
    ps::AbstractMatrix{Float64},
    lws::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)::Float64
    lls = map(eachcol(ps)) do p
        ibis_conditional_dynamics_logpdf(id, p, x, u, xn)
    end
    mod_lws = softmax(lws .+ lls)
    return dot(mod_lws, lls) - logsumexp(lls .+ lws) + logsumexp(lws)
end


function ibis_info_gain_increment(
    id::IBISDynamics,
    ps::AbstractMatrix{Float64},
    lws::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
    sc::AbstractMatrix{Float64},
)::Float64
    lls = ibis_conditional_dynamics_logpdf(id, ps, x, u, xn, sc)
    mod_lws = softmax(lws .+ lls)
    return dot(mod_lws, lls) - logsumexp(lls .+ lws) + logsumexp(lws)
end


function rao_blackwell_info_gain_increment(
    bd::RaoBlackwellDynamics,
    q::Gaussian,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)::Float64
    cond_covar = rao_blackwell_conditional_dynamics_covar(bd)

    h = bd.step
    A = [1.0 h; 0.0 1.0]
    B = [0.0; h] .* bd.feature_fn(x, u)'

    inv_S = inv(cond_covar + (B * q.covar) * B')
    K = (q.covar * B') * inv_S

    qn = Gaussian(
        q.mean + K * (xn - A * x - B * q.mean),
        (I - K * B) * q.covar
    )

    diff = xn - (A * x + B * qn.mean)
    expectation_of_log = (
        - 0.5 * bd.xdim * log(2.0 * pi)
        - 0.5 * logdet(cond_covar)
        - 0.5 * diff' * inv(cond_covar) * diff
        - 0.5 * tr(inv(cond_covar) * (B * qn.covar * B'))
    )

    diff = xn - (A * x + B * q.mean)
    marg_covar = cond_covar + (B * q.covar) * B'
    log_of_expectation = (
        - 0.5 * bd.xdim * log(2.0 * pi)
        - 0.5 * logdet(marg_covar)
        - 0.5 * diff' * inv(marg_covar) * diff
    )
    return expectation_of_log - log_of_expectation
end


@inline function inverse_cdf!(
    idx::AbstractVector{Int},
    uniforms::AbstractVector{Float64},
    weights::AbstractVector{Float64},
)
    M = length(uniforms)

    i = 1
    cumsum = weights[begin]
    for m = 1:M
        while uniforms[m] > cumsum
            i += 1
            cumsum += weights[i]
        end
        idx[m] = i
    end
    return idx
end


function multinomial_resampling!(state_struct::StateStruct)
    nb_particles = state_struct.nb_trajectories
    uniforms = rand(nb_particles + 1)
    cs = cumsum(-1.0 .* log.(uniforms))
    inverse_cdf!(
        state_struct.resampled_idx,
        cs[begin:end-1] ./ cs[end],
        state_struct.weights,
    )
end


function systematic_resampling!(state_struct::StateStruct)
    uniform = rand()
    map!(
        i -> (uniform + i) / state_struct.nb_trajectories,
        state_struct.rvs,
        0:state_struct.nb_trajectories - 1
    )
    inverse_cdf!(
        state_struct.resampled_idx,
        state_struct.rvs,
        state_struct.weights,
    )
end


function systematic_resampling!(
    time_idx::Int,
    param_struct::IBISParamStruct
)
    uniform = rand()
    map!(
        i -> (uniform + i) / param_struct.nb_particles,
        param_struct.rvs,
        0:param_struct.nb_particles - 1
    )
    inverse_cdf!(
        param_struct.resampled_idx,
        param_struct.rvs,
        view(param_struct.weights, time_idx, :)
    )
end


function normalize_weights!(state_struct::StateStruct)
    return softmax!(state_struct.weights, state_struct.log_weights)
end


function normalize_weights!(
    time_idx::Int,
    param_struct::IBISParamStruct
)
    weights = @view param_struct.weights[time_idx, :]
    log_weights = @view param_struct.log_weights[time_idx, :]
    softmax!(weights, log_weights)
end


@inline function effective_sample_size(weights::AbstractVector{Float64})
    return inv(sum(abs2, weights))
end


@inline function categorical(
    uniform_rv::Float64,
    weights::AbstractVector{Float64}
)
    cum_weights = 0.0
    for idx in eachindex(weights)
        @inbounds cum_weights += weights[idx]
        if uniform_rv <= cum_weights
            return idx
        end
    end
    # In the case where the weights sum to slightly less than 1
    ErrorException("Weights do not sum up to one")
    return NaN
end
