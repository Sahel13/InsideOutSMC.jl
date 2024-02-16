using Random
using Distributions
using LinearAlgebra
using LogExpFunctions

import Zygote
import Flux


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