using Random
using LinearAlgebra
using LogExpFunctions

import DistributionsAD as Distributions

import Flux
import Bijectors



struct MyopicAdaptiveRaoBlackwellPolicy{
    S<:RaoBlackwellClosedLoop
} <:StochasticPolicy
    nb_particles::Int
    closedloop::S
    action_penalty::Float64
    slew_rate_penalty::Float64
    tempering::Float64
end


function _myopic_adaptive_rao_blackwell_policy_sample(
    nb_particles::Int,
    distributions::AbstractVector{Gaussian},
    states::AbstractMatrix{Float64},
    closedloop::RaoBlackwellClosedLoop,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
)
    xdim = closedloop.dyn.xdim

    next_states = rao_blackwell_marginal_closedloop_sample(
        closedloop, distributions, states
    )

    log_weights = zeros(nb_particles)
    @views @inbounds for n = 1:nb_particles
        q = distributions[n]
        x = states[1:xdim, n]
        u = next_states[xdim+1:end, n]
        xn = next_states[1:xdim, n]
        up = states[xdim+1:end, n]

        info_gain = rao_blackwell_info_gain_increment(closedloop.dyn, q, x, u, xn)
        reward = info_gain - action_penalty * dot(u, u) - slew_rate_penalty * dot(u - up, u - up)
        log_weights[n] = tempering * reward
    end

    weights = softmax!(log_weights)
    idx = rand(Categorical(weights))
    return next_states[xdim+1:end, idx]
end


function adaptive_policy_sample(
    mp::MyopicAdaptiveRaoBlackwellPolicy,
    q::Gaussian,
    z::AbstractVector{Float64},
)
    states = repeat(z, 1, mp.nb_particles)
    dists = Vector{Gaussian}(undef, mp.nb_particles)
    @views @inbounds for n in 1:mp.nb_particles
        dists[n] = deepcopy(q)
    end

    return _myopic_adaptive_rao_blackwell_policy_sample(
        mp.nb_particles,
        dists,
        states,
        mp.closedloop,
        mp.action_penalty,
        mp.slew_rate_penalty,
        mp.tempering
    )
end


function adaptive_policy_sample(
    mp::MyopicAdaptiveRaoBlackwellPolicy,
    qs::AbstractVector{Gaussian},
    zs::AbstractMatrix{Float64},
)
    return reduce(hcat,
        map(qs, eachcol(zs)) do q, z
            adaptive_policy_sample(mp, q, z)
        end
    )
end


struct RaoBlackwellAdaptiveLoop{
    S<:RaoBlackwellDynamics,
    T<:StochasticPolicy
}
    dyn::S
    ctl::T
end


function rao_blackwell_marginal_adaptive_loop_sample(
    al::RaoBlackwellAdaptiveLoop,
    qs::AbstractVector{Gaussian},
    zs::AbstractMatrix{Float64},
)
    xs = zs[1:al.dyn.xdim, :]
    us = adaptive_policy_sample(al.ctl, qs, zs)

    _sample_fn = (q, x, u) -> rao_blackwell_marginal_dynamics_sample(al.dyn, q, x, u)
    xns = reduce(hcat, map(_sample_fn, qs, eachcol(xs), eachcol(us)))
    return vcat(xns, us)
end


struct MyopicAdaptiveIBISPolicy{
    S<:IBISClosedLoop
} <:StochasticPolicy
    nb_particles::Int
    closedloop::S
    action_penalty::Float64
    slew_rate_penalty::Float64
    tempering::Float64
    scratch::AbstractArray{Float64,3}
end


function _myopic_adaptive_ibis_policy_sample(
    nb_particles::Int,
    param_particles::AbstractArray{Float64,3},
    param_weights::AbstractMatrix{Float64},
    param_log_weights::AbstractMatrix{Float64},
    states::AbstractMatrix{Float64},
    next_states::AbstractMatrix{Float64},
    closedloop::IBISClosedLoop,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    scratch::AbstractArray{Float64,3},
)
    xdim = closedloop.dyn.xdim

    ibis_marginal_closedloop_sample!(
        closedloop, param_particles, param_weights, states, next_states
    )

    log_weights = zeros(nb_particles)
    @views @inbounds for n = 1:nb_particles
        u = next_states[xdim+1:end, n]
        up = states[xdim+1:end, n]

        info_gain = ibis_info_gain_increment(
            closedloop.dyn,
            param_particles[:, :, n],
            param_log_weights[:, n],
            states[1:xdim, n],
            next_states[xdim+1:end, n],
            next_states[1:xdim, n],
            scratch[:, :, n]
        )
        reward = info_gain - action_penalty * dot(u, u) - slew_rate_penalty * dot(u - up, u - up)
        log_weights[n] = tempering * reward
    end

    weights = softmax!(log_weights)
    idx = rand(Categorical(weights))
    return next_states[xdim+1:end, idx]
end


function adaptive_policy_sample(
    mp::MyopicAdaptiveIBISPolicy,
    p::AbstractMatrix{Float64},
    w::AbstractVector{Float64},
    lw::AbstractVector{Float64},
    z::AbstractVector{Float64},
)
    particles = repeat(p, 1, 1, mp.nb_particles)
    weights = repeat(w, 1, mp.nb_particles)
    log_weights = repeat(lw, 1, mp.nb_particles)
    states = repeat(z, 1, mp.nb_particles)
    next_states = similar(states)  # ugly but necessary

    return _myopic_adaptive_ibis_policy_sample(
        mp.nb_particles,
        particles,
        weights,
        log_weights,
        states,
        next_states,
        mp.closedloop,
        mp.action_penalty,
        mp.slew_rate_penalty,
        mp.tempering,
        mp.scratch
    )
end


function adaptive_policy_sample(
    mp::MyopicAdaptiveIBISPolicy,
    ps::AbstractArray{Float64,3},
    ws::AbstractMatrix{Float64},
    lws::AbstractMatrix{Float64},
    zs::AbstractMatrix{Float64},
)
    return reduce(hcat,
        map(
            eachslice(ps, dims=3),
            eachcol(ws),
            eachcol(lws),
            eachcol(zs)
        ) do p, w, lw, z
            adaptive_policy_sample(mp, p, w, lw, z)
        end
    )
end


struct IBISAdaptiveLoop{
    S<:IBISDynamics,
    T<:StochasticPolicy
}
    dyn::S
    ctl::T
end


function ibis_marginal_adaptive_loop_sample(
    al::IBISAdaptiveLoop,
    ps::AbstractArray{Float64,3},
    ws::AbstractMatrix{Float64},
    lws::AbstractMatrix{Float64},
    zs::AbstractMatrix{Float64},
)
    us = adaptive_policy_sample(al.ctl, ps, ws, lws, zs)

    param_dim, _, nb_samples = size(ps)

    rvs = rand(nb_samples)
    resampled_ps = zeros(param_dim, nb_samples)
    for n in 1:nb_samples
        idx = categorical(rvs[n], ws[:, n])
        resampled_ps[:, n] = ps[:, idx, n]
    end

    xs = zs[1:al.dyn.xdim, :]
    xns = similar(xs)
    ibis_conditional_dynamics_sample!(al.dyn, resampled_ps, xs, us, xns)
    return vcat(xns, us)
end