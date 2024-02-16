using Random
using LinearAlgebra

import DistributionsAD as Distributions

import Flux
import Bijectors


struct ClosedLoop{
    S<:StochasticDynamics,
    T<:StochasticPolicy
}
    dyn::S
    ctl::T
end


function closedloop_mean(
    cl::ClosedLoop,
    z::AbstractVector{Float64},
)
    x = z[begin:cl.dyn.xdim]
    u = policy_mean(cl.ctl, z)
    xn = dynamics_mean(cl.dyn, x, u)
    return vcat(xn, u)
end


function closedloop_sample(
    cl::ClosedLoop,
    zs::AbstractMatrix{Float64}
)
    xs = zs[begin:cl.dyn.xdim, :]
    us = policy_sample(cl.ctl, zs)

    _sample_fn = (x, u) -> dynamics_sample(cl.dyn, x, u)
    xns = reduce(hcat, map(_sample_fn, eachcol(xs), eachcol(us)))
    return vcat(xns, us)
end


function closedloop_logpdf(
    cl::ClosedLoop,
    z::AbstractMatrix{Float64},
    zn::AbstractMatrix{Float64},
)
    x = z[begin:cl.dyn.xdim, :]
    xn = zn[begin:cl.dyn.xdim, :]
    u = zn[cl.dyn.xdim+1:end, :]

    ll = policy_logpdf(cl.ctl, z, u)
    ll += map(eachcol(x), eachcol(u), eachcol(xn)) do _x, _u, _xn
        dynamics_logpdf(cl.dyn, _x, _u, _xn)
    end
    return ll
end

Flux.@functor ClosedLoop


struct IBISClosedLoop{
    S<:IBISDynamics,
    T<:StochasticPolicy
}
    dyn::S
    ctl::T
end


function ibis_conditional_closedloop_mean(
    cl::IBISClosedLoop,
    p::AbstractVector{Float64},
    z::AbstractVector{Float64},
)
    x = z[begin:cl.dyn.xdim]
    u = policy_mean(cl.ctl, z)
    xn = ibis_conditional_dynamics_mean(cl.dyn, p, x, u)
    return vcat(xn, u)
end


function ibis_conditional_closedloop_sample(
    cl::IBISClosedLoop,
    ps::AbstractMatrix{Float64},
    zs::AbstractMatrix{Float64},
)
    xs = zs[begin:cl.dyn.xdim, :]
    us = policy_sample(cl.ctl, zs)
    xns = ibis_conditional_dynamics_sample(cl.dyn, ps, xs, us)
    return vcat(xns, us)
end


function ibis_conditional_closedloop_sample!(
    cl::IBISClosedLoop,
    ps::AbstractMatrix{Float64},
    zs::AbstractMatrix{Float64},
    zns::AbstractMatrix{Float64},
)
    xdim = cl.dyn.xdim
    udim = cl.dyn.udim

    zns[xdim+1:xdim+udim, :] .= policy_sample(cl.ctl, zs)
    ibis_conditional_dynamics_sample!(
        cl.dyn,
        ps,
        view(zs, 1:xdim, :),
        view(zns, xdim+1:xdim+udim, :),
        view(zns, 1:xdim, :)
    )
end


function ibis_marginal_closedloop_sample(
    cl::IBISClosedLoop,
    ps::AbstractArray{Float64,3},
    ws::AbstractMatrix{Float64},
    zs::AbstractMatrix{Float64},
)
    param_dim, _, nb_samples = size(ps)

    rvs = rand(nb_samples)
    resampled_ps = zeros(param_dim, nb_samples)
    for n in 1:nb_samples
        idx = categorical(rvs[n], ws[:, n])
        resampled_ps[:, n] = ps[:, idx, n]
    end

    return ibis_conditional_closedloop_sample(cl, resampled_ps, zs)
end


function ibis_marginal_closedloop_sample!(
    cl::IBISClosedLoop,
    ps::AbstractArray{Float64,3},
    ws::AbstractMatrix{Float64},
    zs::AbstractMatrix{Float64},
    zns::AbstractMatrix{Float64},
)
    param_dim, _, nb_samples = size(ps)

    rvs = rand(nb_samples)
    resampled_ps = zeros(param_dim, nb_samples)
    for n in 1:nb_samples
        idx = categorical(rvs[n], ws[:, n])
        resampled_ps[:, n] = ps[:, idx, n]
    end

    return ibis_conditional_closedloop_sample!(cl, resampled_ps, zs, zns)
end


function ibis_conditional_closedloop_logpdf(
    cl::IBISClosedLoop,
    p::AbstractMatrix{Float64},
    z::AbstractMatrix{Float64},
    zn::AbstractMatrix{Float64},
)
    x = z[begin:cl.dyn.xdim, :]
    xn = zn[begin:cl.dyn.xdim, :]
    u = zn[cl.dyn.xdim+1:end, :]

    ll = policy_logpdf(cl.ctl, z, u)
    ll += map(eachcol(p), eachcol(x), eachcol(u), eachcol(xn)) do _p, _x, _u, _xn
        ibis_conditional_dynamics_logpdf(cl.dyn, _p, _x, _u, _xn)
    end
    return ll
end

Flux.@functor IBISClosedLoop


struct RaoBlackwellClosedLoop{
    S<:RaoBlackwellDynamics,
    T<:StochasticPolicy
}
    dyn::S
    ctl::T
end


function rao_blackwell_marginal_closedloop_mean(
    cl::RaoBlackwellClosedLoop,
    q::Gaussian,
    z::AbstractVector{Float64},
)
    x = z[begin:cl.dyn.xdim]
    u = policy_mean(cl.ctl, z)
    xn = rao_blackwell_marginal_dynamics_mean(cl.dyn, q, x, u)
    return vcat(xn, u)
end


function rao_blackwell_marginal_closedloop_sample(
    cl::RaoBlackwellClosedLoop,
    qs::AbstractVector{Gaussian},
    zs::AbstractMatrix{Float64},
)
    xs = zs[begin:cl.dyn.xdim, :]
    us = policy_sample(cl.ctl, zs)

    _sample_fn = (q, x, u) -> rao_blackwell_marginal_dynamics_sample(cl.dyn, q, x, u)
    xns = reduce(hcat, map(_sample_fn, qs, eachcol(xs), eachcol(us)))
    return vcat(xns, us)
end


function rao_blackwell_marginal_closedloop_logpdf(
    cl::RaoBlackwellClosedLoop,
    q::AbstractVector{Gaussian},
    z::AbstractMatrix{Float64},
    zn::AbstractMatrix{Float64},
)
    x = z[begin:cl.dyn.xdim, :]
    xn = zn[begin:cl.dyn.xdim, :]
    u = zn[cl.dyn.xdim+1:end, :]

    ll = policy_logpdf(cl.ctl, z, u)
    ll += map(q, eachcol(x), eachcol(u), eachcol(xn)) do _q, _x, _u, _xn
        rao_blackwell_marginal_dynamics_logpdf(cl.dyn, _q, _x, _u, _xn)
    end
    return ll
end


function rao_blackwell_marginal_closedloop_sample_and_logpdf(
    cl::RaoBlackwellClosedLoop,
    qs::AbstractVector{Gaussian},
    zs::AbstractMatrix{Float64},
)
    xs = zs[begin:cl.dyn.xdim, :]
    us = policy_sample(cl.ctl, zs)

    _sample_logpdf_fn = (q, x, u) -> rao_blackwell_marginal_dynamics_sample_and_logpdf(cl.dyn, q, x, u)
    res = map(_sample_logpdf_fn, qs, eachcol(xs), eachcol(us))

    xns = reduce(hcat, first.(res))
    ws = reduce(vcat, last.(res))
    return vcat(xns, us), ws
end

Flux.@functor RaoBlackwellClosedLoop