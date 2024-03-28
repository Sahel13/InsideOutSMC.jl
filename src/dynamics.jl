using Random
using LinearAlgebra

import DistributionsAD as Distributions

import Flux
import Bijectors


struct StochasticDynamics{
    T<:Function,
    S<:Function
}
    xdim::Int
    udim::Int
    drift_fn::T
    diffusion_fn::S
    step::Float64
end


function dynamics_mean(
    sd::StochasticDynamics,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)::Vector{Float64}
    h = sd.step
    dx = sd.drift_fn(x, u)
    return x + h .* dx
end


function dynamics_covar(
    sd::StochasticDynamics,
    args::AbstractVector{Float64}...
)::Matrix{Float64}
    h = sd.step
    L = sd.diffusion_fn(args...)
    return Diagonal(@. L^2 * h + 1e-8)
end


function dynamics_sample(
    sd::StochasticDynamics,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)::Vector{Float64}
    mean = dynamics_mean(sd, x, u)
    covar = dynamics_covar(sd, x, u)
    return rand(
        Distributions.TuringDenseMvNormal(mean, covar)
    )
end


function dynamics_logpdf(
    sd::StochasticDynamics,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)
    mean = dynamics_mean(sd, x, u)
    covar = dynamics_covar(sd, x, u)
    return Distributions.logpdf(
        Distributions.TuringDenseMvNormal(mean, covar), xn
    )
end


struct IBISDynamics{
    T<:Function,
    S<:Function
}
    xdim::Int
    udim::Int
    drift_fn::T
    diffusion_fn::S
    step::Float64
end


function ibis_conditional_dynamics_mean(
    id::IBISDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)::Vector{Float64}
    return x + id.step .* id.drift_fn(p, x, u)
end


function ibis_conditional_dynamics_mean!(
    id::IBISDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)
    id.drift_fn(p, x, u, xn)
    xn .= x + id.step .* xn
end


function ibis_conditional_dynamics_covar(
    id::IBISDynamics,
    args::AbstractVector{Float64}...
)::Matrix{Float64}
    h = id.step
    L = id.diffusion_fn(args...)
    return Diagonal(@. L^2 * h + 1e-8)
end


function ibis_conditional_dynamics_sample(
    id::IBISDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)::Vector{Float64}
    mean = ibis_conditional_dynamics_mean(id, p, x, u)
    covar = ibis_conditional_dynamics_covar(id, p, x, u)
    return rand(
        Distributions.TuringDenseMvNormal(mean, covar)
    )
end


function ibis_conditional_dynamics_sample(
    id::IBISDynamics,
    ps::AbstractMatrix{Float64},
    xs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
)::Matrix{Float64}
    return reduce(hcat,
        map(
            eachcol(ps),  # parameters
            eachcol(xs),  # state
            eachcol(us)   # action
        ) do p, x, u
            ibis_conditional_dynamics_sample(id, p, x, u)
        end
    )
end


function ibis_conditional_dynamics_sample!(
    id::IBISDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64}
)
    covar = ibis_conditional_dynamics_covar(id)  # Assumes parameter-independent noise
    dist = Distributions.TuringDenseMvNormal(zeros(id.xdim), covar)
    ibis_conditional_dynamics_mean!(id, p, x, u, xn)
    xn .= xn + rand(dist)
end


function ibis_conditional_dynamics_sample!(
    id::IBISDynamics,
    ps::AbstractMatrix{Float64},
    xs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
    xns::AbstractMatrix{Float64}
)
    covar = ibis_conditional_dynamics_covar(id)  # Assumes parameter-independent noise
    dist = Distributions.TuringDenseMvNormal(zeros(id.xdim), covar)
    map(
        eachcol(ps),   # parameter
        eachcol(xs),   # state
        eachcol(us),   # action
        eachcol(xns)   # next state
    ) do p, x, u, xn
        ibis_conditional_dynamics_mean!(id, p, x, u, xn)
        xn .= xn + rand(dist)
    end
end


function ibis_conditional_dynamics_logpdf(
    id::IBISDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)
    mean = ibis_conditional_dynamics_mean(id, p, x, u)
    covar = ibis_conditional_dynamics_covar(id, p, x, u)
    return Distributions.logpdf(
        Distributions.TuringDenseMvNormal(mean, covar), xn
    )
end


function ibis_conditional_dynamics_logpdf(
    id::IBISDynamics,
    ps::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
    sc::AbstractMatrix{Float64}
)
    _, nb_particles = size(ps)
    covar = ibis_conditional_dynamics_covar(id)  # Assumes parameter-independent noise
    dist = Distributions.TuringDenseMvNormal(zeros(id.xdim), covar)
    @inbounds @views for m in 1:nb_particles
        ibis_conditional_dynamics_mean!(id, ps[:, m], x, u, sc[:, m])
        sc[:, m] .= sc[:, m] - xn
    end
    return Distributions.logpdf(dist, sc)
end


function ibis_marginal_dynamics_logpdf(
    id::IBISDynamics,
    ps::AbstractMatrix{Float64},
    lws::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)
    lls = map(eachcol(ps)) do p
        ibis_conditional_dynamics_logpdf(id, p, x, u, xn)
    end
    return logsumexp(lls + lws) - logsumexp(lws)
end


struct RaoBlackwellDynamics{
    T<:Function,
    S<:Function
}
    xdim::Int
    udim::Int
    feature_fn::T
    diffusion_fn::S
    step::Float64
end


function rao_blackwell_conditional_dynamics_mean(
    bd::RaoBlackwellDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)::Vector{Float64}
    feats = bd.feature_fn(x, u)

    h = bd.step
    A = [1.0 h; 0.0 1.0]
    B = [0.0; h] .* feats'
    return A * x + B * p
end


function rao_blackwell_conditional_dynamics_mean!(
    bd::RaoBlackwellDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)
    feats = bd.feature_fn(x, u)

    h = bd.step
    A = [1.0 h; 0.0 1.0]
    B = [0.0; h] .* feats'
    xn .= A * x + B * p
end


function rao_blackwell_conditional_dynamics_mean!(
    bd::RaoBlackwellDynamics,
    ps::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xns::AbstractMatrix{Float64},
)
    feats = bd.feature_fn(x, u)

    h = bd.step
    A = [1.0 h; 0.0 1.0]
    B = [0.0; h] .* feats'

    _, nb_samples = size(ps)
    @inbounds @views for m in 1:nb_samples
        xns[:, m] = A * x + B * ps[:, m]
    end
end


function rao_blackwell_conditional_dynamics_covar(
    bd::RaoBlackwellDynamics,
    args::AbstractVector{Float64}...
)::Matrix{Float64}
    h = bd.step
    L = bd.diffusion_fn(args...)
    return Diagonal(@. L^2 * h + 1e-8)
end


function rao_blackwell_conditional_dynamics_sample(
    bd::RaoBlackwellDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)::Vector{Float64}
    mean = rao_blackwell_conditional_dynamics_mean(bd, p, x, u)
    covar = rao_blackwell_conditional_dynamics_covar(bd)
    return rand(
        Distributions.TuringDenseMvNormal(mean, covar)
    )
end


function rao_blackwell_conditional_dynamics_sample(
    bd::RaoBlackwellDynamics,
    ps::AbstractMatrix{Float64},
    xs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
)::Matrix{Float64}
    return reduce(hcat,
        map(
            eachcol(ps),  # parameters
            eachcol(xs),  # state
            eachcol(us)   # action
        ) do p, x, u
            rao_blackwell_conditional_dynamics_sample(bd, p, x, u)
        end
    )
end


function rao_blackwell_conditional_dynamics_sample!(
    bd::RaoBlackwellDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64}
)
    covar = rao_blackwell_conditional_dynamics_covar(bd)
    dist = Distributions.TuringDenseMvNormal(zeros(bd.xdim), covar)
    rao_blackwell_conditional_dynamics_mean!(bd, p, x, u, xn)
    xn .= xn + rand(dist)
end


function rao_blackwell_conditional_dynamics_sample!(
    bd::RaoBlackwellDynamics,
    ps::AbstractMatrix{Float64},
    xs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
    xns::AbstractMatrix{Float64}
)
    covar = rao_blackwell_conditional_dynamics_covar(bd)
    dist = Distributions.TuringDenseMvNormal(zeros(bd.xdim), covar)
    map(
        eachcol(ps),   # parameter
        eachcol(xs),   # state
        eachcol(us),   # action
        eachcol(xns)   # next state
    ) do p, x, u, xn
        rao_blackwell_conditional_dynamics_mean!(bd, p, x, u, xn)
        xn .= xn + rand(dist)
    end
end


function rao_blackwell_conditional_dynamics_logpdf(
    bd::RaoBlackwellDynamics,
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)
    mean = rao_blackwell_conditional_dynamics_mean(bd, p, x, u)
    covar = rao_blackwell_conditional_dynamics_covar(bd)
    return Distributions.logpdf(
        Distributions.TuringDenseMvNormal(mean, covar), xn
    )
end


function rao_blackwell_conditional_dynamics_logpdf(
    bd::RaoBlackwellDynamics,
    ps::AbstractMatrix{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
    sc::AbstractMatrix{Float64}
)
    covar = rao_blackwell_conditional_dynamics_covar(bd)
    dist = Distributions.TuringDenseMvNormal(zeros(bd.xdim), covar)
    rao_blackwell_conditional_dynamics_mean!(bd, ps, x, u, sc)
    return Distributions.logpdf(dist, sc .- xn)
end


function rao_blackwell_marginal_dynamics_mean(
    bd::RaoBlackwellDynamics,
    q::Gaussian,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)::Vector{Float64}
    return rao_blackwell_conditional_dynamics_mean(bd, q.mean, x, u)
end


function rao_blackwell_marginal_dynamics_covar(
    bd::RaoBlackwellDynamics,
    q::Gaussian,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)::Matrix{Float64}
    cond_covar = rao_blackwell_conditional_dynamics_covar(bd)

    h = bd.step
    B = [0.0; h] .* bd.feature_fn(x, u)'
    marg_covar = cond_covar + (B * q.covar) * B'
    return Symmetric(marg_covar)
end


function rao_blackwell_marginal_dynamics_sample(
    bd::RaoBlackwellDynamics,
    q::Gaussian,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)::Vector{Float64}
    mean = rao_blackwell_marginal_dynamics_mean(bd, q, x, u)
    covar = rao_blackwell_marginal_dynamics_covar(bd, q, x, u)
    return rand(
        Distributions.TuringDenseMvNormal(mean, covar)
    )
end


function rao_blackwell_marginal_dynamics_logpdf(
    bd::RaoBlackwellDynamics,
    q::Gaussian,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)
    mean = rao_blackwell_marginal_dynamics_mean(bd, q, x, u)
    covar = rao_blackwell_marginal_dynamics_covar(bd, q, x, u)
    return Distributions.logpdf(
        Distributions.TuringDenseMvNormal(mean, covar), xn
    )
end


function rao_blackwell_marginal_dynamics_sample_and_logpdf(
    bd::RaoBlackwellDynamics,
    q::Gaussian,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
)
    cond_covar = rao_blackwell_conditional_dynamics_covar(bd)

    h = bd.step
    A = [1.0 h; 0.0 1.0]
    B = [0.0; h] .* bd.feature_fn(x, u)'

    marg_mean = A * x + B * q.mean
    marg_covar = Symmetric(cond_covar + (B * q.covar) * B')

    dist = Distributions.TuringDenseMvNormal(marg_mean, marg_covar)
    xn = Distributions.rand(dist)
    lp = Distributions.logpdf(dist, xn)
    return xn, lp
end


function rao_blackwell_dynamics_update(
    bd::RaoBlackwellDynamics,
    q::Gaussian,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)
    cond_covar = rao_blackwell_conditional_dynamics_covar(bd)

    h = bd.step
    A = [1.0 h; 0.0 1.0]
    B = [0.0; h] .* bd.feature_fn(x, u)'

    inv_S = inv(cond_covar + (B * q.covar) * B')
    K = (q.covar * B') * inv_S

    post_mean = q.mean + K * (xn - A * x - B * q.mean)
    post_covar = (I - K * B) * q.covar
    return Gaussian(post_mean, Symmetric(post_covar))
end


function rao_blackwell_dynamics_update!(
    bd::RaoBlackwellDynamics,
    q::Gaussian,
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
    qn::Gaussian
)
    cond_covar = rao_blackwell_conditional_dynamics_covar(bd)

    h = bd.step
    A = [1.0 h; 0.0 1.0]
    B = [0.0; h] .* bd.feature_fn(x, u)'

    inv_S = inv(cond_covar + (B * q.covar) * B')
    K = (q.covar * B') * inv_S

    qn.mean .= q.mean + K * (xn - A * x - B * q.mean)
    qn.covar .= Symmetric((I - K * B) * q.covar)
end
