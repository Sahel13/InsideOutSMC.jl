using Random
using LinearAlgebra

import DistributionsAD as Distributions

import Flux
import Bijectors


abstract type StochasticPolicy end


function policy_mean(
    sp::StochasticPolicy,
    z::AbstractVector{Float64},
)
end


function policy_sample(
    sp::StochasticPolicy,
    zs::AbstractMatrix{Float64},
)::Matrix{Float64}
end


function policy_logpdf(
    sp::StochasticPolicy,
    zs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
)
end


abstract type StatefulStochasticPolicy<:StochasticPolicy end

function policy_mean(
    sp::StatefulStochasticPolicy,
    z::AbstractVector{Float64},
)
end


function policy_sample(
    sp::StatefulStochasticPolicy,
    zs::AbstractMatrix{Float64},
)::Matrix{Float64}
end


function policy_logpdf(
    sp::StatefulStochasticPolicy,
    zs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
)
end


struct StatefulHomoschedasticPolicy{
    T<:Function,
    S<:Flux.Chain,
    R<:Flux.Chain,
    V<:Bijectors.ComposedFunction
}<:StatefulStochasticPolicy
    feature_fn::T
    encoder_fn::S
    mean_fn::R
    log_std::Vector{Float64}
    bijector::V
end


function policy_mean(
    sp::StatefulHomoschedasticPolicy,
    z::AbstractVector{Float64}
)
    feat = sp.feature_fn(z)
    hidden = sp.encoder_fn(feat)
    u = sp.mean_fn(hidden)
    return sp.bijector(u)
end


function policy_sample(
    sp::StatefulHomoschedasticPolicy,
    zs::AbstractMatrix{Float64}
)::Matrix{Float64}
    feats = reduce(hcat, map(sp.feature_fn, eachcol(zs)))
    hiddens = sp.encoder_fn(feats)
    ms = sp.mean_fn(hiddens)
    std = exp.(sp.log_std)

    sample = m -> rand(
        Bijectors.transformed(
            Distributions.TuringDenseMvNormal(m, Diagonal(std.^2)),
            sp.bijector
        )
    )
    return reduce(hcat, map(sample, eachcol(ms)))
end


function policy_logpdf(
    sp::StatefulHomoschedasticPolicy,
    zs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
)
    feats = reduce(hcat, map(sp.feature_fn, eachcol(zs)))
    hiddens = sp.encoder_fn(feats)
    ms = sp.mean_fn(hiddens)
    std = exp.(sp.log_std)

    lls = map(eachcol(us), eachcol(ms)) do u, m
        Distributions.logpdf(
            Bijectors.transformed(
                Distributions.TuringDenseMvNormal(m, Diagonal(std.^2)),
                sp.bijector
            ),
            u
        )
    end
    return reduce(vcat, lls)
end


function policy_entropy(
    sp::StatefulHomoschedasticPolicy,
)
    dim = length(sp.log_std)
    covar = Diagonal(exp.(sp.log_std).^2)
    return (
        0.5 * dim * log(2 * pi * exp(1))
        + 0.5 * logdet(covar)
    )
end

Flux.@functor StatefulHomoschedasticPolicy


struct StatefulHeteroschedasticPolicy{
    T<:Function,
    S<:Flux.Chain,
    R<:Flux.Chain,
    W<:Flux.Chain,
    V<:Bijectors.ComposedFunction
}<:StatefulStochasticPolicy
    feature_fn::T
    encoder_fn::S
    mean_fn::R
    log_std_fn::W
    bijector::V
end


function policy_mean(
    sp::StatefulHeteroschedasticPolicy,
    z::AbstractVector{Float64}
)
    feat = sp.feature_fn(z)
    hidden = sp.encoder_fn(feat)
    mean = sp.mean_fn(hidden)
    return sp.bijector(mean)
end


function policy_sample(
    sp::StatefulHeteroschedasticPolicy,
    zs::AbstractMatrix{Float64}
)::Matrix{Float64}
    log_std_min = -5.0 * ones(1)
    log_std_max = 2.0 * ones(1)

    feats = reduce(hcat, map(sp.feature_fn, eachcol(zs)))
    hiddens = sp.encoder_fn(feats)
    ms = sp.mean_fn(hiddens)

    log_std = Flux.tanh.(sp.log_std_fn(hiddens))
    log_std = @. log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
    ps = exp.(log_std)

    sample = (m, p) -> rand(
        Bijectors.transformed(
            Distributions.TuringDenseMvNormal(m, Diagonal(p.^2)),
            sp.bijector
        )
    )
    return reduce(hcat, map(sample, eachcol(ms), eachcol(ps)))
end


function policy_logpdf(
    sp::StatefulHeteroschedasticPolicy,
    zs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
)
    log_std_min = -5.0 * ones(1)
    log_std_max = 2.0 * ones(1)

    feats = reduce(hcat, map(sp.feature_fn, eachcol(zs)))
    hiddens = sp.encoder_fn(feats)
    ms = sp.mean_fn(hiddens)

    log_std = Flux.tanh.(sp.log_std_fn(hiddens))
    log_std = @. log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
    ps = exp.(log_std)

    lls = map(eachcol(us), eachcol(ms), eachcol(ps)) do u, m, p
        Distributions.logpdf(
            Bijectors.transformed(
                Distributions.TuringDenseMvNormal(m, Diagonal(p.^2)),
                sp.bijector
            ),
            u
        )
    end
    return reduce(vcat, lls)
end

Flux.@functor StatefulHeteroschedasticPolicy


struct UniformStochasticPolicy<:StochasticPolicy
    scale::Vector{Float64}
end


function policy_mean(
    sp::UniformStochasticPolicy,
    z::AbstractVector{Float64}
)
    dist = product_distribution(Uniform.(-sp.scale, sp.scale))
    return rand(dist)
end


function policy_sample(
    sp::UniformStochasticPolicy,
    zs::AbstractMatrix{Float64}
)::Matrix{Float64}
    dist = product_distribution(Uniform.(-sp.scale, sp.scale))
    return rand(dist, last(size(zs)))
end


function policy_logpdf(
    sp::UniformStochasticPolicy,
    zs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
)
    dist = product_distribution(Uniform.(-sp.scale, sp.scale))
    return logpdf(dist, us)
end


struct PRBSStochasticPolicy <: StochasticPolicy
    scale::Vector{Float64}
end


function policy_sample(
    sp::PRBSStochasticPolicy,
    zs::AbstractMatrix{Float64}
)
    us = rand((-sp.scale, sp.scale), last(size(zs)))
    return reduce(hcat, us)
end


function policy_mean(
    sp::PRBSStochasticPolicy,
    z::AbstractVector{Float64}
)
    return rand((-sp.scale, sp.scale))
end


struct MaxActionPolicy <: StochasticPolicy
    scale::Vector{Float64}
end


function policy_sample(
    sp::MaxActionPolicy,
    zs::AbstractMatrix{Float64}
)
    nb_samples = last(size(zs))
    return repeat(sp.scale, outer=(1, nb_samples))
end


function policy_mean(
    sp::MaxActionPolicy,
    z::AbstractVector{Float64}
)
    return sp.scale
end