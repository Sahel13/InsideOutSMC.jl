using Random
using StatsFuns
using LinearAlgebra

import DistributionsAD as Distributions

import Zygote
import Flux
import Functors
import Bijectors


struct Tanh <: Bijectors.Bijector end

Bijectors.transform(b::Tanh, x) = Flux.anh(x)
Bijectors.transform(b::Tanh, x::AbstractArray) = @. Flux.tanh(x)

Bijectors.transform(ib::Bijectors.Inverse{<:Tanh}, y) = Flux.atanh(y)
Bijectors.transform(ib::Bijectors.Inverse{<:Tanh}, y::AbstractArray) = @. Flux.atanh(y)


function _logabsdetjac(x::Real)
    return 2 * (StatsFuns.logtwo - x - softplus(-2 * x))
end

function _logabsdetjac(x::AbstractVector)
    return @. 2 * (StatsFuns.logtwo - x - softplus(-2 * x))
end

function with_logabsdet_jacobian(b::Tanh, x::Real)
    y = tanh(x)
    ldj = _logabsdetjac(x)
    return (result=y, logabsdetjac=ldj)
end

function with_logabsdet_jacobian(b::Tanh, x::AbstractVector)
    y = @. tanh(x)
    ldj = _logabsdetjac(x)
    return (result=y, logabsdetjac=first(ldj))
end

function with_logabsdet_jacobian(b::Tanh, x::AbstractMatrix)
    y = @. tanh(x)
    ldj = reduce(vcat, map(_logabsdetjac, eachcol(x)))
    return (result=y, logabsdetjac=ldj)
end

Bijectors.logabsdetjac(b::Tanh, x) = last(with_logabsdet_jacobian(b, x))


function transform(
    zs::AbstractMatrix{Float64}
)
    return hcat(
        sin.(zs[1, :]),
        cos.(zs[1, :]),
        zs[2, :],
    )'
end


mutable struct StochasticPolicy
    network::Flux.Chain
    bijector::Bijectors.ComposedFunction
    log_std::Array{Float64}
end


StochasticPolicy(
    dense_size::Int,
    log_std::Array{Float64},
    scale::Float64,
    shift::Float64,
) = StochasticPolicy(
    Flux.f64(
        Flux.Chain(
            Flux.Dense(3, dense_size, Flux.relu),
            Flux.Dense(dense_size, dense_size, Flux.relu),
            Flux.Dense(dense_size, 1),
        ),
    ),
    (Bijectors.Shift(shift) ∘ Bijectors.Scale(scale) ∘ Tanh()),
    log_std,
)


function (sp::StochasticPolicy)(zs::AbstractMatrix{Float64})
    ms = sp.network(transform(zs))
    std = exp.(sp.log_std)

    sample = m -> rand(
        Bijectors.transformed(
            Distributions.TuringMvNormal(m, std ^ 2 * I),
            sp.bijector
        )
    )
    return reduce(hcat, map(sample, eachcol(ms)))
end


function policy_mean(
    sp::StochasticPolicy,
    zs::AbstractMatrix{Float64}
)
    us = sp.network(transform(zs))
    return sp.bijector(us)
end


function policy_logpdf(
    sp::StochasticPolicy,
    zs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
)
    ms = sp.network(transform(zs))
    std = exp.(sp.log_std)

    ll = 0.0
    for (u, m) in zip(eachcol(us), eachcol(ms))
        ll += Distributions.logpdf(
            Bijectors.transformed(
                Distributions.TuringMvNormal(m, Diagonal(std.^2)),
                sp.bijector
            ),
            u
        )
    end
    return ll
end

Flux.@functor StochasticPolicy

dense_size = 256
log_std = log(sqrt(2.5)) * ones(1)

policy = StochasticPolicy(
    dense_size,
    log_std, 5.0, 0.0,
)

xs = rand(2, 100)
us = rand(1, 100)
lp = policy_logpdf(policy, xs, us)

opt_state = Flux.setup(Flux.Optimise.Adam(1e-3), policy)

lb, grad = Zygote.withgradient(policy) do f
    policy_logpdf(f, xs, us)
end
_, policy = Flux.update!(opt_state, policy, grad[1]);
