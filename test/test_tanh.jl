using Random
using Distributions
using StatsFuns
using LinearAlgebra
using LogExpFunctions: softplus

import Bijectors


struct Tanh <: Bijectors.Bijector end

Bijectors.transform(b::Tanh, x) = tanh(x)
Bijectors.transform(b::Tanh, x::AbstractArray) = @. tanh(x)

Bijectors.transform(ib::Bijectors.Inverse{<:Tanh}, y) = atanh(y)
Bijectors.transform(ib::Bijectors.Inverse{<:Tanh}, y::AbstractArray) = @. atanh(y)


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



dist = Distributions.MvNormal(zeros(1), LinearAlgebra.I)
td = Bijectors.transformed(dist, Tanh())

Random.seed!(1337)
y = Random.rand(td)
ll = Distributions.logpdf(td, y)

Random.seed!(1337)
x = rand(dist)
x_fwd = Tanh()(x)
x_raw = atanh.(x_fwd)
logjac = z -> @. log(1.0 - z^2)
ll_fwd = Distributions.logpdf(dist, x_raw) - first(logjac(x_fwd))

display(ll - ll_fwd)


Random.seed!(1337)
y = Random.rand(td, 10)
ll = Distributions.logpdf(td, y)

Random.seed!(1337)
x = rand(dist, 10)
x_fwd = Tanh()(x)
x_raw = atanh.(x_fwd)
logjac = z -> @. log(1.0 - z^2)
ll_fwd = Distributions.logpdf(dist, x_raw) - reduce(vcat, map(logjac, eachcol(x_fwd)))

display(ll - ll_fwd)
