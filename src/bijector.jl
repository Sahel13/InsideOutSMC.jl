using Random
using LinearAlgebra
using LogExpFunctions: softplus

import Bijectors


function _stable_atanh(y::Real)
    _y = clamp(y, -0.99995, 0.99995)
    return atanh(_y)
end

struct Tanh <: Bijectors.Bijector end


Bijectors.transform(b::Tanh, x) = tanh(x)
Bijectors.transform(b::Tanh, x::AbstractArray) = @. tanh(x)

Bijectors.transform(ib::Bijectors.Inverse{<:Tanh}, y) = _stable_atanh(y)
Bijectors.transform(ib::Bijectors.Inverse{<:Tanh}, y::AbstractArray) = @. _stable_atanh(y)


function _logabsdetjac(x::Real)
    return 2.0 * (log(2.0) - x - softplus(-2.0 * x))
end

function _logabsdetjac(x::AbstractVector)
    return @. 2.0 * (broadcast(-, log(2.0), x) - softplus(-2.0 * x))
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
