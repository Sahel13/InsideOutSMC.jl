using Random
using StatsFuns
using LinearAlgebra

import DistributionsAD as Distributions

import Zygote
import Flux
import Functors
import Bijectors

using JLD2

using InsideOutSMC: StochasticPolicy
using InsideOutSMC: Tanh


log_std_min = -5.0 * ones(1)
log_std_max = 2.0 * ones(1)


struct StatefulStochasticPolicy <: StochasticPolicy
    transform::Function
    encoder_func::Flux.Chain
    mean_func::Flux.Chain
    log_std_func::Flux.Chain
    bijector::Bijectors.ComposedFunction
end


function policy_sample(
    sp::StatefulStochasticPolicy,
    zs::AbstractMatrix{Float64}
)
    feats = reduce(hcat, map(sp.transform, eachcol(zs)))
    hiddens = sp.encoder_func(feats)
    ms = sp.mean_func(hiddens)

    log_std = Flux.tanh.(sp.log_std_func(hiddens))
    log_std = @. log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
    ps = exp.(log_std)

    sample = (m, p) -> rand(
        Bijectors.transformed(
            Distributions.TuringMvNormal(m, Diagonal(p.^2)),
            sp.bijector
        )
    )
    return reduce(hcat, map(sample, eachcol(ms), eachcol(ps)))
end


function policy_mean(
    sp::StatefulStochasticPolicy,
    z::AbstractVector{Float64}
)
    feat = sp.transform(z)
    hidden = sp.encoder_func(feat)
    mean = sp.mean_func(hidden)
    return sp.bijector(mean)
end


function policy_logpdf(
    sp::StatefulStochasticPolicy,
    zs::AbstractMatrix{Float64},
    us::AbstractMatrix{Float64},
)
    feats = reduce(hcat, map(sp.transform, eachcol(zs)))
    hiddens = sp.encoder_func(feats)
    ms = sp.mean_func(hiddens)

    log_std = Flux.tanh.(sp.log_std_func(hiddens))
    log_std = @. log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)
    ps = exp.(log_std)

    lls = map(eachcol(us), eachcol(ms), eachcol(ps)) do u, m, p
        Distributions.logpdf(
            Bijectors.transformed(
                Distributions.TuringMvNormal(m, Diagonal(p.^2)),
                sp.bijector
            ),
            u
        )
    end
    return reduce(vcat, lls)
end

Flux.@functor StatefulStochasticPolicy


input_dim = 3
output_dim = 1
recur_size = 64
dense_size = 256

function transform(
    z::AbstractVector{Float64}
)
    return vcat(
        sin(z[1]),
        cos(z[1]),
        z[2],
    )
end

encoder_func = Flux.f64(
    Flux.Chain(
        Flux.Dense(input_dim, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, recur_size, Flux.relu),
        Flux.LSTM(recur_size, recur_size),
        Flux.LSTM(recur_size, recur_size),
    ),
)

mean_func = Flux.f64(
    Flux.Chain(
        Flux.Dense(recur_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, output_dim),
    ),
)

log_std_func = Flux.f64(
    Flux.Chain(
        Flux.Dense(recur_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, dense_size, Flux.relu),
        Flux.Dense(dense_size, output_dim),
    ),
)

scale = 2.5
shift = 0.0
bijector = (Bijectors.Shift(shift) ∘ Bijectors.Scale(scale) ∘ Tanh())

policy = StatefulStochasticPolicy(
    transform,
    encoder_func,
    mean_func,
    log_std_func,
    bijector,
)

samples = load_object("samples.jld2")

function lower_bound(samples, policy)
    Flux.reset!(policy)

    _, nb_steps, _ = size(samples)
    _logpdf = (x, u) -> policy_logpdf(policy, x, u)

    ll = 0.0
    for t = 2:nb_steps
        state = samples[1:2, t-1, :]
        action = samples[3:end, t, :]
        ll += sum(_logpdf(state, action))
    end
    return ll
end

opt_state = Flux.setup(Flux.Optimise.Adam(5e-4), policy)

# maximization step
batcher = Flux.DataLoader(
    samples,
    batchsize=64,
    shuffle=true
)

for _samples in batcher
    global policy
    global opt_state

    lb, grad = Zygote.withgradient(policy) do f
        -1.0 * lower_bound(_samples, f)
    end
    _, policy = Flux.update!(opt_state, policy, grad[1])
    println("lb: ", lb)
end
