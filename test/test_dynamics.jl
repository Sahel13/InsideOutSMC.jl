using Random
using Distributions
using LinearAlgebra

using InsideOutSMC
using InsideOutSMC: Gaussian
using InsideOutSMC: RaoBlackwellDynamics
using InsideOutSMC: rao_blackwell_dynamics_sample
using InsideOutSMC: rao_blackwell_dynamics_update


xdim = 2
udim = 1
step = 0.05

init_state = [0.0, 0.0]
log_std = log(sqrt(1e-4)) * ones(2)


function featurize(x, u)
    q, q_dot = x
    return [-sin(q) -q_dot u]
end

l, m = 1.0, 1.0
g, d = 9.81, 1e-3

mu = [3.0 * g / (2.0 * l), 3.0 * d / (m * l^2), 3.0 / (m * l^2)]
sigma = 1e-2 * Matrix{Float64}(I, 3, 3)

prior = Gaussian(mu, sigma)


xs = rand(2, 10)
us = rand(1, 10)

feats = reduce(vcat, map(featurize, eachcol(xs), eachcol(us)))'

dynamics = RaoBlackwellDynamics(
    xdim, udim,
    step, log_std,
    featurize
)

x = rand(2)
u = rand(1)

xn = rao_blackwell_dynamics_sample(dynamics, prior, x, u)

posterior = rao_blackwell_dynamics_update(
    dynamics, prior, x, u, xn
)
