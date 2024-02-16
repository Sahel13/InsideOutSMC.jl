using Random
using Distributions
using LinearAlgebra

using InsideOutSMC

using Plots

Random.seed!(321)


xdim = 2
udim = 1
pdim = 2

step_size = 0.05
init_state = [pi/2.0, 0.0, 0.0]

true_params = [1.0, 1.0]


function drift_fn!(
    p::AbstractVector{Float64},
    x::AbstractVector{Float64},
    u::AbstractVector{Float64},
    xn::AbstractVector{Float64},
)
    m, l = p
    g, d = 9.81, 1e-3

    xn[1] = x[2]
    xn[2] = (
        - 3.0 * g / (2.0 * l) * sin(x[1])
        + (only(u) - d * x[2]) * 3.0 / (m * l^2)
    )
end


function diffusion_fn(
    args::AbstractVector{Float64}...
)::Vector{Float64}
    return [0.0, 1e-1]
end


random_policy = UniformStochasticPolicy([2.5])

ibis_dynamics = IBISDynamics(
    xdim, udim,
    drift_fn!,
    diffusion_fn,
    step_size,
)

ibis_loop = IBISClosedLoop(
    ibis_dynamics,
    random_policy
)

nb_steps = 50
nb_trajectories = 100

trajectories = Array{Float64}(undef, xdim+udim, nb_steps+1, nb_trajectories)
trajectories[:, 1, :] .= init_state

for t = 1:nb_steps
    ibis_conditional_closedloop_sample!(
        ibis_loop,
        repeat(true_params, 1, nb_trajectories),
        view(trajectories, :, t, :),
        view(trajectories, :, t+1, :)
    )
end
# plot(trajectories[:, :, 1]')


function param_proposal(
    particles::AbstractMatrix{Float64},
    prop_stddev::Float64=0.1,
)::Matrix{Float64}
    log_particles = log.(particles)
    log_particles .+= prop_stddev .* randn(size(log_particles))
    return exp.(log_particles)
end


nb_particles = 512
param_prior = MvLogNormal(
    MvNormal(
        [2.5, 2.5],
        3.0 * Matrix{Float64}(I, pdim, pdim)
    )
)

nb_moves = 1

xdim = ibis_loop.dyn.xdim
udim = ibis_loop.dyn.udim

scratch = Array{Float64,3}(undef, xdim, nb_particles, nb_trajectories)
state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
param_struct = IBISParamStruct(param_prior, nb_steps, nb_particles, nb_trajectories, scratch)
state_struct.trajectories .= reshape(trajectories, xdim+udim, nb_steps + 1, nb_trajectories)

chunk_size = 10
ranges = partition(1:nb_trajectories, chunk_size)

trajectory_chunks = [view(trajectories, :, :, range) for range in ranges]
param_struct_views = [view_struct(param_struct, range) for range in ranges]

@time begin
    tasks = map(zip(trajectory_chunks, param_struct_views)) do (traj_chunk, struct_view)
        @spawn begin
            batch_ibis!(
                traj_chunk,
                ibis_loop,
                param_prior,
                param_proposal,
                nb_moves,
                struct_view
            );
        end
    end
    fetch.(tasks);
end

# @time batch_ibis!(
#     trajectories,
#     ibis_loop,
#     param_prior,
#     param_proposal,
#     nb_moves,
#     param_struct
# );

num_tests_passed = 0
for n = 1:nb_trajectories
    params = @view param_struct.particles[:, end, :, n]
    param_weights = @view param_struct.weights[end, :, n]
    ibis_mean = sum(repeat(param_weights, 1, 2)' .* params, dims=2)
    if norm(ibis_mean - true_params)^2 < 0.1
        global num_tests_passed += 1
    end
end

println("number of tests passed: ", num_tests_passed)