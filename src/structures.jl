using Random
using Distributions


mutable struct StateStruct
    state_dim::Int
    nb_steps::Int
    nb_trajectories::Int
    trajectories::Array{Float64,3}
    weights::Vector{Float64}
    log_weights::Vector{Float64}
    log_weights_increment::Vector{Float64}
    log_evidence::Float64
    cumulative_return::Vector{Float64}
    resampled_idx::Vector{Int}
    rvs::Vector{Float64}
end


function StateStruct(
    state_dim::Int,
    nb_steps::Int,
    nb_trajectories::Int,
    trajectories::Array{Float64,3}
)
    weights = fill(1 / nb_trajectories, nb_trajectories)
    log_weights = zeros(nb_trajectories)
    log_weights_increment = zeros(nb_trajectories)
    log_evidence = 0.0
    cumulative_return = zeros(nb_trajectories)
    resampled_idx = Vector{Int}(undef, nb_trajectories)
    rvs = Vector{Float64}(undef, nb_trajectories)

    return StateStruct(
        state_dim,
        nb_steps,
        nb_trajectories,
        trajectories,
        weights,
        log_weights,
        log_weights_increment,
        log_evidence,
        cumulative_return,
        resampled_idx,
        rvs
    )
end


function StateStruct(
    init_state::Vector{Float64},
    nb_steps::Int,
    nb_trajectories::Int
)
    state_dim = length(init_state)
    trajectories = Array{Float64}(undef, state_dim, nb_steps + 1, nb_trajectories)
    trajectories[:, 1, :] .= init_state
    return StateStruct(state_dim, nb_steps, nb_trajectories, trajectories)
end


struct Gaussian
    mean::Vector{Float64}
    covar::Matrix{Float64}
end


struct RaoBlackwellParamStruct
    distributions::Matrix{Gaussian}
end


function RaoBlackwellParamStruct(
    prior::Gaussian,
    nb_steps::Int,
    nb_trajectories::Int,
)
    dists = Matrix{Gaussian}(undef, nb_steps + 1, nb_trajectories)
    for t in 1:nb_steps + 1
        for n in 1:nb_trajectories
            dists[t, n] = deepcopy(prior)
        end
    end
    return RaoBlackwellParamStruct(dists)
end


struct RaoBlackwellReference
    trajectory::Matrix{Float64}
    distributions::Vector{Gaussian}
end


struct IBISParamStruct{
    A<:AbstractArray{Float64},
    B<:AbstractArray{Float64},
    C<:AbstractArray{Float64},
    D<:AbstractArray{Float64},
    E<:AbstractArray{Int},
    F<:AbstractArray{Float64},
    G<:AbstractArray{Float64},
}
    param_dim::Int
    nb_particles::Int
    particles::A
    weights::B
    log_weights::C
    log_likelihoods::D
    resampled_idx::E
    rvs::F
    scratch::G
end


function IBISParamStruct(
    param_prior::MultivariateDistribution,
    nb_steps::Int,
    nb_particles::Int,
    nb_trajectories::Int,
    scratch::Array{Float64,3},
)
    param_dim = length(param_prior)
    particles = Array{Float64,4}(undef, param_dim, nb_steps + 1, nb_particles, nb_trajectories)
    weights = fill(1 / nb_particles, nb_steps + 1, nb_particles, nb_trajectories)
    log_weights = zeros(nb_steps + 1, nb_particles, nb_trajectories)
    log_likelihoods = Array{Float64,3}(undef, nb_steps + 1, nb_particles, nb_trajectories)
    resampled_idx = Matrix{Int}(undef, nb_particles, nb_trajectories)
    rvs = Matrix{Float64}(undef, nb_particles, nb_trajectories)

    param_matrix = rand(param_prior, nb_particles, nb_trajectories)
    for m = 1:nb_particles
        for n = 1:nb_trajectories
            particles[:, 1, m, n] = param_matrix[m, n]
        end
        logpdf!(view(log_likelihoods, 1, m, :), param_prior, particles[:, 1, m, :])
    end

    return IBISParamStruct(
        param_dim,
        nb_particles,
        particles,
        weights,
        log_weights,
        log_likelihoods,
        resampled_idx,
        rvs,
        scratch,
    )
end


function view_struct(
    param_struct::IBISParamStruct,
    range::UnitRange{Int},
)
    return IBISParamStruct(
        param_struct.param_dim,
        param_struct.nb_particles,
        view(param_struct.particles, :, :, :, range),
        view(param_struct.weights, :, :, range),
        view(param_struct.log_weights, :, :, range),
        view(param_struct.log_likelihoods, :, :, range),
        view(param_struct.resampled_idx, :, range),
        view(param_struct.rvs, :, range),
        view(param_struct.scratch, :, :, range),
    )
end


function view_struct(
    param_struct::IBISParamStruct,
    idx::Int,
)
    return IBISParamStruct(
        param_struct.param_dim,
        param_struct.nb_particles,
        view(param_struct.particles, :, :, :, idx),
        view(param_struct.weights, :, :, idx),
        view(param_struct.log_weights, :, :, idx),
        view(param_struct.log_likelihoods, :, :, idx),
        view(param_struct.resampled_idx, :, idx),
        view(param_struct.rvs, :, idx),
        view(param_struct.scratch, :, :, idx),
    )
end


struct IBISReference
    trajectory::Matrix{Float64}
    particles::Array{Float64,3}
    weights::Matrix{Float64}
    log_weights::Matrix{Float64}
    log_likelihoods::Matrix{Float64}
end