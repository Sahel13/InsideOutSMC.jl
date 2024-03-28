using Random
using Distributions


function batch_ibis!(
    trajectories::AbstractArray{Float64,3},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct
) where {T<:Function}

    _, _, nb_trajectories = size(trajectories)
    for traj_idx = 1:nb_trajectories
        ibis!(
            view(trajectories, :, :, traj_idx),
            dynamics,
            param_prior,
            param_proposal,
            nb_moves,
            view_struct(param_struct, traj_idx),
        )
    end
end


function ibis!(
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct
) where {T<:Function}

    _, nb_steps_p_1 = size(trajectory)
    nb_steps = nb_steps_p_1 - 1

    for time_idx = 1:nb_steps
        ibis_step!(
            time_idx,
            trajectory,
            dynamics,
            param_prior,
            param_proposal,
            nb_moves,
            param_struct,
        )
    end
end


function batch_ibis_step!(
    time_idx::Int,
    trajectories::AbstractArray{Float64,3},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct,
) where {T<:Function}

    _, _, nb_trajectories = size(trajectories)
    for traj_idx = 1:nb_trajectories
        ibis_step!(
            time_idx,
            view(trajectories, :, :, traj_idx),
            dynamics,
            param_prior,
            param_proposal,
            nb_moves,
            view_struct(param_struct, traj_idx),
        )
    end
end


function ibis_step!(
    time_idx::Int,
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct,
) where {T<:Function}

    # 1. Reweight
    reweight_params!(
        time_idx,
        trajectory,
        dynamics,
        param_struct,
    )

    # 2. Resample-move if necessary
    weights = @view param_struct.weights[time_idx+1, :]
    if effective_sample_size(weights) < 0.75 * param_struct.nb_particles
        resample_params!(
            time_idx,
            param_struct
        )
        move!(
            time_idx,
            trajectory,
            dynamics,
            param_prior,
            param_proposal,
            nb_moves,
            param_struct,
        )
    end
end


function reweight_params!(
    time_idx::Int,
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_struct::IBISParamStruct,
)
    xdim = dynamics.xdim

    # Get the log weight increments
    log_weight_increments = ibis_conditional_dynamics_logpdf(
        dynamics,
        param_struct.particles[:, time_idx, :],
        trajectory[begin:xdim, time_idx],  # state
        trajectory[xdim+1:end, time_idx+1],    # action
        trajectory[begin:xdim, time_idx+1],    # next_state
        param_struct.scratch
    )  # Assumes parameter-independent noise

    # Copy over particles and weights to next time step
    param_struct.particles[:, time_idx+1, :] = param_struct.particles[:, time_idx, :]
    param_struct.weights[time_idx+1, :] = param_struct.weights[time_idx, :]
    param_struct.log_weights[time_idx+1, :] = param_struct.log_weights[time_idx, :]
    param_struct.log_likelihoods[time_idx+1, :] = param_struct.log_likelihoods[time_idx, :]

    # Update log_weights and log_likelihoods
    param_struct.log_weights[time_idx+1, :] += log_weight_increments
    param_struct.log_likelihoods[time_idx+1, :] += log_weight_increments

    # Normalize the updated weights
    normalize_weights!(time_idx+1, param_struct)
end


function resample_params!(
    time_idx::Int,
    param_struct::IBISParamStruct,
)
    systematic_resampling!(time_idx+1, param_struct)

    param_struct.weights[time_idx+1, :] .= 1 / param_struct.nb_particles
    param_struct.log_weights[time_idx+1, :] .= 0.0
    param_struct.particles[:, time_idx+1, :] .= param_struct.particles[:, time_idx+1, param_struct.resampled_idx]
    param_struct.log_likelihoods[time_idx+1, :] .= param_struct.log_likelihoods[time_idx+1, param_struct.resampled_idx]
end


function move!(
    time_idx::Int,
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_moves::Int,
    param_struct::IBISParamStruct,
) where {T<:Function}
    for j = 1:nb_moves
        kernel!(
            time_idx,
            trajectory,
            dynamics,
            param_prior,
            param_proposal,
            param_struct,
        )
    end
end


function kernel!(
    time_idx::Int,
    trajectory::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    param_struct::IBISParamStruct,
) where {T<:Function}

    history = @view trajectory[:, 1:time_idx+1]
    particles = @view param_struct.particles[:, time_idx+1, :]
    weights = @view param_struct.weights[time_idx+1, :]
    prop_particles = param_proposal(particles, weights)

    # populate uniforms
    rand!(param_struct.rvs)

    prop_log_likelihood = accumulate_likelihood(
        history,
        prop_particles,
        dynamics,
        param_prior,
        param_struct.scratch
    )

    acceptance_rate = 0.0
    @views @inbounds for m = 1:param_struct.nb_particles
        log_ratio = prop_log_likelihood[m] - param_struct.log_likelihoods[time_idx+1, m]
        if log(param_struct.rvs[m]) < log_ratio
            param_struct.particles[:, time_idx+1, m] = prop_particles[:, m]
            param_struct.log_likelihoods[time_idx+1, m] = prop_log_likelihood[m]
            acceptance_rate += 1.0
        end
    end
    # println(acceptance_rate / param_struct.nb_particles)
end


function accumulate_likelihood(
    history::AbstractMatrix{Float64},
    particles::AbstractMatrix{Float64},
    dynamics::IBISDynamics,
    param_prior::MultivariateDistribution,
    scratch::AbstractMatrix{Float64},
)
    lls = logpdf(param_prior, particles)

    xdim = dynamics.xdim
    _, nb_steps_p_1 = size(history)

    @views @inbounds for t in 1:nb_steps_p_1 - 1
        lls += ibis_conditional_dynamics_logpdf(
            dynamics,
            particles,
            history[begin:xdim, t],    # state
            history[xdim+1:end, t+1],  # action
            history[begin:xdim, t+1],  # next_state
            scratch
        )  # Assumes parameter-independent noise
    end
    return lls
end
