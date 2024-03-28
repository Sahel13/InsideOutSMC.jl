using Base.Threads: nthreads, @threads, @spawn
using Base.Iterators: partition


function smc(
    nb_steps::Int,
    nb_trajectories::Int,
    init_state::Vector{Float64},
    closedloop::ClosedLoop,
    reward_fn::Function,
    tempering::Float64,
)
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)

    for time_idx = 1:nb_steps
        smc_step!(
            time_idx,
            closedloop,
            reward_fn,
            tempering,
            state_struct,
        )
    end
    return state_struct
end


function csmc(
    nb_steps::Int,
    nb_trajectories::Int,
    init_state::Vector{Float64},
    closedloop::ClosedLoop,
    reward_fn::Function,
    tempering::Float64,
    reference::Matrix{Float64},
)
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)

    state_struct.trajectories[:, 1, 1] .= reference[:, 1]
    for time_idx = 1:nb_steps
        csmc_step!(
            time_idx,
            closedloop,
            reward_fn,
            tempering,
            reference,
            state_struct,
        )
    end
    return state_struct
end


function smc_with_ibis_marginal_dynamics(
    nb_steps::Int,
    nb_trajectories::Int,
    nb_particles::Int,
    init_state::Vector{Float64},
    closedloop::IBISClosedLoop,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_ibis_moves::Int,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    nb_threads::Int = 10,
) where {T<:Function}

    scratch = Array{Float64}(undef, closedloop.dyn.xdim, nb_particles, nb_trajectories)
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = IBISParamStruct(param_prior, nb_steps, nb_particles, nb_trajectories, scratch)

    chunk_size = round(Int, nb_trajectories / nb_threads)
    ranges = partition(1:nb_trajectories, chunk_size)

    trajectory_views = [view(state_struct.trajectories, :, :, range) for range in ranges]
    param_struct_views = [view_struct(param_struct, range) for range in ranges]

    for time_idx in 1:nb_steps
        smc_step_with_ibis_marginal_dynamics!(
            time_idx,
            closedloop,
            action_penalty,
            slew_rate_penalty,
            tempering,
            state_struct,
            param_struct,
        )

        tasks = map(zip(trajectory_views, param_struct_views)) do (traj_view, struct_view)
            @spawn begin
                batch_ibis_step!(
                    time_idx,
                    traj_view,
                    closedloop.dyn,
                    param_prior,
                    param_proposal,
                    nb_ibis_moves,
                    struct_view
                );
            end
        end
        fetch.(tasks);
    end
    return state_struct, param_struct
end


function csmc_with_ibis_marginal_dynamics(
    nb_steps::Int,
    nb_trajectories::Int,
    nb_particles::Int,
    init_state::Vector{Float64},
    closedloop::IBISClosedLoop,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_ibis_moves::Int,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    reference::IBISReference,
    nb_threads::Int = 10,
) where {T<:Function}

    scratch = Array{Float64}(undef, closedloop.dyn.xdim, nb_particles, nb_trajectories)
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = IBISParamStruct(param_prior, nb_steps, nb_particles, nb_trajectories, scratch)

    state_struct.trajectories[:, 1, 1] .= reference.trajectory[:, 1]
    param_struct.particles[:, 1, :, 1] .= reference.particles[:, 1, :]
    param_struct.weights[1, :, 1] .= reference.weights[1, :]
    param_struct.log_weights[1, :, 1] .= reference.log_weights[1, :]
    param_struct.log_likelihoods[1, :, 1] .= reference.log_likelihoods[1, :]

    chunk_size = round(Int, (nb_trajectories - 1) / nb_threads)
    ranges = partition(2:nb_trajectories, chunk_size)

    trajectory_views = [view(state_struct.trajectories, :, :, range) for range in ranges]
    param_struct_views = [view_struct(param_struct, range) for range in ranges]

    for time_idx in 1:nb_steps
        csmc_step_with_ibis_marginal_dynamics!(
            time_idx,
            closedloop,
            action_penalty,
            slew_rate_penalty,
            tempering,
            reference,
            state_struct,
            param_struct,
        )

        param_struct.particles[:, time_idx+1, :, 1] .= reference.particles[:, time_idx+1, :]
        param_struct.weights[time_idx+1, :, 1] .= reference.weights[time_idx+1, :]
        param_struct.log_weights[time_idx+1, :, 1] .= reference.log_weights[time_idx+1, :]
        param_struct.log_likelihoods[time_idx+1, :, 1] .= reference.log_likelihoods[time_idx+1, :]

        tasks = map(zip(trajectory_views, param_struct_views)) do (traj_view, struct_view)
            @spawn begin
                batch_ibis_step!(
                    time_idx,
                    traj_view,
                    closedloop.dyn,
                    param_prior,
                    param_proposal,
                    nb_ibis_moves,
                    struct_view
                );
            end
        end
        fetch.(tasks);
    end
    return state_struct, param_struct
end


function smc_with_rao_blackwell_marginal_dynamics(
    nb_steps::Int,
    nb_trajectories::Int,
    init_state::Vector{Float64},
    closedloop::RaoBlackwellClosedLoop,
    param_prior::Gaussian,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
)
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = RaoBlackwellParamStruct(param_prior, nb_steps, nb_trajectories)

    for time_idx = 1:nb_steps
        smc_step_with_rao_blackwell_marginal_dynamics!(
            time_idx,
            closedloop,
            action_penalty,
            slew_rate_penalty,
            tempering,
            state_struct,
            param_struct,
        )

        @views @inbounds for n = 1:state_struct.nb_trajectories
            q = param_struct.distributions[time_idx, n]
            x = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx, n]
            u = state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx+1, n]
            xn = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx+1, n]
            qn = param_struct.distributions[time_idx+1, n]

            rao_blackwell_dynamics_update!(closedloop.dyn, q, x, u, xn, qn)
        end
    end
    return state_struct, param_struct
end


function csmc_with_rao_blackwell_marginal_dynamics(
    nb_steps::Int,
    nb_trajectories::Int,
    init_state::Vector{Float64},
    closedloop::RaoBlackwellClosedLoop,
    param_prior::Gaussian,
    action_penalty::Float64,
    slew_rate_penalty::Float64,
    tempering::Float64,
    reference::RaoBlackwellReference,
)
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = RaoBlackwellParamStruct(param_prior, nb_steps, nb_trajectories)

    state_struct.trajectories[:, 1, 1] .= reference.trajectory[:, 1]
    param_struct.distributions[1, 1] = deepcopy(reference.distributions[1])
    for time_idx = 1:nb_steps
        csmc_step_with_rao_blackwell_marginal_dynamics!(
            time_idx,
            closedloop,
            action_penalty,
            slew_rate_penalty,
            tempering,
            reference,
            state_struct,
            param_struct,
        )

        param_struct.distributions[time_idx+1, 1] = deepcopy(reference.distributions[time_idx+1])
        @views @inbounds for n = 2:state_struct.nb_trajectories
            q = param_struct.distributions[time_idx, n]
            x = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx, n]
            u = state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx+1, n]
            xn = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx+1, n]
            qn = param_struct.distributions[time_idx+1, n]

            rao_blackwell_dynamics_update!(closedloop.dyn, q, x, u, xn, qn)
        end
    end
    return state_struct, param_struct
end


function ancestor_sampling_csmc_with_rao_blackwell_magrinal_dynamics(
    nb_steps::Int,
    nb_trajectories::Int,
    init_state::Vector{Float64},
    closedloop::RaoBlackwellClosedLoop,
    param_prior::Gaussian,
    action_penalty::Float64,
    tempering::Float64,
    reference::Matrix{Float64},
)
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = RaoBlackwellParamStruct(param_prior, nb_steps, nb_trajectories)

    state_struct.trajectories[:, 1, 1] .= reference[:, 1]
    for time_idx = 1:nb_steps
        ancestor_sampling_csmc_step_with_rao_blackwell_marginal_dynmics!(
            time_idx,
            closedloop,
            action_penalty,
            tempering,
            reference,
            state_struct,
            param_struct,
        )

        param_struct.distributions[time_idx+1, 1] = deepcopy(reference.distributions[time_idx+1])
        @views for n in 2:state_struct.nb_trajectories
            q = param_struct.distributions[time_idx, n]
            x = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx, n]
            u = state_struct.trajectories[closedloop.dyn.xdim+1:end, time_idx+1, n]
            xn = state_struct.trajectories[1:closedloop.dyn.xdim, time_idx+1, n]
            qn = param_struct.distributions[time_idx+1, n]

            # Update the posterior
            rao_blackwell_dynamics_update!(closedloop.dyn, q, x, u, xn, qn)
        end
    end
    return state_struct, param_struct
end


function myopic_smc_with_ibis_marginal_dynamics(
    nb_steps::Int,
    nb_trajectories::Int,
    nb_particles::Int,
    init_state::Vector{Float64},
    adaptive_loop::IBISAdaptiveLoop,
    param_prior::MultivariateDistribution,
    param_proposal::T,
    nb_ibis_moves::Int,
    nb_threads::Int = 10,
) where {T<:Function}

    scratch = Array{Float64}(undef, adaptive_loop.dyn.xdim, nb_particles, nb_trajectories)
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = IBISParamStruct(param_prior, nb_steps, nb_particles, nb_trajectories, scratch)

    chunk_size = round(Int, nb_trajectories / nb_threads)
    ranges = partition(1:nb_trajectories, chunk_size)

    trajectory_views = [view(state_struct.trajectories, :, :, range) for range in ranges]
    param_struct_views = [view_struct(param_struct, range) for range in ranges]

    for time_idx in 1:nb_steps
        myopic_smc_step_with_ibis_marginal_dynamics!(
            time_idx,
            adaptive_loop,
            state_struct,
            param_struct,
        )

        tasks = map(zip(trajectory_views, param_struct_views)) do (traj_view, struct_view)
            @spawn begin
                batch_ibis_step!(
                    time_idx,
                    traj_view,
                    adaptive_loop.dyn,
                    param_prior,
                    param_proposal,
                    nb_ibis_moves,
                    struct_view
                );
            end
        end
        fetch.(tasks);
    end
    return state_struct, param_struct
end


function myopic_smc_with_rao_blackwell_marginal_dynamics(
    nb_steps::Int,
    nb_trajectories::Int,
    init_state::Vector{Float64},
    adaptive_loop::RaoBlackwellAdaptiveLoop,
    param_prior::Gaussian,
)
    state_struct = StateStruct(init_state, nb_steps, nb_trajectories)
    param_struct = RaoBlackwellParamStruct(param_prior, nb_steps, nb_trajectories)

    for time_idx = 1:nb_steps
        myopic_smc_step_with_rao_blackwell_marginal_dynamics!(
            time_idx,
            adaptive_loop,
            state_struct,
            param_struct,
        )

        @views @inbounds for n = 1:state_struct.nb_trajectories
            q = param_struct.distributions[time_idx, n]
            x = state_struct.trajectories[1:adaptive_loop.dyn.xdim, time_idx, n]
            u = state_struct.trajectories[adaptive_loop.dyn.xdim+1:end, time_idx+1, n]
            xn = state_struct.trajectories[1:adaptive_loop.dyn.xdim, time_idx+1, n]
            qn = param_struct.distributions[time_idx+1, n]

            rao_blackwell_dynamics_update!(adaptive_loop.dyn, q, x, u, xn, qn)
        end
    end
    return state_struct, param_struct
end