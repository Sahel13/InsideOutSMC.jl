using InsideOutSMC

using Random
using Distributions
using LinearAlgebra


function compare_policies_with_rao_blackwell_updates(
    dynamics_obj::StochasticDynamics,
    rb_dynamics_obj::RaoBlackwellDynamics,
    policy_dict::Dict,
    param_prior::Gaussian,
    init_state::Vector{Float64},
    nb_steps::Int,
    nb_evals::Int,
)
    xdim = rb_dynamics_obj.xdim
    udim = rb_dynamics_obj.udim

    entropy_vals = Dict(key => [] for key in keys(policy_dict))

    for _ in 1:nb_evals
        dynamics_seed = rand(1:Int(1e9))
        Random.seed!(dynamics_seed)

        cond_covar = rao_blackwell_conditional_dynamics_covar(rb_dynamics_obj)
        dist = MvNormal(zeros(xdim), cond_covar)
        dynamics_noise = rand(dist, nb_steps)

        policy_seed = rand(1:Int(1e9))

        for (policy_name, policy_obj) in pairs(policy_dict)
            Random.seed!(policy_seed)

            trajectory = Array{Float64}(undef, xdim+udim, nb_steps + 1)
            trajectory[:, 1] = init_state

            Flux.reset!(policy_obj)
            for t = 1:nb_steps
                state = trajectory[1:xdim, t]
                action = policy_mean(policy_obj, trajectory[:, t])
                next_state = dynamics_mean(dynamics_obj, state, action) + dynamics_noise[:, t]
                trajectory[:, t+1] = vcat(next_state, action)
            end

            param_posterior = deepcopy(param_prior)
            for t in 1:nb_steps
                param_posterior = rao_blackwell_dynamics_update(
                    rb_dynamics_obj,
                    param_posterior,
                    trajectory[1:xdim, t],
                    trajectory[xdim+1:end, t+1],
                    trajectory[1:xdim, t+1]
                )
            end
            mvn = MvNormal(param_posterior.mean, Symmetric(param_posterior.covar))
            push!(entropy_vals[policy_name], entropy(mvn))
        end
    end

    for (key, val) in pairs(entropy_vals)
        entropy_vals[key] = [mean(val), std(val)]
    end

    return entropy_vals
end


function rollout_with_rao_blackwell_updates(
    dynamics_obj::StochasticDynamics,
    rb_dynamics_obj::RaoBlackwellDynamics,
    policy_obj::StochasticPolicy,
    param_prior::Gaussian,
    init_state::Vector{Float64},
    nb_steps::Int,
)
    xdim = rb_dynamics_obj.xdim
    udim = rb_dynamics_obj.udim

    dynamics_seed = rand(1:Int(1e9))
    Random.seed!(dynamics_seed)

    cond_covar = rao_blackwell_conditional_dynamics_covar(rb_dynamics_obj)
    dist = MvNormal(zeros(xdim), cond_covar)
    dynamics_noise = rand(dist, nb_steps)

    trajectory = Array{Float64}(undef, xdim+udim, nb_steps + 1)
    trajectory[:, 1] = init_state

    param_posterior = deepcopy(param_prior)

    Flux.reset!(policy_obj)
    for t = 1:nb_steps
        state = trajectory[1:xdim, t]
        action = policy_mean(policy_obj, trajectory[:, t])
        next_state = dynamics_mean(dynamics_obj, state, action) + dynamics_noise[:, t]
        trajectory[:, t+1] = vcat(next_state, action)

        param_posterior = rao_blackwell_dynamics_update(
            rb_dynamics_obj,
            param_posterior,
            trajectory[1:xdim, t],
            trajectory[xdim+1:end, t+1],
            trajectory[1:xdim, t+1]
        )
    end
    return trajectory, param_posterior
end
