def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """
    # Start with a random policy
    import numpy as np
    policy = np.ones([env.nS, env.nA]) / env.nA

    while True:
        policy_stable = True
        V = policy_eval_fn(policy, env, discount_factor)
        for s in range(env.nS):
            old_a = np.argmax(policy[s])
            action_value_fn = np.zeros(env.nA)
            for a in range(env.nA):
                for prob_t, next_state, reward, done in env.P[s][a]:
                    action_value_fn[a] += prob_t * (reward + discount_factor * V[next_state])
            new_a = np.argmax(action_value_fn)

            if old_a != new_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[new_a]
        if policy_stable:
            return policy, V
