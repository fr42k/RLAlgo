def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    import numpy as np
    V = np.zeros(env.nS)
    while True:
        delta = 0.0
        for s in range(env.nS):
            v = 0
            for a, prob_a in enumerate(policy[s]):
                for prob_t, next_state, reward, done in env.P[s][a]:
                    v += prob_a * prob_t * (reward + discount_factor * V[next_state])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)
