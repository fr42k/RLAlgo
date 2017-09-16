def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA])

    def get_action_value_f(env, state, V, discount_factor):
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[s][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
        return A

    while True:
        delta = 0
        for s in range(env.nS):
            A = get_action_value_f(env, s, V, discount_factor)
            best_v = np.max(A)
            delta = max(delta, abs(V[s] - best_v))
            V[s] = best_v
        if (delta < theta):
            break

    for s in range(env.nS):
        A = get_action_value_f(env, s, V, discount_factor)
        a = np.argmax(A)
        policy[s, a] = 1.0

    return policy, V