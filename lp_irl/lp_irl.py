
import random

import numpy as np
from cvxopt import matrix, solvers

def compute_reward(n_states, n_actions, trans_prob, policy, discount,
                    max_reward, l1_reg):
    """
    Find the reward function with linear programming IRL

    n_states: number of states
    n_actions: number of actions
    trans_prob: transition probability which is a numpy array which maps
                state(t), action(t) to state(t+1) (state_t, action_t, state_tp1)
    policy: mapping of states to actions
    discount: discount factor
    max_reward: maximum reward
    l1_reg: l1 regularization

    return: reward vector
    """

    A = set(range(n_actions))
    trans_prob = np.transpose(trans_prob, (1, 0, 2))
    
    # (P_a1(i) - P_a(i)) * inv(I - gamma * P_a1(i))
    def T(a, s):
        return np.dot(trans_prob[policy[s], s] -
                      trans_prob[a, s],
                      np.linalg.inv(np.eye(n_states) - 
                        discount * trans_prob[policy[s]]))
    
    c = -np.hstack([np.zeros(n_states), np.ones(n_states),
                    -l1_reg*np.ones(n_states)])
    zero_stack1 = np.zeros((n_states*(n_actions-1), n_states))
    T_stack = np.vstack([
        -T(a, s)
        for s in range(n_states)
        for a in A - {policy[s]} # other than the chosen a
    ])
    I_stack1 = np.vstack([
        np.eye(1, n_states, s)
        for s in range(n_states)
        for a in A - {policy[s]}
    ])
    I_stack2 = np.eye(n_states)
    zero_stack2 = np.zeros((n_states, n_states))
    
    D_left = np.vstack([T_stack, T_stack, -I_stack2, I_stack2])
    D_middle = np.vstack([I_stack1, zero_stack1, zero_stack2, zero_stack2])
    D_right = np.vstack([zero_stack1, zero_stack1, -I_stack2, -I_stack2])

    D = np.hstack([D_left, D_middle, D_right])
    b = np.zeros((n_states * (n_actions - 1) * 2 + 2 * n_states, 1))
    bounds = np.array([(None, None)] * 2 * n_states + [(-max_reward, max_reward)] * n_states)

    D_bounds = np.hstack([
        np.vstack([
            -np.eye(n_states),
            np.eye(n_states)]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))]),
        np.vstack([
            np.zeros((n_states, n_states)),
            np.zeros((n_states, n_states))])])

    b_bounds = np.vstack([max_reward*np.ones((n_states, 1))]*2)
    D = np.vstack((D, D_bounds))
    b = np.vstack((b, b_bounds))
    A_ub = matrix(D)
    b = matrix(b)
    c = matrix(c)
    results = solvers.lp(c, A_ub, b)
    r = np.asarray(results["x"][:n_states], dtype=np.double)

    return r.reshape((n_states))


