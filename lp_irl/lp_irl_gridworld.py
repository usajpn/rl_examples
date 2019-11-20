"""
Linear programming IRL on gridworld MDP
"""

import numpy as np
import matplotlib.pyplot as plt

import gridworld
import lp_irl

def lp_irl_gridworld(grid_size, discount):
    wind = 0.3
    traj_len = 3 * grid_size

    gw = gridworld.Gridworld(grid_size, wind, discount)
    gt_reward = np.array([gw.reward(s) for s in range(gw.n_states)])
    policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)]
    r = lp_irl.compute_reward(gw.n_states, gw.n_actions, gw.transition_probability,
            policy, gw.discount, 1, 5)

    plt.subplot(1, 2, 1)
    plt.pcolor(gt_reward.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(gt_reward.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

lp_irl_gridworld(5, 0.2)
