import numpy as np
import gym
import matplotlib.pyplot as plt
import pickle

pos_space = np.linspace(-1.2, 0.6, 12)
vel_space = np.linspace(-0.07, 0.07, 20)

# Utility functions
def digitize_obs(obs):
    pos, vel = obs
    return int(np.digitize(pos, pos_space)), int(np.digitize(vel, vel_space))

def create_q_table(obss, actions):
    # create Q table
    Q = {}
    for obs in obss:
        for action in actions:
            Q[obs, action] = 0
    return Q

def action_value(obs, Q, actions):
    return np.array([Q[obs, a] for a in actions])

# RL specific functions
def e_greedy_action(cur_obs, cur_Q, actions, eps):
    action = np.random.choice(actions) if np.random.random() < eps \
                       else np.argmax(action_value(cur_obs, cur_Q, actions))
    return action

def q_learning_mountain_car():
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = 1000
    render = False
    num_episodes = 50000
    alpha = 0.1
    gamma = 0.99
    eps = 1.0

    print('Observation:', env.observation_space)
    print('Action:', env.action_space)

    # create a digitized state space
    obss = []
    for pos in range(22):
        for vel in range(21):
            obss.append((pos, vel))
    
    # create action space (0:push left, 1:do nothing, 2:push right)
    actions = [0, 1, 2]
    
    Q = create_q_table(obss, actions)
    total_ep_rewards = np.zeros(num_episodes)

    for ep in range(num_episodes):
        obs_t = env.reset()
        obs_t = digitize_obs(obs_t)
        ep_reward = 0
        while True:
            action_t = e_greedy_action(obs_t, Q, actions, eps)
            obs_tp1, reward, done, _ = env.step(action_t)
            if render:
                env.render()
            obs_tp1 = digitize_obs(obs_tp1)
            ep_reward += reward

            action_tp1 = np.argmax(action_value(obs_t, Q, actions))
            Q[obs_t, action_t] = Q[obs_t, action_t] + \
                    alpha * (reward + gamma * Q[obs_tp1, action_tp1] - \
                             Q[obs_t, action_t])
            obs_t = obs_tp1
            
            if done:
                break
        eps = eps - 2 / num_episodes if eps > 0.01 else 0.01
        if ep % 100 == 0:
            print(ep, ep_reward, eps)

        total_ep_rewards[ep] = ep_reward
    
    # plot graph
    smoothed_ep_rewards = np.zeros(num_episodes)
    for ep in range(num_episodes):
        smoothed_ep_rewards[ep] = np.mean(total_ep_rewards[max(0, ep-50):(ep+1)])
    plt.plot(smoothed_ep_rewards)
    plt.savefig('q_learning_mountaincar.png')

    f = open('mountaincar.pkl', 'wb')
    pickle.dump(Q, f)
    f.close()

q_learning_mountain_car()


