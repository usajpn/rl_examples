import numpy as np
from pprint import pprint as pp
from gridworld import GridworldEnv
env = GridworldEnv()

def print_policy(policy):
    print("Grid Policy (0=up, 1=right, 2=down, 3=left)")
    print(np.reshape(np.argmax(policy, axis=1), env.shape))
    print("")

def print_V(V):
    print("Value")
    print(np.reshape(V, env.shape))
    print("")

def print_policy_V(policy, V):
    print_policy(policy)  
    print_V(V)

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    env:
        env.P[s][a]: transition probabilities (prob, next_state, reward, done)
        env.nS: number of states
        env.nA: number of actions

    env index:
        0  1  2  3
        4  5  6  7
        8  9 10 11
       12 13 14 15

    action directions:
              0
              ^
              |
        3 <--   --> 1
              |
              V
              2
    
    theta: the value for which we stop iterating. We stop iterating when
           max(v(s_t+1) - v(s_t)) < theta

    returns:
        (policy, V) which is the optimal policy and value function
    
    variables:
        V: value function V(s)
        policy: pi(a|s)
    """
    
    def one_step_lookahead(state, V):
        """
        Calucate value of when taking action at a given state
        
        state:
            the state to consider
        V:
            the value to use for estimating

        returns:
            the action-value Q for each action
        """
        Q = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                Q[a] += prob * (reward + discount_factor * V[next_state])
        return Q 

    # value for every state is initialized by 0
    V = np.zeros(env.nS)

    # policy for state x action
    policy = np.zeros([env.nS, env.nA])

    i = 0
    print("starting value iteration with initial V:")
    print_V(V)

    # loop until the max value difference between timesteps is less than theta
    while True:

        # the max difference of the value
        max_delta = 0
        # loop in state x action
        for s in range(env.nS):
            # one step lookahead to calculate all Q-values in each state
            Q = one_step_lookahead(s, V)
            best_Q = np.max(Q)
            
            # calculate delta across all states
            max_delta = max(max_delta, np.abs(best_Q - V[s]))
            
            # assuming that we take the best action, replace V with the best 
            # action value
            V[s] = best_Q
        
        i += 1
        print("iter:{}".format(i))
        print_V(V)

        if max_delta < theta:
            print(
             "max_delta {} has gone below theta {}!!".format(max_delta, theta))
            print("")
            break
    
    # A deterministic policy creation. Just take the best action in every state
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        Q = one_step_lookahead(s, V)
        best_Q = np.max(Q)
        best_action = np.argmax(Q)
        policy[s, best_action] = 1.0

    return policy, V

def main():
    policy, V = value_iteration(env)
    print_policy_V(policy, V)

main()


