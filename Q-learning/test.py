import numpy as np
import pandas as pd
import time

np.random.seed(2)

N_STATES = 6
ACTIONS = ['left', 'right']
EPSILON = 0.9  # greedy search policy
ALPHA = 0.1  # learning rate
GAMMA = 0.9  # discount factor
MAX_EPISODES = 13
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))),  # q_table initial values
        columns=actions,  # actions' name
    )
    # print(table)
    return table


# x_axis: state, y_axis: actions, and the value will be Q(s,a) then
# uncomment line 27 to show initial value of Q-table
# build_q_table(N_STATES, ACTIONS)

def choose_action(state, q_table):
    # this is how to choose an action
    # get one row of some "state" in Q-table
    state_actions = q_table.iloc[state, :]
    if (np.random.uniform() > EPSILON) or ((state_actions == 0).all()):
        action_name = np.random.choice(ACTIONS)
    else:
        # get the max one who has the max-Q
        action_name = state_actions.idxmax()    # replace argmax to idxmax as argmax means a different function in newer version of pandas
    return action_name


def get_env_feedback(state, action):
    # this is how agent will interact with the environment
    if action == 'right':
        if state == N_STATES - 2:  # terminate
            state__ = 'terminal'
            reward = 1
        else:
            state__ = state + 1  # move right
            reward = 0
    else:       # move left
        reward = 0
        if state == 0:
            state__ = state  # reach start_point
        else:
            state__ = state - 1  # move_left
    return state__, reward  # initial state or continues to move left, there will be no reward


def update_env(state, episode, step_counter):
    # this if how environment will be updated
    env_list = ['-'] * (N_STATES - 1) + ['T']   # '---------T' our environment
    if state == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
        print('\r{}'.format(interaction), end='')
        time.sleep(2)
        print('\r                                ', end='')
    else:
        env_list[state] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction), end='')
        time.sleep(FRESH_TIME)


def rl():
    # main part of RL loop
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        state = 0
        is_terminated = False
        update_env(state, episode, step_counter)

        while not is_terminated:
            action = choose_action(state, q_table)
            state__, reward = get_env_feedback(state, action)  # take action and get feedback, move to next state
            q_predict = q_table.loc[state, action]  # predicted value of Q(state, action)
            if state__ != 'terminal':  # next state is not terminal
                q_target = reward + GAMMA * q_table.iloc[state__, :].max()
            else:  # next state is exactly terminal
                q_target = reward
                is_terminated = True

            q_table.loc[state, action] += ALPHA * (q_target - q_predict)
            state = state__

            update_env(state, episode, step_counter+1)
            step_counter += 1
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
