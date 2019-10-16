# Author: Aqeel Anwar(ICSRL)
# Created: 9/21/2019, 7:16 PM
# Email: aqeel.anwar@gatech.edu

import numpy as np
import time

def minibatch_double(data_tuple, batch_size, choose, ReplayMemory, input_size, agent, target_agent, Q_clip, gamma):
    # Needs NOT to be in DeepAgent
    # NO TD error term, and using huber loss instead
    # Bellman Optimality equation update, with less computation, updated

    if batch_size==1:
        train_batch = data_tuple
        idx=None
    else:
        batch = ReplayMemory.sample(batch_size)
        train_batch = np.array([b[1][0] for b in batch])
        idx = [b[0] for b in batch]


    actions = np.zeros(shape=(batch_size), dtype=int)
    # crashes = np.zeros(shape=(batch_size))
    rewards = np.zeros(shape=batch_size)
    curr_states = np.zeros(shape=(batch_size, input_size, input_size, 3))
    new_states = np.zeros(shape=(batch_size, input_size, input_size, 3))
    for ii, m in enumerate(train_batch):
        curr_state_m, action_m, new_state_m, reward_m, crash_m = m
        curr_states[ii, :, :, :] = curr_state_m[...]
        actions[ii] = action_m
        new_states[ii, :, :, :] = new_state_m
        rewards[ii] = reward_m
        # crashes[ii] = crash_m

    #
    # oldQval = np.zeros(shape = [batch_size, num_actions])
    if choose:
        oldQval_A = target_agent.Q_val(curr_states)
        newQval_A = target_agent.Q_val(new_states)
        newQval_B = agent.Q_val(new_states)
    else:
        oldQval_A = agent.Q_val(curr_states)
        newQval_A = agent.Q_val(new_states)
        newQval_B = target_agent.Q_val(new_states)


    TD = np.zeros(shape=[batch_size])
    Q_target = np.zeros(shape=[batch_size])

    term_ind = np.where(rewards==-1)[0]
    nonterm_ind = np.where(rewards!=-1)[0]

    TD[nonterm_ind] = rewards[nonterm_ind] + gamma* newQval_B[nonterm_ind, np.argmax(newQval_A[nonterm_ind], axis=1)] - oldQval_A[nonterm_ind, actions[nonterm_ind].astype(int)]
    TD[term_ind] = rewards[term_ind]

    if Q_clip:
        TD_clip = np.clip(TD, -1, 1)
    else:
        TD_clip = TD

    Q_target[nonterm_ind] = oldQval_A[nonterm_ind, actions[nonterm_ind].astype(int)] + TD_clip[nonterm_ind]
    Q_target[term_ind] = TD_clip[term_ind]

    err=abs(TD) # or abs(TD_clip)
    return curr_states, Q_target, actions, err, idx



def recover(data_tuple, new_state, action, reverse_action, drone, drone_handle ):
    reward = -1
    while reward == -1:
        action_type = 'crash_recover'
        # Moving forward from crash is crash too
        data_tuple.append([new_state, 0, new_state, reward, action_type])
        print('Crashed: Moving ', reverse_action(action_transpose[action]))
        # drone.move_action(reverse_action[action], speed)
        drone.take_action_3(drone_handle, reverse_action[action])
        if action == 0:
            time.sleep(0.7)
            drone.hover()
            time.sleep(1)
        else:
            time.sleep(0.6)
            drone.hover()
            time.sleep(0.4)
        print('getting new frame ', end='')
        temp_state, temp_frame, cam = get_state(cam)
        temp_state, temp_frame, cam = get_state(cam)
        temp_state, temp_frame, cam = get_state(cam)
        # print('Done : ', end='')
        depth_map, reward = reward_gen(L_old, C_old, R_old, temp_frame, action)
        print('depth generated')
        if reward == -1:
            # can't go forward either'
            data_tuple.append([temp_state, 0, temp_state, reward, action_type])

        # Transpose of the action is crash too - out of the for loop so that the uncrash frame is captured too
        data_tuple.append([temp_state, action, temp_state, -1, 'crash_recover'])

    avoid_action = action

def load_data(load, load_path):
    if load:
        print("Loading data_tuple from: ", load_path)
        data_tuple = list(np.load(load_path + 'data_tuple.npy'))
        stat = list(np.load(load_path + 'stat.npy'))
        iteration = int(stat[-1][0])
    else:
        data_tuple = []
        stat = []
        iteration = 0

    print("Data tuple loaded: ", len(data_tuple))

    return data_tuple, stat, iteration

def get_errors(data_tuple, choose, input_size, agent, target_agent, Q_clip, gamma):

    _, Q_target, _, err, _ = minibatch_double(data_tuple, len(data_tuple), choose, 0, input_size, agent, target_agent, Q_clip, gamma)

    return err