# Author: Aqeel Anwar(ICSRL)
# Created: 5/16/2019, 3:17 PM
# Email: aqeel.anwar@gatech.edu


# The code uses already vailable tello_py repository which has been modified to fit the needs of this code
# https://github.com/hanyazou/TelloPy
import traceback
from tellopy.tello_drone import tello_drone
from DeepNet.network.agent import DeepAgent
from DeepNet.network.heat_map import heat_map
from DeepNet.util.Memory import Memory
from RL_functions import *

import numpy as np
from sys import platform


import time
import os, sys
import configparser as cp
from configs.read_cfg import read_cfg
import cv2
from dotmap import DotMap

# Read the configuration file
cfg = read_cfg(verbose=True)
print('test')

# ---------- Initialize necessary variables
stat_path = 'DeepNet/models/'+cfg.run_name+'/'+cfg.env_type+'/stat.npy'
network_path_half = 'DeepNet/models/'+cfg.run_name+'/'+cfg.env_type+'/'
network_path = network_path_half + '/agent/agent'
data_path = network_path_half+'data_tuple.npy'
input_size = 227
epsilon = 0
action_type = 'Begin'
data_tuple, stat, iteration = load_data(cfg.load_data, cfg.load_data_path)
epi=0
old_state = []
choose = False
reverse_action = [3, 2, 1, 0]
recovering = False
ReplayMemory = Memory(cfg.buffer_len, cfg.load_data, cfg.load_data_path)
action = 0
avoid_action = -1
num_actions = cfg.num_actions

# ---------- Initialize agents
agent = DeepAgent(input_size=input_size,
                  num_actions=cfg.num_actions,
                  train_fc='e2e',
                  name='agent',
                  env_type='Indoor',
                  custom_load=cfg.custom_load,
                  custom_load_path=cfg.custom_load_path,
                  tensorboard_path = cfg.load_data_path)

target_agent = DeepAgent(input_size=input_size,
                  num_actions=cfg.num_actions,
                  train_fc='e2e',
                  name='target_agent',
                  env_type='Indoor',
                  custom_load=cfg.custom_load,
                  custom_load_path=cfg.custom_load_path,
                  tensorboard_path = cfg.load_data_path)
# Load heat map network and initliaze the drone connection
DepthAgent = heat_map()

# --------- Initialize drone
drone = tello_drone()

# --------- Initiliaze dict
dict = DotMap()
dict.stat_path = stat_path
dict.network_path = network_path
dict.agent = agent
dict.target_agent = target_agent
dict.data_tuple = data_tuple
dict.data_path = data_path
dict.stat = stat
dict.stat = stat
dict.load_path = cfg.load_data_path
dict.Replay_memory = ReplayMemory
just_begin = True
# I am running the code on two platforms. My MacBook is much slower
# than my GPU installed windows laptop, hence higher skip_frame
if platform == 'win32':
    skip_frame = 60
elif platform == 'darwin':
    skip_frame = 150

if __name__ == '__main__':
    screen = drone.pygame_connect(960, 720)
    container, drone_handle = drone.connect()
    manual = True
    frame_skip = skip_frame
    while True:
        # flightDataHandler()
        try:
            for frame in container.decode(video=0):
                if 0 < frame_skip:
                    frame_skip = frame_skip - 1
                    continue
                    # print(frame)
                else:
                    start_time = time.time()
                    frame_skip = skip_frame

                    # Define control take-over
                    if manual:
                        # print("Entering manual mode")
                        drone.take_action_3(drone_handle, -1)
                        manual = drone.check_action(drone_handle, manual, dict)
                        drone.display(frame, manual)
                    else:

                        # Check in manual over-ride
                        manual = drone.if_takeover(drone_handle)

                        # Update necessary variables here
                        iteration += 1
                        drone_stat = drone.get_drone_stats()
                        dict.iteration = iteration
                        dict.data_tuple = data_tuple
                        dict.stat = stat
                        dict.Replay_memory = ReplayMemory

                        # Do calculations here
                        # Display image from front camera
                        drone.display(frame, manual)
                        new_state = agent.state_from_frame(frame)
                        depth_map_3D, depth_float_2D, global_depth = DepthAgent.depth_map_gen(frame)
                        reward, done = agent.reward_gen(depth_float_2D, depth_map_3D, action, crash_threshold=cfg.crash_thresh, display=True)
                        # print('Reward is: ', reward)
                        if not just_begin:
                            if not recovering: # or remove this condition
                                data_tuple = []
                                data_tuple.append([old_state, action, new_state, reward, action_type])
                                err = get_errors(data_tuple, choose, input_size, agent, target_agent, cfg.Q_clip, cfg.gamma)
                                ReplayMemory.add(err, data_tuple)
                                stat.append([iteration, epi, action, action_type, epsilon, reward, cfg.lr])
                            else:
                                data_tuple=[]
                                data_tuple.append([new_state, action, new_state, reward, action_type])
                                err = get_errors(data_tuple, choose, input_size, agent, target_agent, cfg.Q_clip, cfg.gamma)
                                ReplayMemory.add(err, data_tuple)
                                data_tuple = []
                                data_tuple.append([new_state, 0, new_state, reward, action_type])
                                err = get_errors(data_tuple, choose, input_size, agent, target_agent, cfg.Q_clip, cfg.gamma)
                                ReplayMemory.add(err, data_tuple)

                        if reward == -1:
                            epi = epi+1
                            action_type = 'Rcvr'
                            recovering = True
                            # reverse action

                            rev_action = reverse_action[action]
                            drone.take_action_3(drone_handle, rev_action)
                            time.sleep(0.7)
                            drone.take_action_3(drone_handle, -1)
                            time.sleep(0.4)
                            # Add data augmentation to tuple

                # ----------------------- End of episode ___________________#

                        else:
                            recovering = False
                            # Step 1: Policy evaluation
                            action, action_type, epsilon = agent.policy(
                                                            epsilon=epsilon,
                                                            curr_state=new_state,
                                                            iter=iteration,
                                                            eps_sat=cfg.epsilon_saturation,
                                                            eps_model='exponential',
                                                            avoid_action=avoid_action)
                            # action = 1
                            # drone.mark_frame(action, num_actions, frame)
                            drone.take_action_3(drone_handle, action)
                            time.sleep(0.8)
                            drone.take_action_3(drone_handle, -1)
                            time.sleep(0.4)

                        # Train if required
                        if iteration >= cfg.wait_before_train:
                            if iteration % cfg.update_target_interval == 0:
                                choose = not choose
                                agent.save_network(iteration, network_path)

                            old_states, Qvals, actions, err, idx = minibatch_double(
                                                        data_tuple=data_tuple,
                                                        batch_size=cfg.batch_size,
                                                        choose=choose,
                                                        ReplayMemory=ReplayMemory,
                                                        input_size=input_size,
                                                        agent=agent,
                                                        target_agent=target_agent,
                                                        Q_clip=cfg.Q_clip,
                                                        gamma=cfg.gamma)

                            for i in range(cfg.batch_size):
                                ReplayMemory.update(idx[i], err[i])

                            if choose:
                                target_agent.train_n(old_states, Qvals, actions, cfg.batch_size, cfg.dropout_rate, cfg.lr, epsilon,
                                                     iteration)
                            else:
                                agent.train_n(old_states, Qvals, actions, cfg.batch_size, cfg.dropout_rate, cfg.lr,
                                                     epsilon,
                                                     iteration)


                            # save network
                            if iteration % 100 == 0:
                                print('Saving the learned network...')
                                np.save(stat_path, stat)
                                agent.save_network(iteration, network_path)
                                np.save(data_path, data_tuple)
                                Memory.save(cfg.load_path)

                        # # Training
                        # # Generate Heat Map
                        # depth_map_3D, depth_float_2D, global_depth = DepthAgent.depth_map_gen(frame, display=False)
                        # reward, done = agent.reward_gen(depth_float_2D, depth_map_3D, act, crash_threshold=2.0)
                        # print('Reward is: ', reward)
                        # reward = gen_reward(heat_map)
                        # data_tuple([s, a, s_, r])

                        old_state = new_state

                        print_action = action
                        if recovering:
                            print_action = rev_action

                        print(
                            'Iteration: {:>4d} / {:<3d} Action: {:<3d} - {:>4s} Eps: {:<1.4f} Reward:  {:>+1.4f}  lr: {:>f} len D: {:<5d}'.format(
                                iteration,
                                epi,
                                print_action,
                                action_type,
                                epsilon,
                                reward,
                                cfg.lr,
                                len(data_tuple)
                                )
                        )
                        just_begin=False
                    end_time = time.time()
                    time_per_iter = end_time-start_time
                    frame_skip = int(time_per_iter*60)
                    # print("Frame skip: ", frame_skip)


        except Exception as e:
            print('------------- Error -------------')
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)
            print(e)
            print(traceback.format_exc())
            print('Landing the drone and shutting down')
            print('---------------------------------')
            drone_handle.land()
            time.sleep(5)
            drone_handle.quit()
            exit(1)









