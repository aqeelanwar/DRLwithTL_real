import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os
import tensorflow as tf
import cv2
from DeepNet.network.network import *
# import airsim
import random
# import psutil
# from os import getpid

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


from DeepNet.network.loss_functions import *
from numpy import linalg as LA
# class DeepAgentConditional():
#     def __init__(self, input_size, num_actions, train_fc, name):
#         self.g = tf.Graph()
#         with self.g.as_default():
#
#             self.stat_writer = tf.summary.FileWriter('D:/train/return_plot')
#             name_array = 'D:/train/loss'+'/'+name
#             self.loss_writer = tf.summary.FileWriter(name_array)
#             self.input_size = input_size
#             self.num_actions = num_actions
#
#             #Placeholders
#             self.batch_size = tf.placeholder(tf.int32, shape=())
#             self.learning_rate = tf.placeholder(tf.float32, shape=())
#             self.X1 = tf.placeholder(tf.float32, [None, input_size, input_size, 3], name='States')
#
#             #self.X = tf.image.resize_images(self.X1, (227, 227))
#
#
#             self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
#             self.target = tf.placeholder(tf.float32,    shape = [None], name='Qvals')
#             self.actions= tf.placeholder(tf.int32,      shape = [None], name='Actions')
#
#             initial_weights ='imagenet'
#             initial_weights = 'models/weights/weights.npy'
#             self.model = AlexNetConditional(self.X, num_actions, train_fc)
#             ind = tf.one_hot(self.actions, num_actions)
#
#             # Main branch loss
#             pred_Q_main = tf.reduce_sum(tf.multiply(self.model.output_main, ind), axis=1)
#             self.loss_main = huber_loss(pred_Q_main, self.target)
#             self.train_main = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(self.loss_main, name="train_main")
#
#             # Conditional branch loss
#             # Main branch loss
#             pred_Q_cdl = tf.reduce_sum(tf.multiply(self.model.output_cdl, ind), axis=1)
#             self.loss_cdl = huber_loss(pred_Q_cdl, self.target)
#             self.train_cdl = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(self.loss_cdl, name="train_cdl")
#
#
#             self.sess = tf.InteractiveSession()
#             tf.global_variables_initializer().run()
#             tf.local_variables_initializer().run()
#             self.saver = tf.train.Saver()
#             self.sess.graph.finalize()
#
#         self.cdl_count=0
#         self.main_count=0
#
#     def state_from_frame(self, frame):
#         img_rgb = np.asarray(frame.to_image())
#         state = cv2.resize(img_rgb, (self.input_size, self.input_size), cv2.INTER_LINEAR)
#         state = cv2.normalize(state, state, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
#         state_rgb = []
#         state_rgb.append(state[:, :, 0:3])
#         state_rgb = np.array(state_rgb)
#         state_rgb = state_rgb.astype('float32')
#         return state_rgb
#
#
#     def Q_val(self, xs, branch_name):
#         target = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
#         actions = np.zeros(dtype=int, shape=[xs.shape[0]])
#         if branch_name=='main':
#             return self.sess.run(self.model.output_main,
#                       feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
#                                  self.target: target, self.actions:actions})
#         elif branch_name=='cdl':
#             return self.sess.run(self.model.output_cdl,
#                                  feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
#                                             self.target: target, self.actions: actions})
#         else:
#             print('Error: Branch can either be main or cdl')
#
#     def train_n(self, xs, ys,actions, batch_size, dropout_rate, lr, epsilon, iter, branch_name):
#
#         if branch_name=='main':
#             _, loss, Q = self.sess.run([self.train_main,self.loss_main, self.model.output_main],
#                       feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.X1: xs,
#                                                self.target: ys, self.actions: actions})
#             summ_tag = '_main'
#
#         elif branch_name=='cdl':
#             _, loss, Q = self.sess.run([self.train_cdl, self.loss_cdl, self.model.output_cdl],
#                                        feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.X1: xs,
#                                                   self.target: ys, self.actions: actions})
#             summ_tag = '_cdl'
#
#         meanQ = np.mean(Q)
#         maxQ = np.max(Q)
#
#         summary = tf.Summary()
#         # summary.value.add(tag='Loss', simple_value=LA.norm(loss[ind, actions.astype(int)]))
#         summary.value.add(tag='Loss'+summ_tag, simple_value=LA.norm(loss)/batch_size)
#         self.loss_writer.add_summary(summary, iter)
#
#         summary = tf.Summary()
#         summary.value.add(tag='Epsilon', simple_value=epsilon)
#         self.loss_writer.add_summary(summary, iter)
#
#         summary = tf.Summary()
#         summary.value.add(tag='Learning Rate', simple_value=lr)
#         self.loss_writer.add_summary(summary, iter)
#
#         summary = tf.Summary()
#         summary.value.add(tag='MeanQ'+summ_tag, simple_value=meanQ)
#         self.loss_writer.add_summary(summary, iter)
#
#         summary = tf.Summary()
#         summary.value.add(tag='MaxQ'+summ_tag, simple_value=maxQ)
#         self.loss_writer.add_summary(summary, iter)
#
#         # return _correct
#
#     def action_selection(self, state, branch_name=''):
#         target = np.zeros(shape=[state.shape[0]], dtype=np.float32)
#         actions = np.zeros(dtype=int, shape=[state.shape[0]])
#
#         qvals_main= self.sess.run(self.model.output_main,
#                              feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
#                                         self.X1: state,
#                                         self.target: target, self.actions:actions})
#
#         qvals_cdl = self.sess.run(self.model.output_cdl,
#                                    feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
#                                               self.X1: state,
#                                               self.target: target, self.actions: actions})
#
#         # summary = tf.summary.histogram("normal/moving_mean", qvals)
#         # # summary = tf.summary.merge_all()
#         # # summary.value.add(tag='Learning Rate', simple_value=lr)
#         # self.loss_writer.add_summary(summ)
#         if branch_name=='main':
#             qvals = qvals_main
#         elif branch_name=='cdl':
#             qvals = qvals_cdl
#         elif branch_name=='smart':
#             # threshold the cdl q values
#             # ss  = np.sum(np.exp(qvals_cdl), axis=1)
#             qvals_cdl_prob = np.exp(qvals_cdl)/np.sum(np.exp(qvals_cdl), axis=1)
#             qvals_main_prob = np.exp(qvals_main) / np.sum(np.exp(qvals_main), axis=1)
#
#             # print('Main max q val prob: ', np.amax(qvals_main_prob))
#             # print('CDL  max q val prob: ', np.amax(qvals_cdl_prob))
#             if np.amax(qvals_cdl_prob)>(1):
#                 qvals = qvals_cdl
#                 self.cdl_count+=1
#             else:
#                 qvals = qvals_main
#                 self.main_count += 1
#
#         else:
#             # used for training
#             if np.amax(qvals_main) > np.amax(qvals_cdl):
#                 qvals = qvals_main
#             else:
#                 qvals = qvals_cdl
#
#         if qvals.shape[0]>1:
#             # Evaluating batch
#             action = np.argmax(qvals, axis=1)
#         else:
#             # Evaluating one sample
#             # self.pred_action_count = self.pred_action_count+1
#             action = np.zeros(1)
#             action[0]=np.argmax(qvals)
#             cc=1
#             # Find mean and std of Q VALUE AND REPORT THROUGH TENSORBOARD
#             # summary = tf.Summary()
#             # summary.value.add(tag='Mean Q', simple_value=np.max(qvals))
#             # summary.value.add(tag='Std Q', simple_value=np.std(qvals))
#             # self.test_writer_ret.add_summary(summary, self.pred_action_count)
#
#
#             # self.action_array[action[0].astype(int)]+=1
#         return action.astype(int)
#
#     def action_counts(self):
#         return self.main_count, self.cdl_count
#
#     # def take_action(self, action, num_actions):
#     #     # Set Paramaters
#     #     fov_v = 45 * np.pi / 180
#     #     fov_h = 80 * np.pi / 180
#     #     r = 0.4
#     #
#     #     ignore_collision = False
#     #     sqrt_num_actions = np.sqrt(num_actions)
#     #
#     #
#     #
#     #     # action_array_ind = range(sqrt_num_actions)
#     #     # theta_array = np.array([-2, -1, 0, 1, 2]) * fov_v / sqrt_num_actions
#     #     # psi_array = np.array([-2, -1, 0, 1, 2]) * fov_h / sqrt_num_actions
#     #
#     #     posit = self.client.simGetVehiclePose()
#     #     pos = posit.position
#     #     orientation = posit.orientation
#     #
#     #     quat = (orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)
#     #     eulers = euler_from_quaternion(quat)
#     #     alpha = eulers[2]
#     #
#     #     theta_ind = int(action[0] / sqrt_num_actions)
#     #     psi_ind = action[0] % sqrt_num_actions
#     #
#     #     theta = fov_v/sqrt_num_actions * (theta_ind - (sqrt_num_actions - 1) / 2)
#     #     psi = fov_h / sqrt_num_actions * (psi_ind - (sqrt_num_actions - 1) / 2)
#     #
#     #
#     #     # theta = theta_array[theta_ind]
#     #     # psi = psi_array[psi_ind]
#     #
#     #     # print('psi ', psi)
#     #     # print('theta ', theta)
#     #     # Rotating the action field - TEMPORARY
#     #     # psi = psi+fov_h/(5*4)
#     #     # theta = theta +fov_v/(5*4)
#     #
#     #     # Dilating the action field
#     #     #psi = psi * 1.2
#     #     #theta = theta * 1.2
#     #     noise_theta = (fov_v / sqrt_num_actions) / 6
#     #     noise_psi = (fov_h / sqrt_num_actions) / 6
#     #
#     #     psi = psi + random.uniform(-1, 1)*noise_psi
#     #     theta = theta + random.uniform(-1, 1)*noise_theta
#     #
#     #     # print('Theta: ', theta * 180 / np.pi, end='')
#     #     # print(' Psi: ', psi * 180 / np.pi)
#     #
#     #
#     #
#     #
#     #     x = pos.x_val + r * np.cos(alpha + psi)
#     #     y = pos.y_val + r * np.sin(alpha + psi)
#     #     z = pos.z_val + r * np.sin(theta)  # -ve because Unreal has -ve z direction going upwards
#     #
#     #     self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, alpha + psi)),
#     #                              ignore_collison=ignore_collision)
#
#     # def get_depth(self):
#     #     env_type = 'Indoor'
#     #     responses = self.client.simGetImages([airsim.ImageRequest(2, airsim.ImageType.DepthVis, False, False)])
#     #     depth = []
#     #     img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
#     #     depth = img1d.reshape(responses[0].height, responses[0].width, 4)[:, :, 0]
#     #     if env_type == 'Outdoor':
#     #         thresh = 20
#     #     elif env_type == 'Indoor':
#     #         thresh = 50
#     #     super_threshold_indices = depth > thresh
#     #     depth[super_threshold_indices] = thresh
#     #     depth = depth / thresh
#     #     # plt.imshow(depth)
#     #     # # plt.gray()
#     #     # plt.show()
#     #     return depth
#     #
#     # def get_state(self):
#     #     responses1 = self.client.simGetImages([  # depth visualization image
#     #         airsim.ImageRequest("1", airsim.ImageType.Scene, False,
#     #                             False)])  # scene vision image in uncompressed RGBA array
#     #
#     #     response = responses1[0]
#     #     img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
#     #     img_rgba = img1d.reshape(response.height, response.width, 4)
#     #     img = Image.fromarray(img_rgba)
#     #     img_rgb = img.convert('RGB')
#     #
#     #     state = np.asarray(img_rgb)
#     #
#     #     state = cv2.resize(state, (self.input_size, self.input_size), cv2.INTER_LINEAR)
#     #     state = cv2.normalize(state, state, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
#     #     state_rgb = []
#     #     state_rgb.append(state[:, :, 0:3])
#     #     state_rgb = np.array(state_rgb)
#     #     state_rgb = state_rgb.astype('float32')
#     #     # plt.imshow(state)
#     #     # plt.show()
#     #     # return_dict[0] = state_rgb
#     #     return state_rgb
#
#     def avg_depth(self, depth_map1):
#         # Version 0.2
#         # Thresholded depth map to ignore objects too far and give them a constant value
#         # Globally (not locally as in the version 0.1) Normalise the thresholded map between 0 and 1
#         # Threshold depends on the environment nature (indoor/ outdoor)
#         depth_map = depth_map1
#         # dynamic_window = False
#         plot_depth = False
#
#         # if env=='Complex_Outdoor':
#         #     d_div = 170
#         # elif env=='Complex_Indoor':
#         #     d_div=290
#         # elif env=='Test_Indoor_House':
#         #     d_div=290
#         # elif env=='Test_Indoor_Apartment':
#         #     d_div=290
#         # # depth_map1=depth_map
#         # # Smooth the image to avoid small spaces betwen bushes and all
#         # # kernel = np.ones((3, 3), np.float32) / 25
#         # # depth_map = cv2.filter2D(depth_map, -1, kernel)
#         # # depth_map = 255 - (255 * (depth_map - np.max(depth_map)) / np.ptp(depth_map)).astype(np.uint8)
#         #
#         # #Rescale between [0,1]
#         # depth_map = 1 - depth_map1 / d_div
#         #
#         # # global_depth = 200 / np.mean(depth_map)
#         # # # depth_map=depth_map1
#         # # n = 5 / 6.5 * global_depth + 0.5461
#         # # n=global_depth/2
#         # #const window size
#         # # if not dynamic_window:
#         n = 3
#         H = np.size(depth_map, 0)
#         W = np.size(depth_map, 1)
#         grid_size = np.array([H, W]) / n
#         # scale by 0.8 to select the window towards top from the mid line
#         if self.env_type == 'Outdoor':
#             h = 0.9 * H * (n - 1) / (2 * n)
#         elif self.env_type == 'Indoor':
#             h = 0.9 * H * (n - 1) / (2 * n)
#
#         w = W * (n - 1) / (2 * n)
#         grid_location = [h, w]
#         x1 = int(round(grid_location[0]))
#         y = int(round(grid_location[1]))
#
#         a4 = int(round(grid_location[0] + grid_size[0]))
#
#         a5 = int(round(grid_location[0] + grid_size[0]))
#         b5 = int(round(grid_location[1] + grid_size[1]))
#
#         a2 = int(round(grid_location[0] - grid_size[0]))
#         b2 = int(round(grid_location[1] + grid_size[1]))
#
#         a8 = int(round(grid_location[0] + 2 * grid_size[0]))
#         b8 = int(round(grid_location[1] + grid_size[1]))
#
#         b4 = int(round(grid_location[1] - grid_size[1]))
#         if b4 < 0:
#             b4 = 0
#
#         a6 = int(round(grid_location[0] + grid_size[0]))
#         b6 = int(round(grid_location[1] + 2 * grid_size[1]))
#         if b6 > 319:
#             b6 = 319
#
#         # L = 1 / np.min(depth_map[x1:a4, b4:y])
#         # C = 1 / np.min(depth_map[x1:a5, y:b5])
#         # R = 1 / np.min(depth_map[x1:a6, b5:b6])
#
#         fract_min = 0.05
#
#         L_map = depth_map[x1:a4, b4:y]
#         C_map = depth_map[x1:a5, y:b5]
#         R_map = depth_map[x1:a6, b5:b6]
#
#         L_sort = np.sort(L_map.flatten())
#         end_ind = int(np.round(fract_min * len(L_sort)))
#         L1 = np.mean(L_sort[0:end_ind])
#
#         R_sort = np.sort(R_map.flatten())
#         end_ind = int(np.round(fract_min * len(R_sort)))
#         R1 = np.mean(R_sort[0:end_ind])
#
#         C_sort = np.sort(C_map.flatten())
#         end_ind = int(np.round(fract_min * len(C_sort)))
#         C1 = np.mean(C_sort[0:end_ind])
#         if plot_depth:
#             cv2.rectangle(depth_map1, (y, x1), (b5, a5), (0, 0, 0), 1)
#             cv2.rectangle(depth_map1, (y, x1), (b4, a4), (0, 0, 0), 1)
#             cv2.rectangle(depth_map1, (b5, x1), (b6, a6), (0, 0, 0), 1)
#
#             dispL = str(np.round(L1, 3))
#             dispC = str(np.round(C1, 3))
#             dispR = str(np.round(R1, 3))
#             cv2.putText(depth_map1, dispL, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
#             cv2.putText(depth_map1, dispC, (110, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0))
#             cv2.putText(depth_map1, dispR, (200, 75), cv2.FONT_HERSHEY_COMPLEX, 0.3, (0, 0, 0))
#
#             # plt.imshow(depth_map1)
#             # plt.show()
#             xxx = 1
#             # time.sleep(0.7)
#         #
#         xxxxx = 1
#         # print(L1, C1, R1)
#         return L1, C1, R1
#
#
#
#     def reward_gen(self, d_new, action, crash_threshold):
#         L_new, C_new, R_new = self.avg_depth(d_new)
#         # print('Rew_C', C_new)
#         # print(L_new, C_new, R_new)
#         # For now, lets keep the reward a simple one
#         if C_new < crash_threshold:
#             done = True
#             reward = -1
#         else:
#             done = False
#             if action == 0:
#                 reward = C_new
#             else:
#                 # reward = C_new/3
#                 reward = C_new
#
#             # if action != 0:
#             #     reward = 0
#
#         return reward, done
#
#     def GetAgentState(self, agent_type):
#         return self.client.simGetCollisionInfo()
#
#     def return_plot(self, ret, epi, env_type, mem_percent, iter, dist):
#         # ret, epi1, int(level/4), mem_percent, iter
#         summary = tf.Summary()
#         env_array = ['Pyramid', 'FrogEyes', 'UpDown','Long', 'VanLeer', 'ComplexIndoor', 'Techno', 'GT']
#         tag = env_array[env_type]
#         summary.value.add(tag=tag, simple_value=ret)
#         self.stat_writer.add_summary(summary, epi)
#
#         summary = tf.Summary()
#         summary.value.add(tag='Memory-GB', simple_value=mem_percent)
#         self.stat_writer.add_summary(summary, iter)
#
#         summary = tf.Summary()
#         summary.value.add(tag='Safe Flight', simple_value=dist)
#         self.stat_writer.add_summary(summary, epi)
#
#     def save_network(self, save_path):
#         self.saver.save(self.sess, save_path)
#
#     def save_weights(self, save_path):
#         name = ['conv1W', 'conv1b', 'conv2W', 'conv2b', 'conv3W', 'conv3b', 'conv4W', 'conv4b', 'conv5W', 'conv5b',
#                 'fc6aW', 'fc6ab', 'fc7aW', 'fc7ab', 'fc8aW', 'fc8ab', 'fc9aW', 'fc9ab', 'fc10aW', 'fc10ab',
#                 'fc6vW', 'fc6vb', 'fc7vW', 'fc7vb', 'fc8vW', 'fc8vb', 'fc9vW', 'fc9vb', 'fc10vW', 'fc10vb'
#                 ]
#         weights = {}
#         print('Saving weights in .npy format')
#         for i in range(0, 30):
#             # weights[name[i]] = self.sess.run(self.sess.graph._collections['variables'][i])
#             if i==0:
#                 str1 = 'Variable:0'
#             else:
#                 str1 = 'Variable_'+str(i)+':0'
#             weights[name[i]] = self.sess.run(str1)
#         save_path = save_path+'weights.npy'
#         np.save(save_path, weights)
#
#     def load_network(self, load_path):
#         self.saver.restore(self.sess, load_path)
#
#
#
#     def get_weights(self):
#         xs=np.zeros(shape=(32, 227,227,3))
#         actions = np.zeros(dtype=int, shape=[xs.shape[0]])
#         ys = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
#         return self.sess.run(self.weights,
#                              feed_dict={self.batch_size: xs.shape[0],  self.learning_rate: 0,
#                                         self.X1: xs,
#                                         self.target: ys, self.actions:actions})
#

class DeepAgent():
    def __init__(self, input_size, num_actions,train_fc, name, env_type, custom_load, custom_load_path, tensorboard_path):
        print('--------------------- Loading DeepAgent ---------------------')
        self.g = tf.Graph()
        self.env_type=env_type
        self.iter=0
        with self.g.as_default():

            self.stat_writer = tf.summary.FileWriter(tensorboard_path)
            name_array = tensorboard_path+name
            self.loss_writer = tf.summary.FileWriter(name_array)

            self.input_size = input_size
            self.num_actions = num_actions

            #Placeholders
            self.batch_size = tf.placeholder(tf.int32, shape=())
            self.learning_rate = tf.placeholder(tf.float32, shape=())
            self.X1 = tf.placeholder(tf.float32, [None, input_size, input_size, 3], name='States')

            #self.X = tf.image.resize_images(self.X1, (227, 227))


            self.X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.X1)
            self.target = tf.placeholder(tf.float32,    shape = [None], name='Qvals')
            self.actions= tf.placeholder(tf.int32,      shape = [None], name='Actions')

            initial_weights ='imagenet'
            initial_weights = 'models/weights/weights.npy'
            self.model = AlexNetDuel(self.X, num_actions, train_fc)

            self.predict = self.model.output
            ind = tf.one_hot(self.actions, num_actions)
            pred_Q = tf.reduce_sum(tf.multiply(self.model.output, ind), axis=1)
            actual_cost = huber_loss(pred_Q, self.target)



            beta = 0.00001
            self.loss = tf.reduce_mean(actual_cost + beta * self.model.regularizer)




            self.train = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.99).minimize(self.loss, name="train")

            self.sess = tf.InteractiveSession()
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            self.saver = tf.train.Saver()

            self.sess.graph.finalize()

            if custom_load:
                print('Loading weights from: ', custom_load_path)
                self.load_network(custom_load_path)


    def state_from_frame(self, frame):
        img_rgb = np.asarray(frame.to_image())
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        state = cv2.resize(img_rgb, (self.input_size, self.input_size), cv2.INTER_LINEAR)
        state = cv2.normalize(state, state, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        state_rgb = []
        state_rgb.append(state[:, :, 0:3])
        state_rgb = np.array(state_rgb)
        state_rgb = state_rgb.astype('float32')
        return state_rgb

    def Q_val(self, xs):
        target = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        return self.sess.run(self.predict,
                      feed_dict={self.batch_size: xs.shape[0], self.learning_rate: 0, self.X1: xs,
                                 self.target: target, self.actions:actions})


    def train_n(self, xs, ys,actions, batch_size, dropout_rate, lr, epsilon, iter):
        # loss = self.sess.run(self.loss,
        #                      feed_dict={self.batch_size: batch_size, self.dropout_rate: dropout_rate, self.learning_rate: lr, self.X: xs,
        #                                        self.Y: ys, self.actions:actions})
        _, loss, Q = self.sess.run([self.train,self.loss, self.predict],
                      feed_dict={self.batch_size: batch_size, self.learning_rate: lr, self.X1: xs,
                                               self.target: ys, self.actions: actions})
        meanQ = np.mean(Q)
        maxQ = np.max(Q)



        summary = tf.Summary()
        # summary.value.add(tag='Loss', simple_value=LA.norm(loss[ind, actions.astype(int)]))
        summary.value.add(tag='Loss', simple_value=LA.norm(loss)/batch_size)
        self.loss_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='Epsilon', simple_value=epsilon)
        self.loss_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='Learning Rate', simple_value=lr)
        self.loss_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='MeanQ', simple_value=meanQ)
        self.loss_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='MaxQ', simple_value=maxQ)
        self.loss_writer.add_summary(summary, iter)

        # return _correct

    def action_selection(self, state):
        target = np.zeros(shape=[state.shape[0]], dtype=np.float32)
        actions = np.zeros(dtype=int, shape=[state.shape[0]])
        qvals= self.sess.run(self.predict,
                             feed_dict={self.batch_size: state.shape[0], self.learning_rate: 0.0001,
                                        self.X1: state,
                                        self.target: target, self.actions:actions})

        # summary = tf.summary.histogram("normal/moving_mean", qvals)
        # # summary = tf.summary.merge_all()
        # # summary.value.add(tag='Learning Rate', simple_value=lr)
        # self.loss_writer.add_summary(summ)

        if qvals.shape[0]>1:
            # Evaluating batch
            action = np.argmax(qvals, axis=1)
        else:
            # Evaluating one sample
            # self.pred_action_count = self.pred_action_count+1
            action = np.zeros(1)
            action[0]=np.argmax(qvals)
            cc=1
            # Find mean and std of Q VALUE AND REPORT THROUGH TENSORBOARD
            # summary = tf.Summary()
            # summary.value.add(tag='Mean Q', simple_value=np.max(qvals))
            # summary.value.add(tag='Std Q', simple_value=np.std(qvals))
            # self.test_writer_ret.add_summary(summary, self.pred_action_count)


            # self.action_array[action[0].astype(int)]+=1
        return action.astype(int)

    #
    # def take_action(self, action):
    #     # Set Paramaters
    #     fov_v = 45 * np.pi / 180
    #     fov_h = 80 * np.pi / 180
    #     r = 1
    #
    #     ignore_collision = False
    #     theta_array = np.array([-2, -1, 0, 1, 2]) * fov_v / 5
    #     psi_array = np.array([-2, -1, 0, 1, 2]) * fov_h / 5
    #
    #     posit = self.client.simGetVehiclePose()
    #     pos = posit.position
    #     orientation = posit.orientation
    #
    #     quat = (orientation.w_val, orientation.x_val, orientation.y_val, orientation.z_val)
    #     eulers = euler_from_quaternion(quat)
    #     alpha = eulers[2]
    #
    #     theta_ind = int(action[0] / 5)
    #     psi_ind = action[0] % 5
    #
    #     theta = theta_array[theta_ind]
    #     psi = psi_array[psi_ind]
    #
    #     # print('psi ', psi)
    #     # print('theta ', theta)
    #     # Rotating the action field - TEMPORARY
    #     # psi = psi+fov_h/(5*4)
    #     # theta = theta +fov_v/(5*4)
    #
    #     # Dilating the action field
    #     #psi = psi * 1.2
    #     #theta = theta * 1.2
    #
    #     psi = psi + random.uniform(0, 1) / 15
    #     theta = theta + random.uniform(0, 1) / 15
    #
    #
    #
    #
    #
    #     x = pos.x_val + r * np.cos(alpha + psi)
    #     y = pos.y_val + r * np.sin(alpha + psi)
    #     z = pos.z_val + r * np.sin(theta)  # -ve because Unreal has -ve z direction going upwards
    #
    #     self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(0, 0, alpha + psi)),
    #                              ignore_collison=ignore_collision)

    # def get_depth(self):
    #     env_type = 'Indoor'
    #     responses = self.client.simGetImages([airsim.ImageRequest(2, airsim.ImageType.DepthVis, False, False)])
    #     depth = []
    #     img1d = np.fromstring(responses[0].image_data_uint8, dtype=np.uint8)
    #     depth = img1d.reshape(responses[0].height, responses[0].width, 4)[:, :, 0]
    #     if env_type == 'Outdoor':
    #         thresh = 20
    #     elif env_type == 'Indoor':
    #         thresh = 50
    #     super_threshold_indices = depth > thresh
    #     depth[super_threshold_indices] = thresh
    #     depth = depth / thresh
    #     # plt.imshow(depth)
    #     # # plt.gray()
    #     # plt.show()
    #     return depth
    #
    # def get_state(self):
    #     responses1 = self.client.simGetImages([  # depth visualization image
    #         airsim.ImageRequest("1", airsim.ImageType.Scene, False,
    #                             False)])  # scene vision image in uncompressed RGBA array
    #
    #     response = responses1[0]
    #     img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)  # get numpy array
    #     img_rgba = img1d.reshape(response.height, response.width, 4)
    #     img = Image.fromarray(img_rgba)
    #     img_rgb = img.convert('RGB')
    #     self.iter = self.iter+1
    #     state = np.asarray(img_rgb)
    #     im = Image.fromarray(state)
    #     strr = 'images/'+str(self.iter)+'.jpeg'
    #     im.save(strr)
    #
    #
    #     # plt.imshow(state)
    #     # plt.show()
    #     state = cv2.resize(state, (self.input_size, self.input_size), cv2.INTER_LINEAR)
    #     state = cv2.normalize(state, state, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    #     state_rgb = []
    #     state_rgb.append(state[:, :, 0:3])
    #     state_rgb = np.array(state_rgb)
    #     state_rgb = state_rgb.astype('float32')
    #     # plt.imshow(state)
    #     # plt.show()
    #     # return_dict[0] = state_rgb
    #     return state_rgb

    def avg_depth(self, depth_float_2D, depth_map_3D, display=True):

        H,W  = np.shape(depth_float_2D)

        global_depth = np.max(depth_float_2D)
        n = 5 / 6.5 * global_depth + 0.5461
        # n=global_depth/2
        grid_size = [H, W] / n
        h = H * (n - 1) / (2 * n)
        w = W * (n - 1) / (2 * n)
        grid_location = [h, w]
        x1 = int(round(grid_location[0]))
        y = int(round(grid_location[1]))

        a4 = int(round(grid_location[0] + grid_size[0]))

        a5 = int(round(grid_location[0] + grid_size[0]))
        b5 = int(round(grid_location[1] + grid_size[1]))

        a2 = int(round(grid_location[0] - grid_size[0]))
        b2 = int(round(grid_location[1] + grid_size[1]))

        a8 = int(round(grid_location[0] + 2 * grid_size[0]))
        b8 = int(round(grid_location[1] + grid_size[1]))

        b4 = int(round(grid_location[1] - grid_size[1]))
        if b4 < 0:
            b4 = 0

        b4 = 0

        a6 = int(round(grid_location[0] + grid_size[0]))
        b6 = int(round(grid_location[1] + 2 * grid_size[1]))
        if b6 > 639:
            b6 = 639
        b6 = 639
        L = np.round(np.mean(depth_float_2D[x1:a4, b4:y]) ** 2, 4)
        C = np.round(np.mean(depth_float_2D[x1:a5, y:b5]) ** 2, 4)
        R = np.round(np.mean(depth_float_2D[x1:a6, b5:b6]) ** 2, 4)

        L1 = np.round(np.min(depth_float_2D[x1:a4, b4:y]) ** 2, 4)
        C1 = np.round(np.min(depth_float_2D[x1:a5, y:b5]) ** 2, 4)
        R1 = np.round(np.min(depth_float_2D[x1:a6, b5:b6]) ** 2, 4)

        Write_DM = True
        if Write_DM:
            # cv2.rectangle(frame, (0, up_margin), (int(W / 2) - 1, down_margin), (0, 0, 0), 2)
            # cv2.rectangle(frame, (int(W / 2), up_margin), (W, down_margin), (0, 0, 0), 2)
            # cv2.rectangle(frame, (side_margin, 0), (W - side_margin, H), (0, 0, 0), 2)

            cv2.rectangle(depth_map_3D, (y, x1), (b5, a5), (0, 0, 0), 3)
            cv2.rectangle(depth_map_3D, (y, x1), (b4, a4), (0, 0, 0), 3)
            cv2.rectangle(depth_map_3D, (b5, x1), (b6, a6), (0, 0, 0), 3)

            dispL = str(L1)
            dispC = str(C1)
            dispR = str(R1)
            cv2.putText(depth_map_3D, dispL, (105, 179), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (255, 255, 255))
            cv2.putText(depth_map_3D, dispC, (315, 179), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (0, 0, 0))
            cv2.putText(depth_map_3D, dispR, (530, 179), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3, (0, 0, 0))

        if display:
            cv2.imshow('Depth Map', depth_map_3D)
            cv2.waitKey(1)

        return L1, C1, R1



    def reward_gen(self, depth_float_2D, depth_map_3D, action, crash_threshold, display):
        L_new, C_new, R_new = self.avg_depth(depth_float_2D, depth_map_3D, display)
        # print('Rew_C', C_new)
        # print(L_new, C_new, R_new)
        # print('thresh', crash_threshold)

        if C_new < crash_threshold:
            done = True
            reward = -1
        else:
            done = False
            if action == 0:
                reward = C_new
            else:
                # reward = C_new/3
                reward = C_new

            # if action != 0:
            #     reward = 0
        # print(reward, done)
        return reward, done

    def GetAgentState(self, agent_type):
        return self.client.simGetCollisionInfo()

    def return_plot(self, ret, epi, env_type, mem_percent, iter, dist):
        # ret, epi1, int(level/4), mem_percent, iter
        summary = tf.Summary()
        env_array = ['Pyramid', 'FrogEyes', 'UpDown','Long', 'VanLeer', 'ComplexIndoor', 'Techno', 'GT']
        tag = env_array[env_type]
        summary.value.add(tag=tag, simple_value=ret)
        self.stat_writer.add_summary(summary, epi)

        summary = tf.Summary()
        summary.value.add(tag='Memory-GB', simple_value=mem_percent)
        self.stat_writer.add_summary(summary, iter)

        summary = tf.Summary()
        summary.value.add(tag='Safe Flight', simple_value=dist)
        self.stat_writer.add_summary(summary, epi)

    def save_network(self, iteration, save_path):
        save_path += str(iteration)
        self.saver.save(self.sess, save_path)

    def save_weights(self, save_path):
        name = ['conv1W', 'conv1b', 'conv2W', 'conv2b', 'conv3W', 'conv3b', 'conv4W', 'conv4b', 'conv5W', 'conv5b',
                'fc6aW', 'fc6ab', 'fc7aW', 'fc7ab', 'fc8aW', 'fc8ab', 'fc9aW', 'fc9ab', 'fc10aW', 'fc10ab',
                'fc6vW', 'fc6vb', 'fc7vW', 'fc7vb', 'fc8vW', 'fc8vb', 'fc9vW', 'fc9vb', 'fc10vW', 'fc10vb'
                ]
        weights = {}
        print('Saving weights in .npy format')
        for i in range(0, 30):
            # weights[name[i]] = self.sess.run(self.sess.graph._collections['variables'][i])
            if i==0:
                str1 = 'Variable:0'
            else:
                str1 = 'Variable_'+str(i)+':0'
            weights[name[i]] = self.sess.run(str1)
        save_path = save_path+'weights.npy'
        np.save(save_path, weights)

    def load_network(self, load_path):
        self.saver.restore(self.sess, load_path)



    def get_weights(self):
        xs=np.zeros(shape=(32, 227,227,3))
        actions = np.zeros(dtype=int, shape=[xs.shape[0]])
        ys = np.zeros(shape=[xs.shape[0]], dtype=np.float32)
        return self.sess.run(self.weights,
                             feed_dict={self.batch_size: xs.shape[0],  self.learning_rate: 0,
                                        self.X1: xs,
                                        self.target: ys, self.actions:actions})


    def policy(self, epsilon, curr_state, iter, eps_sat, eps_model,avoid_action):
        # The new action space is {A_new} = {A_old}-avoid_action
        action = avoid_action
        epsilon_ceil = 0.95
        if eps_model == 'linear':
            epsilon = epsilon_ceil * iter/eps_sat
            if epsilon > epsilon_ceil:
                epsilon = epsilon_ceil

        elif eps_model == 'exponential':
            epsilon = 1 - np.exp(-2 / eps_sat*iter)
            if epsilon > epsilon_ceil:
                epsilon = epsilon_ceil

        if random.random() > epsilon:
            while action == avoid_action:
                state_shape = curr_state.shape
                action = np.random.randint(0, self.num_actions, size=state_shape[0], dtype=np.int32)
                action_type = 'Rand'
        else:
            # Use NN to predict action
            action = self.action_selection(curr_state)
            action_type = 'Pred'
            # print(action_array/(np.mean(action_array)))
        return action[0], action_type, epsilon








