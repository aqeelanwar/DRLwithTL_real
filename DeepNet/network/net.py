import numpy as np
import os
import tensorflow as tf
import cv2
from network.alexnet_net import AlexNet
from network.vgg16_net import VGG16




# Freezing the first few layers and training the last layers only
class DeepAgentRL(object):

    def __init__(self, lr_const, weights_path, num_actions, train_type, network_type, regularization, input_size, dtype=np.float32):
        # weights_path = Defines initialization
        #                    Imagenet, Random
        # train_type = Defines the what layers of the network needs to be trained
        # network_type = Defines network topology
        #                    Alexnet, Reduced_alexnet, date
        # regularization = Should the loss function contain regularization loss
        self.pred_action_count=0
        with tf.device('/device:GPU:0'):

            # Define graph and corresponding session
            self.g2 = tf.Graph()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            self.sess_det = tf.Session(config=config, graph=self.g2)

            # Add elements to graph
            with self.g2.as_default():
                self.y_true = tf.placeholder(dtype, shape=[None], name='y_true')
                self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')

                self.NN = AlexNet(num_actions, weights_path, train_type,network_type, regularization, input_size,dtype,  arch_type='alexnet')
                # self.NN = VGG16(num_actions, weights_path, train_type,network_type, regularization, arch_type='vgg16')

                q_t = self.NN.fc10
                # self.y_pred = tf.nn.softmax(q_t, name='y_pred')

                self.init = tf.global_variables_initializer()
                self.sess_det.run(self.init)

                q_summary=[]
                avg_q = tf.reduce_mean(q_t, 0)
                # for idx in range(0, num_actions):
                #     q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
                # self.q_summary = tf.summary.merge(q_summary, 'q_summary')
                # c = tf.range(tf.shape(self.y_true)[0]) * tf.shape(self.y_true)[1]
                # ind = c + self.actions
                # onehot_predictions = tf.gather(tf.reshape(self.y_true, [-1]), ind)

                ind = tf.range(tf.shape(q_t)[0]) * tf.shape(q_t)[1] + self.actions
                onehot_predictions_q_t = tf.gather(tf.reshape(q_t, [-1]), ind)


                # # MSE Loss
                # cost = tf.squared_difference(onehot_predictions_q_t, self.y_true)
                # cost1 = tf.reduce_mean(cost)

                # Huber Loss
                err = self.y_true - onehot_predictions_q_t
                loss_v = tf.where(tf.abs(err) < 1.0,
                                  0.5 * tf.square(err),
                                  tf.abs(err) - 0.5)
                cost1 = tf.reduce_sum(loss_v)

                # cost1 = tf.reduce_mean(tf.square(q_t - self.y_true), name="Loss")
                # cost1 = tf.reduce_mean(tf.square(), name='Loss')
                if regularization:
                    regularizers_w = tf.nn.l2_loss(self.NN.fc6W) + tf.nn.l2_loss(self.NN.fc7W) + tf.nn.l2_loss(
                        self.NN.fc8W) + tf.nn.l2_loss(self.NN.fc9W)+ tf.nn.l2_loss(self.NN.fc10W)+ tf.nn.l2_loss(self.NN.conv1W) + tf.nn.l2_loss(self.NN.conv2W) + tf.nn.l2_loss(
                        self.NN.conv3W) + tf.nn.l2_loss(self.NN.conv4W) + tf.nn.l2_loss(self.NN.conv5W)

                    regularizers_b = tf.nn.l2_loss(self.NN.fc6b) + tf.nn.l2_loss(self.NN.fc7b) + tf.nn.l2_loss(
                        self.NN.fc8b) + tf.nn.l2_loss(self.NN.fc9b) + tf.nn.l2_loss(self.NN.fc10b) + tf.nn.l2_loss(
                        self.NN.conv1b) + tf.nn.l2_loss(self.NN.conv2b) + tf.nn.l2_loss(
                        self.NN.conv3b) + tf.nn.l2_loss(self.NN.conv4b) + tf.nn.l2_loss(self.NN.conv5b)

                    regularizers = regularizers_b+regularizers_w
                    beta = 0.00001
                    # beta=0
                    cost = tf.reduce_mean(cost1 + beta * regularizers)
                else:
                    cost=cost1
                tf.summary.scalar('Loss', cost)

                # clip_error=False
                # if clip_error:
                #     cost = tf.clip_by_value(cost, -1, 1)



                step = tf.Variable(0, trainable=False)
                # learning_rate_start = tf.placeholder(tf.float32, shape=(), name='learning_rate_start')
                # decay_rate = tf.placeholder(tf.float32, shape=(), name="decay_rate")
                # End-End
                # learning_rate_start = tf.constant(lr_const)
                # decay_rate = 0.9992
                # rate_decay = tf.train.exponential_decay(learning_rate_start, step, 1, decay_rate)
                self.rate = tf.placeholder(tf.float32, shape=[])
                # decay_rate = 0.99995
                # rate_decay = tf.train.exponential_decay(self.rate, step, 1, decay_rate)
                rate_decay = self.rate
                tf.summary.scalar('Learning Rate', rate_decay)

                # rate=learning_rate_start
                self.optimizer = tf.train.AdamOptimizer(learning_rate=rate_decay).minimize(cost, global_step=step,
                                                                                name="optimizer")

                # optimizer = tf.train.AdamOptimizer(learning_rate=5e-8).minimize(cost)
                # correct_prediction = tf.equal(y_pred_cls, y_true_cls, name="correct_prediction")
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
                # tf.summary.scalar('Validation Accuracy', accuracy)
                self.merged = tf.summary.merge_all()
                self.test_writer = tf.summary.FileWriter('D:/train/loss')
                # tf.summary.FileWriter('D:/train/loss', self.sess_det.graph)
                self.sess_det.run(tf.global_variables_initializer())
                # y_test_images = np.zeros((1, 3))
                self.saver = tf.train.Saver()

                self.sess_det.graph.finalize()

            self.g_ret = tf.Graph()

            # with self.g_ret.as_default():
            #     self.test_writer_ret = tf.summary.FileWriter('D:/train/return_plot')
            #
            #     self.ret = tf.Variable(0.0)
            #     tf.summary.scalar("Return", self.ret)
            #
            #     self.write_op_ret = tf.summary.merge_all()
            #     self.session_ret = tf.Session(config=config, graph=self.g_ret)
            #     self.session_ret.run(tf.global_variables_initializer())

            with self.g_ret.as_default():
                self.test_writer_ret = tf.summary.FileWriter('D:/train/return_plot')

                self.ret = tf.placeholder(tf.float32, shape=())
                self.mem_per = tf.placeholder(tf.float32, shape=())
                self.summary1 = tf.summary.scalar("Return1", self.ret)
                self.summary2 = tf.summary.scalar("Return2", self.ret)
                self.summary3 = tf.summary.scalar("Return3", self.ret)
                self.summary4 = tf.summary.scalar("Return4", self.ret)

                self.summary5 = tf.summary.scalar("Memory Used(GB)", self.mem_per)

                # self.write_op_ret = tf.summary.merge_all()
                self.session_ret = tf.Session(config=config, graph=self.g_ret)
                self.session_ret.run(tf.global_variables_initializer())

                self.session_ret.graph.finalize()


    def action_selection(self, state):

        qvals = self.sess_det.run(self.NN.fc10, feed_dict={self.NN.x: state})
        # print(qvals)
        if qvals.shape[0]>1:
            # Evaluating batch
            action = np.argmax(qvals, axis=1)
        else:
            # Evaluating one sample
            self.pred_action_count = self.pred_action_count+1
            action = np.zeros(1)
            action[0]=np.argmax(qvals)
            # Find mean and std of Q VALUE AND REPORT THROUGH TENSORBOARD
            summary = tf.Summary()
            summary.value.add(tag='Mean Q', simple_value=np.max(qvals))
            summary.value.add(tag='Std Q', simple_value=np.std(qvals))
            self.test_writer_ret.add_summary(summary, self.pred_action_count)
        return action.astype(int), qvals

    def Q_val(self, state):
        qvals = self.sess_det.run(self.NN.fc10, feed_dict={self.NN.x: state})
        return qvals

    def train(self, old_states, Qvals,i, lr, actions, epsilon):
        summary = self.sess_det.run([self.optimizer, self.merged], feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr, self.actions: actions})
        # summary = self.sess_det.run(self.merged, feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
        self.test_writer.add_summary(summary[1], i)
        summary = tf.Summary()
        summary.value.add(tag='Epsilon', simple_value=epsilon)
        self.test_writer_ret.add_summary(summary, i)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata=tf.RunMetadata()
        # self.test_writer.add_run_metadata(run_metadata, 'step%d', i)

        # self.test_writer.add_graph(self.sess_det.graph)

    # def train(self, old_states, Qvals,i, lr):
    #     summary = self.sess_det.run([self.optimizer, self.merged], feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
    #     # summary = self.sess_det.run(self.merged, feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
    #     self.test_writer.add_summary(summary[1], i)
    #
    #     # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     # run_metadata=tf.RunMetadata()
    #     # self.test_writer.add_run_metadata(run_metadata, 'step%d', i)
    #
    #     # self.test_writer.add_graph(self.sess_det.graph)
    def save_network(self, epoch, save_path, type):

        with self.sess_det.as_default():
            if type=='save_critic':
                self.saver.save(self.sess_det, save_path)
            else:
                conv1_up = [self.NN.conv1W.eval(), self.NN.conv1b.eval()]
                conv2_up = [self.NN.conv2W.eval(), self.NN.conv2b.eval()]
                conv3_up = [self.NN.conv3W.eval(), self.NN.conv3b.eval()]
                conv4_up = [self.NN.conv4W.eval(), self.NN.conv4b.eval()]
                conv5_up = [self.NN.conv5W.eval(), self.NN.conv5b.eval()]
                fc6_up = [self.NN.fc6W.eval(), self.NN.fc6b.eval()]
                fc7_up = [self.NN.fc7W.eval(), self.NN.fc7b.eval()]
                fc8_up = [self.NN.fc8W.eval(), self.NN.fc8b.eval()]
                fc9_up = [self.NN.fc9W.eval(), self.NN.fc9b.eval()]
                fc10_up = [self.NN.fc10W.eval(), self.NN.fc10b.eval()]
                network_param = {
                    'conv1': conv1_up,
                    'conv2': conv2_up,
                    'conv3': conv3_up,
                    'conv4': conv4_up,
                    'conv5': conv5_up,
                    'fc6': fc6_up,
                    'fc7': fc7_up,
                    'fc8': fc8_up,
                    'fc9': fc9_up,
                    'fc10': fc10_up
                }
                name_save = save_path + str(epoch) + '.npy'
                np.save(name_save, network_param)

    def save_data(self, iter, data_tuple, tuple_path):
        tuple_path_now = tuple_path + 'AirSim' + str(iter) + '.npy'

        fileList = os.listdir(tuple_path)
        for fileName in fileList:
            os.remove(tuple_path + "/" + fileName)

        tuple_path_now = tuple_path + '/' + 'AirSim' + str(iter) + '.npy'
        np.save(tuple_path_now, data_tuple)

    def load_network(self, epoch, save_path):
        self.saver.restore(self.sess_det, save_path)


    def return_plot(self, ret, epi, level, mem_per, iter):
        if level==0:
            plot_ret = self.summary1
        elif level==1:
            plot_ret = self.summary2
        elif level==2:
            plot_ret = self.summary3
        elif level==3:
            plot_ret = self.summary4


        summary = self.session_ret.run([plot_ret, self.summary5], {self.ret: ret, self.mem_per: mem_per})
        self.test_writer_ret.add_summary(summary[0], epi)
        self.test_writer_ret.add_summary(summary[1], iter)
        self. test_writer_ret.flush()

    def first_conv(self):
        return self.conv1W

    # def load_network_from_npy(self, load_path, dtype):
    #     train_type = True
    #     net_data = np.load(open(load_path, "rb"), encoding="latin1").item()
    #     self.conv1W = tf.Variable(net_data["conv1"][0], trainable=train_type, dtype=dtype)
    #     self.conv1b = tf.Variable(net_data["conv1"][1], trainable=train_type, dtype=dtype)
    #
    #     self.conv2W = tf.Variable(net_data["conv2"][0], trainable=train_type, dtype=dtype)
    #     self.conv2b = tf.Variable(net_data["conv2"][1], trainable=train_type, dtype=dtype)
    #
    #     self.conv3W = tf.Variable(net_data["conv3"][0], trainable=train_type, dtype=dtype)
    #     self.conv3b = tf.Variable(net_data["conv3"][1], trainable=train_type, dtype=dtype)
    #
    #     self.conv4W = tf.Variable(net_data["conv4"][0], trainable=train_type, dtype=dtype)
    #     self.conv4b = tf.Variable(net_data["conv4"][1], trainable=train_type, dtype=dtype)
    #
    #     self.conv5W = tf.Variable(net_data["conv5"][0], trainable=train_type, dtype=dtype)
    #     self.conv5b = tf.Variable(net_data["conv5"][1], trainable=train_type, dtype=dtype)
    #
    #     self.fc6W = tf.Variable(net_data["fc6"][0], trainable=train_type, dtype=dtype)
    #     self.fc6b = tf.Variable(net_data["fc6"][1], trainable=train_type, dtype=dtype)
    #
    #     self.fc7W = tf.Variable(net_data["fc7"][0])
    #     self.fc7b = tf.Variable(net_data["fc7"][1])
    #
    #     self.fc8W = tf.Variable(net_data["fc8"][0])
    #     self.fc8b = tf.Variable(net_data["fc8"][1])
    #
    #     self.fc9W = tf.Variable(net_data["fc9"][0])
    #     self.fc9b = tf.Variable(net_data["fc9"][1])
    #
    #     self.fc10W = tf.Variable(net_data["fc10"][0])
    #     self.fc10b = tf.Variable(net_data["fc10"][1])
    #
    #
    #     self.sess_det.run(tf.global_variables_initializer())

    # def load_network_from_npy(self, load_path, dtype):
    #     train_type = True
    #     net_data = np.load(open(load_path, "rb"), encoding="latin1").item()
    #     self.conv1W.assign(net_data["conv1"][0])
    #     self.conv1b.assign(net_data["conv1"][1])
    #
    #     self.conv2W.assign(net_data["conv2"][0])
    #     self.conv2b.assign(net_data["conv2"][1])
    #
    #     self.conv3W.assign(net_data["conv3"][0])
    #     self.conv3b.assign(net_data["conv3"][1])
    #
    #     self.conv4W.assign(net_data["conv4"][0])
    #     self.conv4b.assign(net_data["conv4"][1])
    #
    #     self.conv5W.assign(net_data["conv5"][0])
    #     self.conv5b.assign(net_data["conv5"][1])
    #
    #     self.fc6W.assign(net_data["fc6"][0])
    #     self.fc6b.assign(net_data["fc6"][1])
    #
    #     self.fc7W.assign(net_data["fc7"][0])
    #     self.fc7b.assign(net_data["fc7"][1])
    #
    #     self.fc8W.assign(net_data["fc8"][0])
    #     self.fc8b.assign(net_data["fc8"][1])
    #
    #     self.fc9W.assign(net_data["fc9"][0])
    #     self.fc9b.assign(net_data["fc9"][1])
    #
    #     self.fc10W.assign(net_data["fc10"][0])
    #     self.fc10b.assign(net_data["fc10"][1])
    #
    #     self.sess_det.run(self.conv1W)

class DeepAgentRLrollout(object):

    def __init__(self, lr_const, weights_path, num_actions, train_type, network_type, regularization, input_size, dtype=np.float32):
        # weights_path = Defines initialization
        #                    Imagenet, Random
        # train_type = Defines the what layers of the network needs to be trained
        # network_type = Defines network topology
        #                    Alexnet, Reduced_alexnet, date
        # regularization = Should the loss function contain regularization loss
        self.pred_action_count=0
        with tf.device('/device:GPU:0'):
            self.g2 = tf.Graph()
            with self.g2.as_default():
                self.y_true = tf.placeholder(dtype, shape=[None, num_actions], name='y_true')
                # self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')

                self.NN = AlexNet(num_actions, weights_path, train_type,network_type, regularization, input_size,dtype,  arch_type='alexnet')
                # self.NN = VGG16(num_actions, weights_path, train_type,network_type, regularization, arch_type='vgg16')

                q_t = self.NN.fc10
                # self.y_pred = tf.nn.softmax(q_t, name='y_pred')

                self.init = tf.global_variables_initializer()
                config = tf.ConfigProto(allow_soft_placement=True)
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.9

                # sess_depth = tf.Session(config=tf.ConfigProto(allow_soft_placement=True), graph=g1)

                self.sess_det = tf.Session(config=config, graph=self.g2)
                self.sess_det.run(self.init)

                # q_summary=[]
                # avg_q = tf.reduce_mean(q_t, 0)
                # for idx in range(0, num_actions):
                #     q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
                # self.q_summary = tf.summary.merge(q_summary, 'q_summary')
                # c = tf.range(tf.shape(self.y_true)[0]) * tf.shape(self.y_true)[1]
                # ind = c + self.actions
                # onehot_predictions = tf.gather(tf.reshape(self.y_true, [-1]), ind)
                #
                # ind = tf.range(tf.shape(q_t)[0]) * tf.shape(q_t)[1] + self.actions
                # onehot_predictions_q_t = tf.gather(tf.reshape(q_t, [-1]), ind)


                # # MSE Loss
                cost = tf.squared_difference(q_t, self.y_true)
                cost1 = tf.reduce_mean(cost)

                # Huber Loss
                # err = self.y_true - onehot_predictions_q_t
                # loss_v = tf.where(tf.abs(err) < 1.0,
                #                   0.5 * tf.square(err),
                #                   tf.abs(err) - 0.5)
                # cost1 = tf.reduce_sum(loss_v)

                # cost1 = tf.reduce_mean(tf.square(q_t - self.y_true), name="Loss")
                # cost1 = tf.reduce_mean(tf.square(), name='Loss')
                if regularization:
                    regularizers_w = tf.nn.l2_loss(self.NN.fc6W) + tf.nn.l2_loss(self.NN.fc7W) + tf.nn.l2_loss(
                        self.NN.fc8W) + tf.nn.l2_loss(self.NN.fc9W)+ tf.nn.l2_loss(self.NN.fc10W)+ tf.nn.l2_loss(self.NN.conv1W) + tf.nn.l2_loss(self.NN.conv2W) + tf.nn.l2_loss(
                        self.NN.conv3W) + tf.nn.l2_loss(self.NN.conv4W) + tf.nn.l2_loss(self.NN.conv5W)

                    regularizers_b = tf.nn.l2_loss(self.NN.fc6b) + tf.nn.l2_loss(self.NN.fc7b) + tf.nn.l2_loss(
                        self.NN.fc8b) + tf.nn.l2_loss(self.NN.fc9b) + tf.nn.l2_loss(self.NN.fc10b) + tf.nn.l2_loss(
                        self.NN.conv1b) + tf.nn.l2_loss(self.NN.conv2b) + tf.nn.l2_loss(
                        self.NN.conv3b) + tf.nn.l2_loss(self.NN.conv4b) + tf.nn.l2_loss(self.NN.conv5b)

                    regularizers = regularizers_b+regularizers_w
                    beta = 0.00001
                    # beta=0
                    cost = tf.reduce_mean(cost1 + beta * regularizers)
                else:
                    cost=cost1
                tf.summary.scalar('Loss', cost)
                # clip_error=False
                # if clip_error:
                #     cost = tf.clip_by_value(cost, -1, 1)



                step = tf.Variable(0, trainable=False)
                # learning_rate_start = tf.placeholder(tf.float32, shape=(), name='learning_rate_start')
                # decay_rate = tf.placeholder(tf.float32, shape=(), name="decay_rate")
                # End-End
                # learning_rate_start = tf.constant(lr_const)
                # decay_rate = 0.9992
                # rate_decay = tf.train.exponential_decay(learning_rate_start, step, 1, decay_rate)
                self.rate = tf.placeholder(tf.float32, shape=[])
                # decay_rate = 0.99995
                # rate_decay = tf.train.exponential_decay(self.rate, step, 1, decay_rate)
                rate_decay = self.rate
                tf.summary.scalar('Learning Rate', rate_decay)

                # rate=learning_rate_start
                self.optimizer = tf.train.AdamOptimizer(learning_rate=rate_decay).minimize(cost, global_step=step,
                                                                                name="optimizer")

                # optimizer = tf.train.AdamOptimizer(learning_rate=5e-8).minimize(cost)
                # correct_prediction = tf.equal(y_pred_cls, y_true_cls, name="correct_prediction")
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
                # tf.summary.scalar('Validation Accuracy', accuracy)
                self.merged = tf.summary.merge_all()
                self.test_writer = tf.summary.FileWriter('D:/train/loss')
                # tf.summary.FileWriter('D:/train/loss', self.sess_det.graph)
                self.sess_det.run(tf.global_variables_initializer())
                # y_test_images = np.zeros((1, 3))
                self.saver = tf.train.Saver()

                self.sess_det.graph.finalize()

            self.g_ret = tf.Graph()

            # with self.g_ret.as_default():
            #     self.test_writer_ret = tf.summary.FileWriter('D:/train/return_plot')
            #
            #     self.ret = tf.Variable(0.0)
            #     tf.summary.scalar("Return", self.ret)
            #
            #     self.write_op_ret = tf.summary.merge_all()
            #     self.session_ret = tf.Session(config=config, graph=self.g_ret)
            #     self.session_ret.run(tf.global_variables_initializer())

            with self.g_ret.as_default():
                self.test_writer_ret = tf.summary.FileWriter('D:/train/return_plot')

                self.ret = tf.placeholder(tf.float32, shape=())
                self.mem_per = tf.placeholder(tf.float32, shape=())
                self.summary1 = tf.summary.scalar("Return1", self.ret)
                self.summary2 = tf.summary.scalar("Return2", self.ret)
                self.summary3 = tf.summary.scalar("Return3", self.ret)
                self.summary4 = tf.summary.scalar("Return4", self.ret)

                self.summary5 = tf.summary.scalar("Memory Used(GB)", self.mem_per)

                # self.write_op_ret = tf.summary.merge_all()
                self.session_ret = tf.Session(config=config, graph=self.g_ret)
                self.session_ret.run(tf.global_variables_initializer())

                self.session_ret.graph.finalize()


    def action_selection(self, state):

        qvals = self.sess_det.run(self.NN.fc10, feed_dict={self.NN.x: state})
        # print(qvals)
        if qvals.shape[0]>1:
            # Evaluating batch
            action = np.argmax(qvals, axis=1)
        else:
            # Evaluating one sample
            self.pred_action_count = self.pred_action_count+1
            action = np.zeros(1)
            action[0]=np.argmax(qvals)
            # Find mean and std of Q VALUE AND REPORT THROUGH TENSORBOARD
            summary = tf.Summary()
            summary.value.add(tag='Mean Q', simple_value=np.mean(qvals))
            summary.value.add(tag='Std Q', simple_value=np.std(qvals))
            self.test_writer_ret.add_summary(summary, self.pred_action_count)
        return action.astype(int), q

    def Q_val(self, state):
        qvals = self.sess_det.run(self.NN.fc10, feed_dict={self.NN.x: state})
        return qvals

    def train(self, old_states, Qvals, lr, i):
        summary = self.sess_det.run([self.optimizer, self.merged], feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
        # summary = self.sess_det.run(self.merged, feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
        self.test_writer.add_summary(summary[1],i)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata=tf.RunMetadata()
        # self.test_writer.add_run_metadata(run_metadata, 'step%d', i)

        # self.test_writer.add_graph(self.sess_det.graph)

    # def train(self, old_states, Qvals,i, lr):
    #     summary = self.sess_det.run([self.optimizer, self.merged], feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
    #     # summary = self.sess_det.run(self.merged, feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
    #     self.test_writer.add_summary(summary[1], i)
    #
    #     # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     # run_metadata=tf.RunMetadata()
    #     # self.test_writer.add_run_metadata(run_metadata, 'step%d', i)
    #
    #     # self.test_writer.add_graph(self.sess_det.graph)
    def save_network(self, epoch, save_path, type):

        with self.sess_det.as_default():
            if type=='save_critic':
                self.saver.save(self.sess_det, save_path)
            else:
                conv1_up = [self.NN.conv1W.eval(), self.NN.conv1b.eval()]
                conv2_up = [self.NN.conv2W.eval(), self.NN.conv2b.eval()]
                conv3_up = [self.NN.conv3W.eval(), self.NN.conv3b.eval()]
                conv4_up = [self.NN.conv4W.eval(), self.NN.conv4b.eval()]
                conv5_up = [self.NN.conv5W.eval(), self.NN.conv5b.eval()]
                fc6_up = [self.NN.fc6W.eval(), self.NN.fc6b.eval()]
                fc7_up = [self.NN.fc7W.eval(), self.NN.fc7b.eval()]
                fc8_up = [self.NN.fc8W.eval(), self.NN.fc8b.eval()]
                fc9_up = [self.NN.fc9W.eval(), self.NN.fc9b.eval()]
                fc10_up = [self.NN.fc10W.eval(), self.NN.fc10b.eval()]
                network_param = {
                    'conv1': conv1_up,
                    'conv2': conv2_up,
                    'conv3': conv3_up,
                    'conv4': conv4_up,
                    'conv5': conv5_up,
                    'fc6': fc6_up,
                    'fc7': fc7_up,
                    'fc8': fc8_up,
                    'fc9': fc9_up,
                    'fc10': fc10_up
                }
                name_save = save_path + str(epoch) + '.npy'
                np.save(name_save, network_param)

    def save_data(self, iter, data_tuple, tuple_path):
        tuple_path_now = tuple_path + 'AirSim' + str(iter) + '.npy'

        fileList = os.listdir(tuple_path)
        for fileName in fileList:
            os.remove(tuple_path + "/" + fileName)

        tuple_path_now = tuple_path + '/' + 'AirSim' + str(iter) + '.npy'
        np.save(tuple_path_now, data_tuple)

    def load_network(self, epoch, save_path):
        self.saver.restore(self.sess_det, save_path)


    def return_plot(self, ret, epi, level, mem_per, iter):
        if level==0:
            plot_ret = self.summary1
        elif level==1:
            plot_ret = self.summary2
        elif level==2:
            plot_ret = self.summary3
        elif level==3:
            plot_ret = self.summary4


        summary = self.session_ret.run([plot_ret, self.summary5], {self.ret: ret, self.mem_per: mem_per})
        self.test_writer_ret.add_summary(summary[0], epi)
        self.test_writer_ret.add_summary(summary[1], iter)
        self. test_writer_ret.flush()

    def first_conv(self):
        return self.conv1W





################################ DFA code #####################################################

class DeepAgentDFA(object):

    def __init__(self, lr_const, weights_path, num_actions, train_type, network_type, regularization, input_size, dtype=np.float32):
        # weights_path = Defines initialization
        #                    Imagenet, Random
        # train_type = Defines the what layers of the network needs to be trained
        # network_type = Defines network topology
        #                    Alexnet, Reduced_alexnet, date
        # regularization = Should the loss function contain regularization loss
        self.pred_action_count=0
        with tf.device('/device:GPU:0'):

            # Define graph and corresponding session
            self.g2 = tf.Graph()
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            self.sess_det = tf.Session(config=config, graph=self.g2)

            # Add elements to graph
            with self.g2.as_default():
                self.y_true = tf.placeholder(dtype, shape=[None], name='y_true')
                self.actions = tf.placeholder(tf.int32, shape=[None], name='actions')

                l0 = Convolution(input_sizes=[None, 28, 28, 1], filter_sizes=[3, 3, 1, 32], num_classes=10,
                                 init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate,
                                 activation=Tanh(), bias=bias, last_layer=False, name='conv1', load=weights_conv,
                                 train=train_conv)
                l1 = FeedbackConv(size=[batch_size, 28, 28, 32], num_classes=10, sparse=args.sparse, rank=args.rank,
                                  name='conv1_fb')

                l2 = Convolution(input_sizes=[batch_size, 28, 28, 32], filter_sizes=[3, 3, 32, 64], num_classes=10,
                                 init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate,
                                 activation=Tanh(), bias=bias, last_layer=False, name='conv2', load=weights_conv,
                                 train=train_conv)
                l3 = MaxPool(size=[batch_size, 28, 28, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
                l4 = Dropout(rate=dropout_rate / 2.)
                l5 = FeedbackConv(size=[batch_size, 14, 14, 64], num_classes=10, sparse=args.sparse, rank=args.rank,
                                  name='conv2_fb')

                l6 = ConvToFullyConnected(shape=[14, 14, 64])
                l7 = FullyConnected(size=[14 * 14 * 64, 128], num_classes=10, init_weights=args.init,
                                    alpha=learning_rate, activation=Tanh(), bias=bias, last_layer=False, name='fc1',
                                    load=weights_fc, train=train_fc)
                l8 = Dropout(rate=dropout_rate)
                l9 = FeedbackFC(size=[14 * 14 * 64, 128], num_classes=10, sparse=args.sparse, rank=args.rank,
                                name='fc1_fb')

                l10 = FullyConnected(size=[128, 10], num_classes=10, init_weights=args.init, alpha=learning_rate,
                                     activation=Linear(), bias=bias, last_layer=True, name='fc2', load=weights_fc,
                                     train=train_fc)

                model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10])
                q_t = self.NN.fc10
                # self.y_pred = tf.nn.softmax(q_t, name='y_pred')

                self.init = tf.global_variables_initializer()
                self.sess_det.run(self.init)

                q_summary=[]
                avg_q = tf.reduce_mean(q_t, 0)
                # for idx in range(0, num_actions):
                #     q_summary.append(tf.summary.histogram('q/%s' % idx, avg_q[idx]))
                # self.q_summary = tf.summary.merge(q_summary, 'q_summary')
                # c = tf.range(tf.shape(self.y_true)[0]) * tf.shape(self.y_true)[1]
                # ind = c + self.actions
                # onehot_predictions = tf.gather(tf.reshape(self.y_true, [-1]), ind)

                ind = tf.range(tf.shape(q_t)[0]) * tf.shape(q_t)[1] + self.actions
                onehot_predictions_q_t = tf.gather(tf.reshape(q_t, [-1]), ind)


                # # MSE Loss
                # cost = tf.squared_difference(onehot_predictions_q_t, self.y_true)
                # cost1 = tf.reduce_mean(cost)

                # Huber Loss
                err = self.y_true - onehot_predictions_q_t
                loss_v = tf.where(tf.abs(err) < 1.0,
                                  0.5 * tf.square(err),
                                  tf.abs(err) - 0.5)
                cost1 = tf.reduce_sum(loss_v)

                # cost1 = tf.reduce_mean(tf.square(q_t - self.y_true), name="Loss")
                # cost1 = tf.reduce_mean(tf.square(), name='Loss')
                if regularization:
                    regularizers_w = tf.nn.l2_loss(self.NN.fc6W) + tf.nn.l2_loss(self.NN.fc7W) + tf.nn.l2_loss(
                        self.NN.fc8W) + tf.nn.l2_loss(self.NN.fc9W)+ tf.nn.l2_loss(self.NN.fc10W)+ tf.nn.l2_loss(self.NN.conv1W) + tf.nn.l2_loss(self.NN.conv2W) + tf.nn.l2_loss(
                        self.NN.conv3W) + tf.nn.l2_loss(self.NN.conv4W) + tf.nn.l2_loss(self.NN.conv5W)

                    regularizers_b = tf.nn.l2_loss(self.NN.fc6b) + tf.nn.l2_loss(self.NN.fc7b) + tf.nn.l2_loss(
                        self.NN.fc8b) + tf.nn.l2_loss(self.NN.fc9b) + tf.nn.l2_loss(self.NN.fc10b) + tf.nn.l2_loss(
                        self.NN.conv1b) + tf.nn.l2_loss(self.NN.conv2b) + tf.nn.l2_loss(
                        self.NN.conv3b) + tf.nn.l2_loss(self.NN.conv4b) + tf.nn.l2_loss(self.NN.conv5b)

                    regularizers = regularizers_b+regularizers_w
                    beta = 0.00001
                    # beta=0
                    cost = tf.reduce_mean(cost1 + beta * regularizers)
                else:
                    cost=cost1
                tf.summary.scalar('Loss', cost)

                # clip_error=False
                # if clip_error:
                #     cost = tf.clip_by_value(cost, -1, 1)



                step = tf.Variable(0, trainable=False)
                # learning_rate_start = tf.placeholder(tf.float32, shape=(), name='learning_rate_start')
                # decay_rate = tf.placeholder(tf.float32, shape=(), name="decay_rate")
                # End-End
                # learning_rate_start = tf.constant(lr_const)
                # decay_rate = 0.9992
                # rate_decay = tf.train.exponential_decay(learning_rate_start, step, 1, decay_rate)
                self.rate = tf.placeholder(tf.float32, shape=[])
                # decay_rate = 0.99995
                # rate_decay = tf.train.exponential_decay(self.rate, step, 1, decay_rate)
                rate_decay = self.rate
                tf.summary.scalar('Learning Rate', rate_decay)

                # rate=learning_rate_start
                self.optimizer = tf.train.AdamOptimizer(learning_rate=rate_decay).minimize(cost, global_step=step,
                                                                                name="optimizer")

                # optimizer = tf.train.AdamOptimizer(learning_rate=5e-8).minimize(cost)
                # correct_prediction = tf.equal(y_pred_cls, y_true_cls, name="correct_prediction")
                # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
                # tf.summary.scalar('Validation Accuracy', accuracy)
                self.merged = tf.summary.merge_all()
                self.test_writer = tf.summary.FileWriter('D:/train/loss')
                # tf.summary.FileWriter('D:/train/loss', self.sess_det.graph)
                self.sess_det.run(tf.global_variables_initializer())
                # y_test_images = np.zeros((1, 3))
                self.saver = tf.train.Saver()

                self.sess_det.graph.finalize()

            self.g_ret = tf.Graph()

            # with self.g_ret.as_default():
            #     self.test_writer_ret = tf.summary.FileWriter('D:/train/return_plot')
            #
            #     self.ret = tf.Variable(0.0)
            #     tf.summary.scalar("Return", self.ret)
            #
            #     self.write_op_ret = tf.summary.merge_all()
            #     self.session_ret = tf.Session(config=config, graph=self.g_ret)
            #     self.session_ret.run(tf.global_variables_initializer())

            with self.g_ret.as_default():
                self.test_writer_ret = tf.summary.FileWriter('D:/train/return_plot')

                self.ret = tf.placeholder(tf.float32, shape=())
                self.mem_per = tf.placeholder(tf.float32, shape=())
                self.summary1 = tf.summary.scalar("Return1", self.ret)
                self.summary2 = tf.summary.scalar("Return2", self.ret)
                self.summary3 = tf.summary.scalar("Return3", self.ret)
                self.summary4 = tf.summary.scalar("Return4", self.ret)

                self.summary5 = tf.summary.scalar("Memory Used(GB)", self.mem_per)

                # self.write_op_ret = tf.summary.merge_all()
                self.session_ret = tf.Session(config=config, graph=self.g_ret)
                self.session_ret.run(tf.global_variables_initializer())

                self.session_ret.graph.finalize()


    def action_selection(self, state):

        qvals = self.sess_det.run(self.NN.fc10, feed_dict={self.NN.x: state})
        # print(qvals)
        if qvals.shape[0]>1:
            # Evaluating batch
            action = np.argmax(qvals, axis=1)
        else:
            # Evaluating one sample
            self.pred_action_count = self.pred_action_count+1
            action = np.zeros(1)
            action[0]=np.argmax(qvals)
            # Find mean and std of Q VALUE AND REPORT THROUGH TENSORBOARD
            summary = tf.Summary()
            summary.value.add(tag='Mean Q', simple_value=np.max(qvals))
            summary.value.add(tag='Std Q', simple_value=np.std(qvals))
            self.test_writer_ret.add_summary(summary, self.pred_action_count)
        return action.astype(int), qvals

    def Q_val(self, state):
        qvals = self.sess_det.run(self.NN.fc10, feed_dict={self.NN.x: state})
        return qvals

    def train(self, old_states, Qvals,i, lr, actions, epsilon):
        summary = self.sess_det.run([self.optimizer, self.merged], feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr, self.actions: actions})
        # summary = self.sess_det.run(self.merged, feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
        self.test_writer.add_summary(summary[1], i)
        summary = tf.Summary()
        summary.value.add(tag='Epsilon', simple_value=epsilon)
        self.test_writer_ret.add_summary(summary, i)
        # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        # run_metadata=tf.RunMetadata()
        # self.test_writer.add_run_metadata(run_metadata, 'step%d', i)

        # self.test_writer.add_graph(self.sess_det.graph)

    # def train(self, old_states, Qvals,i, lr):
    #     summary = self.sess_det.run([self.optimizer, self.merged], feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
    #     # summary = self.sess_det.run(self.merged, feed_dict={self.NN.x: old_states, self.y_true: Qvals, self.rate:lr})
    #     self.test_writer.add_summary(summary[1], i)
    #
    #     # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #     # run_metadata=tf.RunMetadata()
    #     # self.test_writer.add_run_metadata(run_metadata, 'step%d', i)
    #
    #     # self.test_writer.add_graph(self.sess_det.graph)
    def save_network(self, epoch, save_path, type):

        with self.sess_det.as_default():
            if type=='save_critic':
                self.saver.save(self.sess_det, save_path)
            else:
                conv1_up = [self.NN.conv1W.eval(), self.NN.conv1b.eval()]
                conv2_up = [self.NN.conv2W.eval(), self.NN.conv2b.eval()]
                conv3_up = [self.NN.conv3W.eval(), self.NN.conv3b.eval()]
                conv4_up = [self.NN.conv4W.eval(), self.NN.conv4b.eval()]
                conv5_up = [self.NN.conv5W.eval(), self.NN.conv5b.eval()]
                fc6_up = [self.NN.fc6W.eval(), self.NN.fc6b.eval()]
                fc7_up = [self.NN.fc7W.eval(), self.NN.fc7b.eval()]
                fc8_up = [self.NN.fc8W.eval(), self.NN.fc8b.eval()]
                fc9_up = [self.NN.fc9W.eval(), self.NN.fc9b.eval()]
                fc10_up = [self.NN.fc10W.eval(), self.NN.fc10b.eval()]
                network_param = {
                    'conv1': conv1_up,
                    'conv2': conv2_up,
                    'conv3': conv3_up,
                    'conv4': conv4_up,
                    'conv5': conv5_up,
                    'fc6': fc6_up,
                    'fc7': fc7_up,
                    'fc8': fc8_up,
                    'fc9': fc9_up,
                    'fc10': fc10_up
                }
                name_save = save_path + str(epoch) + '.npy'
                np.save(name_save, network_param)

    def save_data(self, iter, data_tuple, tuple_path):
        tuple_path_now = tuple_path + 'AirSim' + str(iter) + '.npy'

        fileList = os.listdir(tuple_path)
        for fileName in fileList:
            os.remove(tuple_path + "/" + fileName)

        tuple_path_now = tuple_path + '/' + 'AirSim' + str(iter) + '.npy'
        np.save(tuple_path_now, data_tuple)

    def load_network(self, epoch, save_path):
        self.saver.restore(self.sess_det, save_path)


    def return_plot(self, ret, epi, level, mem_per, iter):
        if level==0:
            plot_ret = self.summary1
        elif level==1:
            plot_ret = self.summary2
        elif level==2:
            plot_ret = self.summary3
        elif level==3:
            plot_ret = self.summary4


        summary = self.session_ret.run([plot_ret, self.summary5], {self.ret: ret, self.mem_per: mem_per})
        self.test_writer_ret.add_summary(summary[0], epi)
        self.test_writer_ret.add_summary(summary[1], iter)
        self. test_writer_ret.flush()

    def first_conv(self):
        return self.conv1W