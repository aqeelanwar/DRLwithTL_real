import tensorflow as tf
import numpy as np
import os


class AlexNet(object):
    def __init__(self, num_actions, weights_path, train_type,network_type, regularization, input_size,dtype,  arch_type='alexnet'):
        # def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
        #     '''From https://github.com/ethereon/caffe-tensorflow
        #     '''
        #     c_i = input.get_shape()[-1]
        #     assert c_i % group == 0
        #     assert c_o % group == 0
        #     convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        #
        #     if group == 1:
        #         conv = convolve(input, kernel)
        #     else:
        #         input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
        #         kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
        #         output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
        #         conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
        #     return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

        # num_classes = num_actions

        # y_true_cls = tf.argmax(self.y_true, axis=1)

        trained_weights_path_end_end = weights_path
        # trained_weights_path_end_end = 'models/cmu/Qlearn/offline/End-End/Feb16_End-End_B1.npy'
        # train_type = True
        # print('Train type: ', train_type)

        # define what layers need to be trained based on train_type
        train_conv_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True
        train_fc10 = True

        if train_type == 'last4':
            train_conv_fc6 = False

        elif train_type == 'last3':
            train_conv_fc6 = False
            train_fc7 = False

        elif train_type == 'last2':
            train_conv_fc6 = False
            train_fc7 = False
            train_fc8 = False

        if arch_type == 'alexnet':
            if os.path.exists(weights_path):
                # print("Loading the learned weights...")
                net_data = np.load(open(weights_path, "rb"), encoding="latin1").item()
                conv1W = net_data["conv1"][0]
                conv1b = net_data["conv1"][1]

                conv2W = net_data["conv1"][0]
                conv2b = net_data["conv1"][1]

                conv3W = net_data["conv1"][0]
                conv3b = net_data["conv1"][1]

                conv4W = net_data["conv1"][0]
                conv4b = net_data["conv1"][1]

                conv5W = net_data["conv1"][0]
                conv5b = net_data["conv1"][1]

                self.conv1W = tf.Variable(net_data["conv1"][0], trainable=train_type, dtype=dtype)
                self.conv1b = tf.Variable(net_data["conv1"][1], trainable=train_type, dtype=dtype)

                self.conv2W = tf.Variable(net_data["conv2"][0], trainable=train_type, dtype=dtype)
                self.conv2b = tf.Variable(net_data["conv2"][1], trainable=train_type, dtype=dtype)

                self.conv3W = tf.Variable(net_data["conv3"][0], trainable=train_type, dtype=dtype)
                self.conv3b = tf.Variable(net_data["conv3"][1], trainable=train_type, dtype=dtype)

                self.conv4W = tf.Variable(net_data["conv4"][0], trainable=train_type, dtype=dtype)
                self.conv4b = tf.Variable(net_data["conv4"][1], trainable=train_type, dtype=dtype)

                self.conv5W = tf.Variable(net_data["conv5"][0], trainable=train_type, dtype=dtype)
                self.conv5b = tf.Variable(net_data["conv5"][1], trainable=train_type, dtype=dtype)

                self.fc6W = tf.Variable(net_data["fc6"][0], trainable=train_type, dtype=dtype)
                self.fc6b = tf.Variable(net_data["fc6"][1], trainable=train_type, dtype=dtype)

                self.fc7W = tf.Variable(net_data["fc7"][0], trainable=train_type, dtype=dtype)
                self.fc7b = tf.Variable(net_data["fc7"][1], trainable=train_type, dtype=dtype)

                self.fc8W = tf.Variable(net_data["fc8"][0], trainable=train_type, dtype=dtype)
                self.fc8b = tf.Variable(net_data["fc8"][1], trainable=train_type, dtype=dtype)

                self.fc9W = tf.Variable(net_data["fc9"][0], trainable=train_type, dtype=dtype)
                self.fc9b = tf.Variable(net_data["fc9"][1], trainable=train_type, dtype=dtype)

                self.fc10W = tf.Variable(net_data["fc10"][0], trainable=train_type, dtype=dtype)
                self.fc10b = tf.Variable(net_data["fc10"][1], trainable=train_type, dtype=dtype)

                # # self.fc7W = tf.Variable(net_data["fc7"][0])
                # # self.fc7b = tf.Variable(net_data["fc7"][1])
                # #
                # # self.fc8W = tf.Variable(net_data["fc8"][0])
                # # self.fc8b = tf.Variable(net_data["fc8"][1])
                #
                # self.fc7W = tf.Variable(tf.truncated_normal(shape=(4096, 4096), stddev=0.05))
                # self.fc7b = tf.Variable(tf.truncated_normal(shape=[4096], stddev=0.05))
                #
                # self.fc8W = tf.Variable(tf.truncated_normal(shape=(4096, num_actions), stddev=0.05))
                # self.fc8b = tf.Variable(tf.constant(0.005, shape=[num_actions]))

            elif weights_path == 'Random':
                # print('Loading random weights')
                self.conv1W = tf.Variable(tf.truncated_normal(shape=(11, 11, 3, 96)), trainable=train_conv_fc6)
                self.conv1b = tf.Variable(tf.truncated_normal(shape=[96]), trainable=train_conv_fc6)

                self.conv2W = tf.Variable(tf.truncated_normal(shape=(5, 5, 96, 256)), trainable=train_conv_fc6)
                self.conv2b = tf.Variable(tf.truncated_normal(shape=[256]), trainable=train_conv_fc6)

                self.conv3W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 384)), trainable=train_conv_fc6)
                self.conv3b = tf.Variable(tf.truncated_normal(shape=[384]), trainable=train_conv_fc6)

                self.conv4W = tf.Variable(tf.truncated_normal(shape=(3, 3, 384, 384)), trainable=train_conv_fc6)
                self.conv4b = tf.Variable(tf.truncated_normal(shape=[384]), trainable=train_conv_fc6)

                self.conv5W = tf.Variable(tf.truncated_normal(shape=(3, 3, 384, 256)), trainable=train_conv_fc6)
                self.conv5b = tf.Variable(tf.truncated_normal(shape=[256]), trainable=train_conv_fc6)

                self.fc6W = tf.Variable(tf.truncated_normal(shape=(9216, 4096), stddev=0.05), trainable=train_conv_fc6)
                self.fc6b = tf.Variable(tf.truncated_normal(shape=[4096], stddev=0.05), trainable=train_conv_fc6)

                self.fc7W = tf.Variable(tf.truncated_normal(shape=(4096, 2048), stddev=0.05), trainable=train_fc7)
                self.fc7b = tf.Variable(tf.truncated_normal(shape=[2048], stddev=0.05), trainable=train_fc7)

                self.fc8W = tf.Variable(tf.truncated_normal(shape=(2048, 2048), stddev=0.05), trainable=train_fc8)
                self.fc8b = tf.Variable(tf.truncated_normal( shape=[2048]), trainable=train_fc8)

                self.fc9W = tf.Variable(tf.truncated_normal(shape=(2048, 1024), stddev=0.05), trainable=train_fc9)
                self.fc9b = tf.Variable(tf.truncated_normal(shape=[1024], stddev=0.05), trainable=train_fc9)

                self.fc10W = tf.Variable(tf.truncated_normal(shape=(1024, num_actions), stddev=0.05),
                                         trainable=train_fc10)
                self.fc10b = tf.Variable(tf.truncated_normal(shape=[num_actions], stddev=0.05), trainable=train_fc10)

            elif weights_path == 'Imagenet':
                # Imagenet can either be alexnet or date network
                # Alexnet 2 - 128 instead of 256
                # print('Loading Imagenet weights')
                # net_data = np.load(open("models/imagenet_weights.npy", "rb"), encoding="latin1").item()

                net_data = np.load(open("models/bvlc_alexnet.npy", "rb"), encoding="latin1").item()
                self.conv1W = tf.Variable(net_data["conv1"][0], trainable=train_conv_fc6)
                self.conv1b = tf.Variable(net_data["conv1"][1], trainable=train_conv_fc6)


                # The weights are from the grouped Alexnet. Ungrouping and copying the weights
                c2w = np.concatenate((net_data["conv2"][0],net_data["conv2"][0]), axis=2)

                self.conv2W = tf.Variable(c2w, trainable=train_conv_fc6)
                self.conv2b = tf.Variable(net_data["conv2"][1], trainable=train_conv_fc6)

                self.conv3W = tf.Variable(net_data["conv3"][0], trainable=train_conv_fc6)
                self.conv3b = tf.Variable(net_data["conv3"][1], trainable=train_conv_fc6)

                c4w = np.concatenate((net_data["conv4"][0], net_data["conv4"][0]), axis=2)
                self.conv4W = tf.Variable(c4w, trainable=train_conv_fc6)
                self.conv4b = tf.Variable(net_data["conv4"][1], trainable=train_conv_fc6)

                c5w = np.concatenate((net_data["conv5"][0], net_data["conv5"][0]), axis=2)
                self.conv5W = tf.Variable(c5w, trainable=train_conv_fc6)
                self.conv5b = tf.Variable(net_data["conv5"][1], trainable=train_conv_fc6)

                self.fc6W = tf.Variable(net_data["fc6"][0], trainable=train_conv_fc6)
                self.fc6b = tf.Variable(net_data["fc6"][1], trainable=train_conv_fc6)

                if network_type == 'date':
                    # Slice 4096x2048 from 4096x4096 of Alexnet
                    w7 = net_data["fc6"][0]
                    b7 = net_data["fc6"][1]
                    w7_date = w7[0:4096, 0:2048]
                    b7_date = b7[0:2048]
                    self.fc7W = tf.Variable(w7_date, trainable=train_fc7)
                    self.fc7b = tf.Variable(b7_date, trainable=train_fc7)

                    self.fc8W = tf.Variable(tf.truncated_normal(shape=(2048, 2048), stddev=0.05), trainable=train_fc8)
                    self.fc8b = tf.Variable(tf.truncated_normal(shape=[2048], stddev=0.05), trainable=train_fc8)

                    self.fc9W = tf.Variable(tf.truncated_normal(shape=(2048, 1024), stddev=0.05), trainable=train_fc9)
                    self.fc9b = tf.Variable(tf.constant(0.005, shape=[1024]), trainable=train_fc9)

                    self.fc10W = tf.Variable(tf.truncated_normal(shape=(1024, num_actions), stddev=0.05),
                                             trainable=train_fc10)
                    self.fc10b = tf.Variable(tf.constant(0.005, shape=[num_actions]), trainable=train_fc10)

                else:

                    self.fc7W = tf.Variable(net_data["fc7"][0])
                    self.fc7b = tf.Variable(net_data["fc7"][1])

                    self.fc8W = tf.Variable(tf.truncated_normal(shape=(1024, num_actions), stddev=0.05))
                    self.fc8b = tf.Variable(tf.constant(0.005, shape=[num_actions]))

                #     self.fc6W = tf.Variable(net_data["fc6"][0], trainable=train_type)
                #     self.fc6b = tf.Variable(net_data["fc6"][1], trainable=train_type)

                # reduced_alexnet=False
                # if reduced_alexnet:
                #     w5 = net_data['conv5'][0][:,:,:,0:128]
                #     b5 = net_data['conv5'][1][0:128]
                #
                #     w6 = net_data['fc6'][0][0:4608]
                #     b6 = net_data['fc6'][1]
                #     self.conv5W = tf.Variable(w5, trainable=train_type)
                #     self.conv5b = tf.Variable(b5, trainable=train_type)
                #
                #     self.fc6W = tf.Variable(w6, trainable=train_type)
                #     self.fc6b = tf.Variable(b6, trainable=train_type)
                # else:
                #     self.conv5W = tf.Variable(net_data["conv5"][0], trainable=train_type)
                #     self.conv5b = tf.Variable(net_data["conv5"][1], trainable=train_type)
                #
                #     self.fc6W = tf.Variable(net_data["fc6"][0], trainable=train_type)
                #     self.fc6b = tf.Variable(net_data["fc6"][1], trainable=train_type)

            # Print summary of network

            # print("Weights loaded")

            train_x = np.zeros((1, input_size, input_size, 3)).astype(dtype)
            # train_y = np.zeros((1, 1000))
            xdim = train_x.shape[1:]
            # ydim = train_y.shape[1]

            self.x = tf.placeholder(dtype, (None,) + xdim)
            # normalization
            self.x = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.x)

            # conv1
            # conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
            k_h = 11
            k_w = 11
            c_o = 96
            s_h = 4
            s_w = 4
            conv1_in = tf.layers.conv2d(inputs=self.x, filters=96, strides = (4,4), kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

            conv1_in = conv(self.x, self.conv1W, self.conv1b, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1)
            conv1 = tf.nn.relu(conv1_in)

            # lrn1
            # lrn(2, 2e-05, 0.75, name='norm1')
            radius = 5
            alpha = 2e-05
            beta = 0.75
            bias = 1.0
            lrn1 = tf.nn.local_response_normalization(conv1,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

            # maxpool1
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
            k_h = 3
            k_w = 3
            s_h = 2
            s_w = 2
            padding = 'VALID'
            maxpool1 = tf.nn.max_pool(lrn1, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # conv2
            # conv(5, 5, 256, 1, 1, group=2, name='conv2')
            k_h = 5
            k_w = 5
            c_o = 256
            s_h = 1
            s_w = 1
            group = 1

            conv2_in = conv(maxpool1, self.conv2W, self.conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv2 = tf.nn.relu(conv2_in)

            # lrn2
            # lrn(2, 2e-05, 0.75, name='norm2')
            radius = 5
            alpha = 2e-05
            beta = 0.75
            bias = 1.0
            lrn2 = tf.nn.local_response_normalization(conv2,
                                                      depth_radius=radius,
                                                      alpha=alpha,
                                                      beta=beta,
                                                      bias=bias)

            # maxpool2
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            k_h = 3
            k_w = 3
            s_h = 2
            s_w = 2
            padding = 'VALID'
            maxpool2 = tf.nn.max_pool(lrn2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # conv3
            # conv(3, 3, 384, 1, 1, name='conv3')
            k_h = 3
            k_w = 3
            c_o = 384
            s_h = 1
            s_w = 1
            group = 1

            conv3_in = conv(maxpool2, self.conv3W, self.conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv3 = tf.nn.relu(conv3_in)

            # conv4
            # conv(3, 3, 384, 1, 1, group=2, name='conv4')
            k_h = 3
            k_w = 3
            c_o = 384
            s_h = 1
            s_w = 1
            group = 1

            conv4_in = conv(conv3, self.conv4W, self.conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv4 = tf.nn.relu(conv4_in)

            # conv5
            # conv(3, 3, 256, 1, 1, group=2, name='conv5')
            k_h = 3
            k_w = 3
            c_o = 256
            s_h = 1
            s_w = 1
            group = 1

            conv5_in = conv(conv4, self.conv5W, self.conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME", group=group)
            conv5 = tf.nn.relu(conv5_in)

            # maxpool5
            # max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
            k_h = 3
            k_w = 3
            s_h = 2
            s_w = 2
            padding = 'VALID'
            maxpool5 = tf.nn.max_pool(conv5, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            self.fc6 = tf.nn.relu_layer(tf.reshape(maxpool5, [-1, int(np.prod(maxpool5.get_shape()[1:]))]), self.fc6W,
                                        self.fc6b)
            self.fc7 = tf.nn.relu_layer(self.fc6, self.fc7W, self.fc7b)
            self.fc8 = tf.nn.relu_layer(self.fc7, self.fc8W, self.fc8b)
            self.fc9 = tf.nn.relu_layer(self.fc8, self.fc9W, self.fc9b)
            self.fc10 = tf.nn.xw_plus_b(self.fc9, self.fc10W, self.fc10b)

            # #Print Summary of the network
            # print('------------------ Summary of the network ------------------')
            print('Network Type: {:>14s}'.format(network_type))
            print('Train Type: {:>15s}'.format(train_type))
            print('Regularization:{:>14s}'.format(str(regularization)))
            print(' ')
            print('Below is the summary of the fully connected layers')
            print(self.fc6, train_conv_fc6)
            print(self.fc7, train_fc7)
            print(self.fc8, train_fc8)
            print(self.fc9, train_fc9)
            print(self.fc10, train_fc10)
            print('------------------------------------------------------------')
#####################################################################################################################



class AlexNetDFA:
    def __init__(self, layers: tuple):
        self.num_layers = len(layers)
        self.layers = layers

    def num_params(self):
        param_sum = 0
        for ii in range(self.num_layers):
            l = self.layers[ii]
            param_sum += l.num_params()
        return param_sum

    ####################################################################

    def train(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])
        # A[ii] contains the intermediate outputs of the layers

        # Define error here

        E = tf.nn.softmax(A[self.num_layers - 1]) - Y
        N = tf.shape(A[self.num_layers - 1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers - 1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers - 1):
                D[ii] = l.backward(A[ii - 1], A[ii], E)
                gvs = l.train(A[ii - 1], A[ii], E)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.backward(X, A[ii], D[ii + 1])
                gvs = l.train(X, A[ii], D[ii + 1])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.backward(A[ii - 1], A[ii], D[ii + 1])
                gvs = l.train(A[ii - 1], A[ii], D[ii + 1])
                grads_and_vars.extend(gvs)

        return grads_and_vars

    def dfa(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])

        E = tf.nn.softmax(A[self.num_layers - 1]) - Y
        N = tf.shape(A[self.num_layers - 1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers - 1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers - 1):
                D[ii] = l.dfa_backward(A[ii - 1], A[ii], E, E)
                gvs = l.dfa(A[ii - 1], A[ii], E, E)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.dfa_backward(X, A[ii], E, D[ii + 1])
                gvs = l.dfa(X, A[ii], E, D[ii + 1])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.dfa_backward(A[ii - 1], A[ii], E, D[ii + 1])
                gvs = l.dfa(A[ii - 1], A[ii], E, D[ii + 1])
                grads_and_vars.extend(gvs)

        return grads_and_vars

    def lel(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])

        E = tf.nn.softmax(A[self.num_layers - 1]) - Y
        N = tf.shape(A[self.num_layers - 1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers - 1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers - 1):
                D[ii] = l.lel_backward(A[ii - 1], A[ii], E, E, Y)
                gvs = l.lel(A[ii - 1], A[ii], E, E, Y)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.lel_backward(X, A[ii], E, D[ii + 1], Y)
                gvs = l.lel(X, A[ii], E, D[ii + 1], Y)
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.lel_backward(A[ii - 1], A[ii], E, D[ii + 1], Y)
                gvs = l.lel(A[ii - 1], A[ii], E, D[ii + 1], Y)
                grads_and_vars.extend(gvs)

        return grads_and_vars

    ####################################################################

    def gvs(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])

        E = tf.nn.softmax(A[self.num_layers - 1]) - Y
        N = tf.shape(A[self.num_layers - 1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers - 1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers - 1):
                D[ii] = l.backward(A[ii - 1], A[ii], E)
                gvs = l.gv(A[ii - 1], A[ii], E)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.backward(X, A[ii], D[ii + 1])
                gvs = l.gv(X, A[ii], D[ii + 1])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.backward(A[ii - 1], A[ii], D[ii + 1])
                gvs = l.gv(A[ii - 1], A[ii], D[ii + 1])
                grads_and_vars.extend(gvs)

        return grads_and_vars

    def dfa_gvs(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])

        E = tf.nn.softmax(A[self.num_layers - 1]) - Y
        N = tf.shape(A[self.num_layers - 1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers - 1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers - 1):
                D[ii] = l.dfa_backward(A[ii - 1], A[ii], E, E)
                gvs = l.dfa_gv(A[ii - 1], A[ii], E, E)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.dfa_backward(X, A[ii], E, D[ii + 1])
                gvs = l.dfa_gv(X, A[ii], E, D[ii + 1])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.dfa_backward(A[ii - 1], A[ii], E, D[ii + 1])
                gvs = l.dfa_gv(A[ii - 1], A[ii], E, D[ii + 1])
                grads_and_vars.extend(gvs)

        return grads_and_vars

    def lel_gvs(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])

        E = tf.nn.softmax(A[self.num_layers - 1]) - Y
        N = tf.shape(A[self.num_layers - 1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers - 1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers - 1):
                D[ii] = l.lel_backward(A[ii - 1], A[ii], E, E, Y)
                gvs = l.lel_gv(A[ii - 1], A[ii], E, E, Y)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.lel_backward(X, A[ii], E, D[ii + 1], Y)
                gvs = l.lel_gv(X, A[ii], E, D[ii + 1], Y)
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.lel_backward(A[ii - 1], A[ii], E, D[ii + 1], Y)
                gvs = l.lel_gv(A[ii - 1], A[ii], E, D[ii + 1], Y)
                grads_and_vars.extend(gvs)

        return grads_and_vars

    ####################################################################

    def backwards(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])

        E = tf.nn.softmax(A[self.num_layers - 1]) - Y
        N = tf.shape(A[self.num_layers - 1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers - 1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers - 1):
                D[ii] = l.backward(A[ii - 1], A[ii], E)
            elif (ii == 0):
                D[ii] = l.backward(X, A[ii], D[ii + 1])
            else:
                D[ii] = l.backward(A[ii - 1], A[ii], D[ii + 1])

        return D

    def dfa_backwards(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])

        E = tf.nn.softmax(A[self.num_layers - 1]) - Y
        N = tf.shape(A[self.num_layers - 1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers - 1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers - 1):
                D[ii] = l.dfa_backward(A[ii - 1], A[ii], E, E)
            elif (ii == 0):
                D[ii] = l.dfa_backward(X, A[ii], E, D[ii + 1])
            else:
                D[ii] = l.dfa_backward(A[ii - 1], A[ii], E, D[ii + 1])

        return D

    ####################################################################

    def predict(self, X):
        A = [None] * self.num_layers

        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])

        return A[self.num_layers - 1]

    ####################################################################

    def get_weights(self):
        weights = {}
        for ii in range(self.num_layers):
            l = self.layers[ii]
            tup = l.get_weights()
            for (key, value) in tup:
                weights[key] = value

        return weights

    def up_to(self, X, N):
        A = [None] * (N + 1)

        for ii in range(N + 1):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii - 1])

        return A[N]