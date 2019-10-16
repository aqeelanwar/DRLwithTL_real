import tensorflow as tf
import numpy as np
import os


class VGG16(object):
    def __init__(self, num_actions, weights_path, train_type,network_type, regularization, arch_type='vgg16'):
        def conv(input, kernel, biases, k_h, k_w, c_o, s_h, s_w, padding="VALID", group=1):
            '''From https://github.com/ethereon/caffe-tensorflow
            '''
            c_i = input.get_shape()[-1]
            assert c_i % group == 0
            assert c_o % group == 0
            convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)

            if group == 1:
                conv = convolve(input, kernel)
            else:
                input_groups = tf.split(input, group, 3)  # tf.split(3, group, input)
                kernel_groups = tf.split(kernel, group, 3)  # tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
                conv = tf.concat(output_groups, 3)  # tf.concat(3, output_groups)
            return tf.reshape(tf.nn.bias_add(conv, biases), [-1] + conv.get_shape().as_list()[1:])

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

        if arch_type == 'vgg16':
            if os.path.exists(trained_weights_path_end_end):
                # print("Loading the learned weights...")
                net_data = np.load(open(trained_weights_path_end_end, "rb"), encoding="latin1").item()
                self.conv1W = tf.Variable(net_data["conv1"][0], trainable=train_type)
                self.conv1b = tf.Variable(net_data["conv1"][1], trainable=train_type)

                self.conv2W = tf.Variable(net_data["conv2"][0], trainable=train_type)
                self.conv2b = tf.Variable(net_data["conv2"][1], trainable=train_type)

                self.conv3W = tf.Variable(net_data["conv3"][0], trainable=train_type)
                self.conv3b = tf.Variable(net_data["conv3"][1], trainable=train_type)

                self.conv4W = tf.Variable(net_data["conv4"][0], trainable=train_type)
                self.conv4b = tf.Variable(net_data["conv4"][1], trainable=train_type)

                self.conv5W = tf.Variable(net_data["conv5"][0], trainable=train_type)
                self.conv5b = tf.Variable(net_data["conv5"][1], trainable=train_type)

                self.conv6W = tf.Variable(net_data["conv6"][0], trainable=train_type)
                self.conv6b = tf.Variable(net_data["conv6"][1], trainable=train_type)

                self.conv7W = tf.Variable(net_data["conv7"][0], trainable=train_type)
                self.conv7b = tf.Variable(net_data["conv7"][1], trainable=train_type)

                self.conv8W = tf.Variable(net_data["conv8"][0], trainable=train_type)
                self.conv8b = tf.Variable(net_data["conv8"][1], trainable=train_type)

                self.conv9W = tf.Variable(net_data["conv9"][0], trainable=train_type)
                self.conv9b = tf.Variable(net_data["conv9"][1], trainable=train_type)

                self.conv10W = tf.Variable(net_data["conv10"][0], trainable=train_type)
                self.conv10b = tf.Variable(net_data["conv10"][1], trainable=train_type)

                self.fc6W = tf.Variable(net_data["fc6"][0], trainable=train_type)
                self.fc6b = tf.Variable(net_data["fc6"][1], trainable=train_type)

                # self.fc7W = tf.Variable(net_data["fc7"][0])
                # self.fc7b = tf.Variable(net_data["fc7"][1])
                #
                # self.fc8W = tf.Variable(net_data["fc8"][0])
                # self.fc8b = tf.Variable(net_data["fc8"][1])

                self.fc7W = tf.Variable(tf.truncated_normal(shape=(4096, 4096), stddev=0.05))
                self.fc7b = tf.Variable(tf.truncated_normal(shape=[4096], stddev=0.05))

                self.fc8W = tf.Variable(tf.truncated_normal(shape=(4096, num_actions), stddev=0.05))
                self.fc8b = tf.Variable(tf.constant(0.005, shape=[num_actions]))

            elif weights_path == 'Random':
                # print('Loading random weights')
                self.conv1W = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 64)), trainable=False)
                self.conv1b = tf.Variable(tf.truncated_normal(shape=[64]), trainable=False)

                self.conv2W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64)), trainable=False)
                self.conv2b = tf.Variable(tf.truncated_normal(shape=[64]), trainable=False)

                self.conv3W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128)), trainable=False)
                self.conv3b = tf.Variable(tf.truncated_normal(shape=[128]), trainable=False)

                self.conv4W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128)), trainable=False)
                self.conv4b = tf.Variable(tf.truncated_normal(shape=[128]), trainable=False)

                self.conv5W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 256)), trainable=False)
                self.conv5b = tf.Variable(tf.truncated_normal(shape=[256]), trainable=False)

                self.conv6W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256)), trainable=False)
                self.conv6b = tf.Variable(tf.truncated_normal(shape=[256]), trainable=False)

                self.conv7W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256)), trainable=False)
                self.conv7b = tf.Variable(tf.truncated_normal(shape=[256]), trainable=False)

                self.conv8W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256)), trainable=False)
                self.conv8b = tf.Variable(tf.truncated_normal(shape=[256]), trainable=False)

                self.conv9W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256)), trainable=False)
                self.conv9b = tf.Variable(tf.truncated_normal(shape=[256]), trainable=False)

                self.conv10W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256)), trainable=False)
                self.conv10b = tf.Variable(tf.truncated_normal(shape=[256]), trainable=False)


                self.fc6W = tf.Variable(tf.truncated_normal(shape=(9216, 4096), stddev=0.05))
                self.fc6b = tf.Variable(tf.truncated_normal(shape=[4096], stddev=0.05))

                self.fc7W = tf.Variable(tf.truncated_normal(shape=(4096, 2048), stddev=0.05))
                self.fc7b = tf.Variable(tf.truncated_normal(shape=[2048], stddev=0.05))

                self.fc8W = tf.Variable(tf.truncated_normal(shape=(2048, 2048), stddev=0.05))
                self.fc8b = tf.Variable(tf.truncated_normal(shape=[2048], stddev=0.005))

                self.fc9W = tf.Variable(tf.truncated_normal(shape=(2048, 1024), stddev=0.05))
                self.fc9b = tf.Variable(tf.truncated_normal(shape=[1024], stddev=0.05))

                self.fc10W = tf.Variable(tf.truncated_normal(shape=(1024, num_actions), stddev=0.05))
                self.fc10b = tf.Variable(tf.truncated_normal(shape=[num_actions], stddev=0.005))

            #Yet to edit
            elif weights_path == 'Imagenet':
                # Imagenet can either be alexnet or date network
                # Alexnet 2 - 128 instead of 256
                # print('Loading Imagenet weights')
                # net_data = np.load(open("models/imagenet_weights.npy", "rb"), encoding="latin1").item()

                net_data = np.load(open('models/vgg16_weights.npz', "rb"), encoding="latin1")
                c = net_data['conv1_1_W.npy']
                self.conv1W = tf.Variable(net_data["conv1_1_W.npy"], trainable=train_conv_fc6)
                self.conv1b = tf.Variable(net_data["conv1_1_b.npy"], trainable=train_conv_fc6)

                self.conv2W = tf.Variable(net_data["conv1_2_W.npy"], trainable=train_conv_fc6)
                self.conv2b = tf.Variable(net_data["conv1_2_b.npy"], trainable=train_conv_fc6)

                self.conv3W = tf.Variable(net_data["conv2_1_W.npy"], trainable=train_conv_fc6)
                self.conv3b = tf.Variable(net_data["conv2_1_b.npy"], trainable=train_conv_fc6)

                self.conv4W = tf.Variable(net_data["conv2_2_W.npy"], trainable=train_conv_fc6)
                self.conv4b = tf.Variable(net_data["conv2_2_b.npy"], trainable=train_conv_fc6)

                self.conv5W = tf.Variable(net_data["conv3_1_W.npy"], trainable=train_conv_fc6)
                self.conv5b = tf.Variable(net_data["conv3_1_b.npy"], trainable=train_conv_fc6)

                self.conv6W = tf.Variable(net_data["conv3_2_W.npy"], trainable=train_conv_fc6)
                self.conv6b = tf.Variable(net_data["conv3_2_b.npy"], trainable=train_conv_fc6)

                self.conv7W = tf.Variable(net_data["conv3_3_W.npy"], trainable=train_conv_fc6)
                self.conv7b = tf.Variable(net_data["conv3_3_b.npy"], trainable=train_conv_fc6)

                self.conv8W = tf.Variable(net_data["conv3_2_W.npy"], trainable=train_conv_fc6)
                self.conv8b = tf.Variable(net_data["conv3_2_b.npy"], trainable=train_conv_fc6)

                self.conv9W = tf.Variable(net_data["conv3_2_W.npy"], trainable=train_conv_fc6)
                self.conv9b = tf.Variable(net_data["conv3_2_b.npy"], trainable=train_conv_fc6)

                self.conv10W = tf.Variable(net_data["conv3_3_W.npy"], trainable=train_conv_fc6)
                self.conv10b = tf.Variable(net_data["conv3_3_b.npy"], trainable=train_conv_fc6)

                fc6W = net_data['fc6_W.npy']
                fc6b = net_data['fc6_b.npy']


                self.fc6W = tf.Variable(fc6W[0:9216, 0:4096])
                self.fc6b = tf.Variable(fc6b[0:4096])

                self.fc7W = tf.Variable(tf.truncated_normal(shape=(4096, 2048), stddev=0.05))
                self.fc7b = tf.Variable(tf.truncated_normal(shape=[2048], stddev=0.05))

                self.fc8W = tf.Variable(tf.truncated_normal(shape=(2048, 2048), stddev=0.05))
                self.fc8b = tf.Variable(tf.truncated_normal(shape=[2048], stddev=0.005))

                self.fc9W = tf.Variable(tf.truncated_normal(shape=(2048, 1024), stddev=0.05))
                self.fc9b = tf.Variable(tf.truncated_normal(shape=[1024], stddev=0.05))

                self.fc10W = tf.Variable(tf.truncated_normal(shape=(1024, num_actions), stddev=0.05))
                self.fc10b = tf.Variable(tf.truncated_normal(shape=[num_actions], stddev=0.005))






            train_x = np.zeros((1, 224, 224, 3)).astype(np.float32)
            train_y = np.zeros((1, 1000))
            xdim = train_x.shape[1:]
            ydim = train_y.shape[1]

            self.x = tf.placeholder(tf.float32, (None,) + xdim)

            # ----------------- conv1 -----------------#
            k_h = 3
            k_w = 3
            c_o = 64
            s_h = 1
            s_w = 1
            conv1_in = conv(self.x, self.conv1W, self.conv1b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv1 = tf.nn.relu(conv1_in)



            # ----------------- conv2 -----------------#
            k_h = 3
            k_w = 3
            c_o = 64
            s_h = 1
            s_w = 1
            conv2_in = conv(conv1, self.conv2W, self.conv2b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv2 = tf.nn.relu(conv2_in)

            # radius = 2
            # alpha = 2e-05
            # beta = 0.75
            # bias = 1.0
            # lrn1 = tf.nn.local_response_normalization(conv2,
            #                                           depth_radius=radius,
            #                                           alpha=alpha,
            #                                           beta=beta,
            #                                           bias=bias)
            # ----------------- pool1 -----------------#
            k_h = 2
            k_w = 2
            s_h = 2
            s_w = 2
            padding = 'VALID'
            pool1 = tf.nn.max_pool(conv2, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # ----------------- conv3 -----------------#
            k_h = 3
            k_w = 3
            c_o = 128
            s_h = 1
            s_w = 1
            conv3_in = conv(pool1, self.conv3W, self.conv3b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv3 = tf.nn.relu(conv3_in)

            # ----------------- conv4 -----------------#
            k_h = 3
            k_w = 3
            c_o = 128
            s_h = 1
            s_w = 1
            conv4_in = conv(conv3, self.conv4W, self.conv4b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv4 = tf.nn.relu(conv4_in)

            # radius = 2
            # alpha = 2e-05
            # beta = 0.75
            # bias = 1.0
            # lrn2 = tf.nn.local_response_normalization(conv4,
            #                                           depth_radius=radius,
            #                                           alpha=alpha,
            #                                           beta=beta,
            #                                           bias=bias)

            # ----------------- pool2 -----------------#
            k_h = 2
            k_w = 2
            s_h = 2
            s_w = 2
            padding = 'VALID'
            pool2 = tf.nn.max_pool(conv4, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # ----------------- conv5 -----------------#
            k_h = 3
            k_w = 3
            c_o = 256
            s_h = 1
            s_w = 1
            conv5_in = conv(pool2, self.conv5W, self.conv5b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv5 = tf.nn.relu(conv5_in)

            # ----------------- conv6 -----------------#
            k_h = 3
            k_w = 3
            c_o = 256
            s_h = 1
            s_w = 1
            conv6_in = conv(conv5, self.conv6W, self.conv6b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv6 = tf.nn.relu(conv6_in)

            # ----------------- conv7 -----------------#
            k_h = 3
            k_w = 3
            c_o = 256
            s_h = 1
            s_w = 1
            conv7_in = conv(conv6, self.conv7W, self.conv7b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv7 = tf.nn.relu(conv7_in)

            # radius = 2
            # alpha = 2e-05
            # beta = 0.75
            # bias = 1.0
            # lrn3 = tf.nn.local_response_normalization(conv7,
            #                                           depth_radius=radius,
            #                                           alpha=alpha,
            #                                           beta=beta,
            #                                           bias=bias)


            # ----------------- pool3 -----------------#
            k_h = 3
            k_w = 3
            s_h = 3
            s_w = 3
            padding = 'VALID'
            pool3 = tf.nn.max_pool(conv7, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            # ----------------- conv8 -----------------#
            k_h = 3
            k_w = 3
            c_o = 256
            s_h = 1
            s_w = 1
            conv8_in = conv(pool3, self.conv8W, self.conv8b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv8 = tf.nn.relu(conv8_in)

            # ----------------- conv9 -----------------#
            k_h = 3
            k_w = 3
            c_o = 256
            s_h = 1
            s_w = 1
            conv9_in = conv(conv8, self.conv9W, self.conv9b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv9 = tf.nn.relu(conv9_in)

            # ----------------- conv10 -----------------#
            k_h = 3
            k_w = 3
            c_o = 256
            s_h = 1
            s_w = 1
            conv10_in = conv(conv9, self.conv10W, self.conv10b, k_h, k_w, c_o, s_h, s_w, padding="SAME")
            conv10 = tf.nn.relu(conv10_in)
            # ----------------- pool4 -----------------#
            k_h = 3
            k_w = 3
            s_h = 3
            s_w = 3
            padding = 'VALID'
            pool4 = tf.nn.max_pool(conv10, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding)

            self.fc6 = tf.nn.relu_layer(tf.reshape(pool4, [-1, int(np.prod(pool4.get_shape()[1:]))]), self.fc6W,self.fc6b)
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
