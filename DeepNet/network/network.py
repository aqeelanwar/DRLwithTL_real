import tensorflow as tf
import numpy as np
from DeepNet.network.loss_functions import huber_loss


class AlexNetDuel(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        weights_path = 'DeepNet/models/imagenet.npy'
        weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        print('Loading imagenet weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1, self.conv1W, self.conv1b = self.conv(self.x, weights["conv1"][0], weights["conv1"][1], k=11, out=96, s=4, p="VALID",trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2, self.conv2W, self.conv2b = self.conv(self.maxpool1, weights["conv2"][0], weights["conv2"][1], k=5, out=256, s=1, p="SAME",trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3, self.conv3W, self.conv3b = self.conv(self.maxpool2, weights["conv3"][0], weights["conv3"][1], k=3, out=384, s=1, p="SAME",trainable=train_conv)
        self.conv4, self.conv4W, self.conv4b = self.conv(self.conv3, weights["conv4"][0], weights["conv4"][1], k=3, out=384, s=1, p="SAME",trainable=train_conv)
        self.conv5, self.conv5W, self.conv5b = self.conv(self.conv4, weights["conv5"][0], weights["conv5"][1], k=3, out=256, s=1, p="SAME",trainable=train_conv)
        self.maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat = tf.contrib.layers.flatten(self.maxpool5)

        # Advantage Network
        self.fc6_a, self.fc6_aW, self.fc6_ab = self.FullyConnected(self.flat,     units_in=9216, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_a, self.fc7_aW, self.fc7_ab = self.FullyConnected(self.fc6_a,    units_in=2048, units_out=1024, act='relu', trainable=train_fc7)
        self.fc8_a, self.fc8_aW, self.fc8_ab = self.FullyConnected(self.fc7_a,    units_in=1024, units_out=1024, act='relu', trainable=train_fc8)
        self.fc9_a, self.fc9_aW, self.fc9_ab = self.FullyConnected(self.fc8_a,    units_in=1024, units_out=512, act='relu', trainable=train_fc9)
        self.fc10_a, self.fc10_aW, self.fc10_ab = self.FullyConnected(self.fc9_a,   units_in=512,  units_out=num_actions, act='linear', trainable=True)

        # Value Network
        self.fc6_v, self.fc6_vW, self.fc6_vb = self.FullyConnected(self.flat,     units_in=9216, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_v, self.fc7_vW, self.fc7_vb = self.FullyConnected(self.fc6_v,    units_in=2048, units_out=1024, act='relu', trainable=train_fc7)
        self.fc8_v, self.fc8_vW, self.fc8_vb = self.FullyConnected(self.fc7_v,    units_in=1024, units_out=1024, act='relu', trainable=train_fc8)
        self.fc9_v, self.fc9_vW, self.fc9_vb = self.FullyConnected(self.fc8_v,    units_in=1024, units_out=512, act='relu', trainable=train_fc9)
        self.fc10_v, self.fc10_vW, self.fc10_vb = self.FullyConnected(self.fc9_v,   units_in=512,  units_out=1, act='linear', trainable=True)

        self.output = self.fc10_v + tf.subtract(self.fc10_a, tf.reduce_mean(self.fc10_a, axis=1, keep_dims=True))

        regularizers = tf.nn.l2_loss(self.conv1W) + tf.nn.l2_loss(self.conv1b)
        regularizers += tf.nn.l2_loss(self.conv2W) + tf.nn.l2_loss(self.conv2b)
        regularizers += tf.nn.l2_loss(self.conv3W) + tf.nn.l2_loss(self.conv3b)
        regularizers += tf.nn.l2_loss(self.conv4W) + tf.nn.l2_loss(self.conv4b)
        regularizers += tf.nn.l2_loss(self.conv5W) + tf.nn.l2_loss(self.conv5b)

        regularizers += tf.nn.l2_loss(self.fc6_aW) + tf.nn.l2_loss(self.fc6_ab) + tf.nn.l2_loss(
            self.fc6_vW) + tf.nn.l2_loss(self.fc6_vb)
        regularizers += tf.nn.l2_loss(self.fc7_aW) + tf.nn.l2_loss(self.fc7_ab) + tf.nn.l2_loss(
            self.fc7_vW) + tf.nn.l2_loss(self.fc7_vb)
        regularizers += tf.nn.l2_loss(self.fc8_aW) + tf.nn.l2_loss(self.fc8_ab) + tf.nn.l2_loss(
            self.fc8_vW) + tf.nn.l2_loss(self.fc8_vb)
        regularizers += tf.nn.l2_loss(self.fc9_aW) + tf.nn.l2_loss(self.fc9_ab) + tf.nn.l2_loss(
            self.fc9_vW) + tf.nn.l2_loss(self.fc9_vb)
        regularizers += tf.nn.l2_loss(self.fc10_aW) + tf.nn.l2_loss(self.fc10_ab) + tf.nn.l2_loss(
            self.fc10_vW) + tf.nn.l2_loss(self.fc10_vb)

        self.regularizer = regularizers
    def conv(self, input, W, b, k, out, s, p, trainable=True):
        assert (W.shape[0] == k)
        assert (W.shape[1] == k)
        assert (W.shape[3] == out)
        W = tf.Variable(W, trainable)
        b = tf.Variable(b, trainable)
        conv_kernel_1 = tf.nn.conv2d(input, W, [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, b)

        return tf.nn.relu(bias_layer_1), W, b

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W,b), W, b
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b), W, b
        else:
            assert (1 == 0)



class AlexNetConditional(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        weights_path = 'DeepNet/models/imagenet.npy'
        weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        print('Loading imagenet weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, weights["conv1"][0], weights["conv1"][1], k=11, out=96, s=4, p="VALID",trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, weights["conv2"][0], weights["conv2"][1], k=5, out=256, s=1, p="SAME",trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, weights["conv3"][0], weights["conv3"][1], k=3, out=384, s=1, p="SAME",trainable=train_conv)

        # Divide the network stream from this point onwards

        # One - Main Network
        self.conv4_main = self.conv(self.conv3, weights["conv4"][0], weights["conv4"][1], k=3, out=384, s=1, p="SAME",trainable=train_conv)
        self.conv5_main = self.conv(self.conv4_main, weights["conv5"][0], weights["conv5"][1], k=3, out=256, s=1, p="SAME",trainable=train_conv)
        self.maxpool5_main = tf.nn.max_pool(self.conv5_main, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat_main = tf.contrib.layers.flatten(self.maxpool5_main)

        # Advantage Network
        self.fc6_a_main = self.FullyConnected(self.flat_main,     units_in=9216, units_out=4096, act='relu', trainable=train_fc6)
        self.fc7_a_main = self.FullyConnected(self.fc6_a_main,    units_in=4096, units_out=2048, act='relu', trainable=train_fc7)
        self.fc8_a_main = self.FullyConnected(self.fc7_a_main,    units_in=2048, units_out=num_actions, act='linear', trainable=train_fc8)

        # Value Network
        self.fc6_v_main = self.FullyConnected(self.flat_main,     units_in=9216, units_out=4096, act='relu', trainable=train_fc6)
        self.fc7_v_main = self.FullyConnected(self.fc6_v_main,    units_in=4096, units_out=2048, act='relu', trainable=train_fc7)
        self.fc8_v_main = self.FullyConnected(self.fc7_v_main,   units_in=2048,  units_out=1, act='linear', trainable=True)

        self.output_main = self.fc8_v_main + tf.subtract(self.fc8_a_main, tf.reduce_mean(self.fc8_a_main, axis=1, keep_dims=True))

        # Two - Conditional Network
        conv4_cdl_k = np.random.rand(3, 3, 384, 256).astype(np.float32)
        conv4_cdl_b = np.random.rand(256).astype(np.float32)
        self.conv4_cdl = self.conv(self.conv3, conv4_cdl_k, conv4_cdl_b, k=3, out=256, s=1, p="SAME",trainable=train_conv)
        self.maxpool4_cdl = tf.nn.max_pool(self.conv4_cdl, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat_cdl = tf.contrib.layers.flatten(self.maxpool4_cdl)

        # Advantage Network
        self.fc6_a_cdl = self.FullyConnected(self.flat_cdl, units_in=9216, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_a_cdl = self.FullyConnected(self.fc6_a_cdl, units_in=2048, units_out=num_actions, act='linear',trainable=train_fc7)

        # Value Network
        self.fc6_v_cdl = self.FullyConnected(self.flat_cdl, units_in=9216, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_v_cdl = self.FullyConnected(self.fc6_v_cdl, units_in=2048, units_out=1, act='linear',trainable=train_fc7)

        self.output_cdl = self.fc7_v_cdl + tf.subtract(self.fc7_a_cdl,tf.reduce_mean(self.fc7_a_cdl, axis=1, keep_dims=True))

    def conv(self, input, W, b, k, out, s, p, trainable=True):
        assert (W.shape[0] == k)
        assert (W.shape[1] == k)
        assert (W.shape[3] == out)

        conv_kernel_1 = tf.nn.conv2d(input, tf.Variable(W, trainable), [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W,b)
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b)
        else:
            assert (1 == 0)





class AlexNetDuelPrune(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        # weights_path = 'models/imagenet.npy'
        weights_path = 'models/prune_weights.npy'
        weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        print('Loading pruned weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, weights["conv1W"], weights["conv1b"], k=11, out=64, s=4, p="VALID",trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, weights["conv2W"], weights["conv2b"], k=5, out=192, s=1, p="SAME",trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, weights["conv3W"],weights["conv3b"], k=3, out=288, s=1, p="SAME",trainable=train_conv)
        self.conv4 = self.conv(self.conv3, weights["conv4W"], weights["conv4b"], k=3, out=288, s=1, p="SAME",trainable=train_conv)
        self.conv5 = self.conv(self.conv4, weights["conv5W"], weights["conv5b"], k=3, out=256, s=1, p="SAME",trainable=train_conv)
        self.maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat = tf.contrib.layers.flatten(self.maxpool5)

        # Advantage Network
        self.fc6_a = self.FullyConnected(self.flat,     units_in=9216, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_a = self.FullyConnected(self.fc6_a,    units_in=2048, units_out=1024, act='relu', trainable=train_fc7)
        self.fc8_a = self.FullyConnected(self.fc7_a,    units_in=1024, units_out=1024, act='relu', trainable=train_fc8)
        self.fc9_a = self.FullyConnected(self.fc8_a,    units_in=1024, units_out=512, act='relu', trainable=train_fc9)
        self.fc10_a = self.FullyConnected(self.fc9_a,   units_in=512,  units_out=num_actions, act='linear', trainable=True)

        # Value Network
        self.fc6_v = self.FullyConnected(self.flat,     units_in=9216, units_out=2048, act='relu', trainable=train_fc6)
        self.fc7_v = self.FullyConnected(self.fc6_v,    units_in=2048, units_out=1024, act='relu', trainable=train_fc7)
        self.fc8_v = self.FullyConnected(self.fc7_v,    units_in=1024, units_out=1024, act='relu', trainable=train_fc8)
        self.fc9_v = self.FullyConnected(self.fc8_v,    units_in=1024, units_out=512, act='relu', trainable=train_fc9)
        self.fc10_v = self.FullyConnected(self.fc9_v,   units_in=512,  units_out=1, act='linear', trainable=True)

        self.output = self.fc10_v + tf.subtract(self.fc10_a, tf.reduce_mean(self.fc10_a, axis=1, keep_dims=True))


    def conv(self, input, W, b, k, out, s, p, trainable=True):
        assert (W.shape[0] == k)
        assert (W.shape[1] == k)
        assert (W.shape[3] == out)

        conv_kernel_1 = tf.nn.conv2d(input, tf.Variable(W, trainable), [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.Variable(tf.truncated_normal(shape=(units_in, units_out), stddev=0.05), trainable=trainable)
        b = tf.Variable(tf.truncated_normal(shape=[units_out], stddev=0.05), trainable=trainable)

        if act == 'relu':
            return tf.nn.relu_layer(input, W,b)
        elif act == 'linear':
            return tf.nn.xw_plus_b(input, W, b)
        else:
            assert (1 == 0)





class AlexNet(object):

    def __init__(self, x, num_actions, train_type):
        self.x = x
        weights_path = 'DeepNet/models/imagenet.npy'
        weights = np.load(open(weights_path, "rb"), encoding="latin1").item()
        print('Loading imagenet weights for the conv layers and random for fc layers')
        train_conv = True
        train_fc6 = True
        train_fc7 = True
        train_fc8 = True
        train_fc9 = True

        if train_type == 'last4':
            train_conv = False
            train_fc6 = False
        elif train_type == 'last3':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
        elif train_type == 'last2':
            train_conv = False
            train_fc6 = False
            train_fc7 = False
            train_fc8 = False

        self.conv1 = self.conv(self.x, weights["conv1"][0], weights["conv1"][1], k=11, out=96, s=4, p="VALID",trainable=train_conv)
        self.maxpool1 = tf.nn.max_pool(self.conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv2 = self.conv(self.maxpool1, weights["conv2"][0], weights["conv2"][1], k=5, out=256, s=1, p="SAME",trainable=train_conv)
        self.maxpool2 = tf.nn.max_pool(self.conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.conv3 = self.conv(self.maxpool2, weights["conv3"][0], weights["conv3"][1], k=3, out=384, s=1, p="SAME",trainable=train_conv)
        self.conv4 = self.conv(self.conv3, weights["conv4"][0], weights["conv4"][1], k=3, out=384, s=1, p="SAME",trainable=train_conv)
        self.conv5 = self.conv(self.conv4, weights["conv5"][0], weights["conv5"][1], k=3, out=256, s=1, p="SAME",trainable=train_conv)
        self.maxpool5 = tf.nn.max_pool(self.conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")

        self.flat = tf.contrib.layers.flatten(self.maxpool5)


        self.fc6 = self.FullyConnected(self.flat,   units_in=9216, units_out=4096, act='relu', trainable=train_fc6)
        self.fc7 = self.FullyConnected(self.fc6,    units_in=4096, units_out=2048, act='relu', trainable=train_fc7)
        self.fc8 = self.FullyConnected(self.fc7,    units_in=2048, units_out=2048, act='relu', trainable=train_fc8)
        self.fc9 = self.FullyConnected(self.fc8,    units_in=2048, units_out=1024, act='relu', trainable=train_fc9)
        self.fc10 = self.FullyConnected(self.fc9,   units_in=1024,  units_out=num_actions, act='linear', trainable=True)


        self.output = self.fc10
        print(self.conv1)
        print(self.conv2)
        print(self.conv3)
        print(self.conv4)
        print(self.conv5)
        print(self.fc6)
        print(self.fc7)
        print(self.fc8)
        print(self.fc9)
        print(self.fc10)

    def conv(self, input, W, b, k, out, s, p, trainable=True):
        assert (W.shape[0] == k)
        assert (W.shape[1] == k)
        assert (W.shape[3] == out)

        conv_kernel_1 = tf.nn.conv2d(input, tf.Variable(W, trainable), [1, s, s, 1], padding=p)
        bias_layer_1 = tf.nn.bias_add(conv_kernel_1, tf.Variable(b, trainable))

        return tf.nn.relu(bias_layer_1)

    def FullyConnected(self, input, units_in, units_out, act, trainable=True):
        W = tf.truncated_normal(shape=(units_in, units_out), stddev=0.05)
        b = tf.truncated_normal(shape=[units_out], stddev=0.05)

        if act == 'relu':
            return tf.nn.relu_layer(input, tf.Variable(W, trainable), tf.Variable(b, trainable))
        elif act == 'linear':
            return tf.nn.xw_plus_b(input,  tf.Variable(W, trainable), tf.Variable(b, trainable))
        else:
            assert (1 == 0)