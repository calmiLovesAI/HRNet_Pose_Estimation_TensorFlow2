import tensorflow as tf


class BasicBlock(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
        if stride != 1:
            self.downsample = tf.keras.Sequential()
            self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=stride, padding="same"))
            self.downsample.add(tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5))
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu = tf.nn.relu(bn1)
        conv2 = self.conv2(relu)
        bn2 = self.bn2(conv2, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, bn2]))

        return output


class BottleNeck(tf.keras.layers.Layer):
    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(1, 1), strides=1, padding="same", use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv2 = tf.keras.layers.Conv2D(filters=filter_num, kernel_size=(3, 3), strides=stride, padding="same", use_bias=False)
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)
        self.conv3 = tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5)

        self.downsample = tf.keras.Sequential()
        self.downsample.add(tf.keras.layers.Conv2D(filters=filter_num * 4, kernel_size=(1, 1), strides=stride, padding="same", use_bias=False))
        self.downsample.add(tf.keras.layers.BatchNormalization(momentum=0.1, epsilon=1e-5))

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        conv1 = self.conv1(inputs)
        bn1 = self.bn1(conv1, training=training)
        relu1 = tf.nn.relu(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2, training=training)
        relu2 = tf.nn.relu(bn2)
        conv3 = self.conv3(relu2)
        bn3 = self.bn3(conv3, training=training)

        output = tf.nn.relu(tf.keras.layers.add([residual, bn3]))

        return output


def make_basic_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


def make_bottleneck_layer(filter_num, blocks, stride=1):
    res_block = tf.keras.Sequential()
    res_block.add(BottleNeck(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BottleNeck(filter_num, stride=1))

    return res_block