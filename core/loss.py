import tensorflow as tf


class JointsMSELoss(tf.losses.Loss):
    def __init__(self):
        super(JointsMSELoss, self).__init__()
        pass

    def call(self, y_true, y_pred):
        pass