import tensorflow as tf


class JointsMSELoss(object):
    def __init__(self):
        self.mse = tf.losses.MeanSquaredError()

    def __call__(self, y_pred, target, target_weight):
        batch_size = y_pred.shape[0]
        num_of_joints = y_pred.shape[-1]
        pred = tf.reshape(tensor=y_pred, shape=(batch_size, -1, num_of_joints))
        heatmap_pred_list = tf.split(value=pred, num_or_size_splits=num_of_joints, axis=-1)
        gt = tf.reshape(tensor=target, shape=(batch_size, -1, num_of_joints))
        heatmap_gt_list = tf.split(value=gt, num_or_size_splits=num_of_joints, axis=-1)
        loss = 0.0
        for i in range(num_of_joints):
            heatmap_pred = tf.squeeze(heatmap_pred_list[i])
            heatmap_gt = tf.squeeze(heatmap_gt_list[i])
            loss += 0.5 * self.mse(y_true=heatmap_pred * target_weight[:, i],
                                   y_pred=heatmap_gt * target_weight[:, i])
        return loss / num_of_joints
