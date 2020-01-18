import numpy as np
from utils.work_flow import get_max_preds


class PCK(object):
    def __init__(self):
        self.threshold = 0.5

    def __call__(self, network_output, target):
        _, h, w, c = network_output.shape
        index = list(range(c))
        pred, _ = get_max_preds(heatmap_tensor=network_output)
        target, _ = get_max_preds(heatmap_tensor=target)
        normalize = np.ones((pred.shape[0], 2)) * np.array([h, w]) / 10
        distance = self.__calculate_distance(pred, target, normalize)

        accuracy = np.zeros((len(index) + 1))
        average_accuracy = 0
        count = 0

        for i in range(c):
            accuracy[i + 1] = self.__distance_accuracy(distance[index[i]])
            if accuracy[i + 1] > 0:
                average_accuracy += accuracy[i + 1]
                count += 1
        average_accuracy = average_accuracy / count if count != 0 else 0
        if count != 0:
            accuracy[0] = average_accuracy
        return accuracy, average_accuracy, count, pred

    @staticmethod
    def __calculate_distance(pred, target, normalize):
        pred = pred.astype(np.float32)
        target = target.astype(np.float32)
        distance = np.zeros((pred.shape[-1], pred.shape[0]))
        for n in range(pred.shape[0]):
            for c in range(pred.shape[-1]):
                if target[n, 0, c] > 1 and target[n, 1, c] > 1:
                    normed_preds = pred[n, :, c] / normalize[n]
                    normed_targets = target[n, :, c] / normalize[n]
                    distance[c, n] = np.linalg.norm(normed_preds - normed_targets)
                else:
                    distance[c, n] = -1
        return distance

    def __distance_accuracy(self, distance):
        distance_calculated = np.not_equal(distance, -1)
        num_dist_cal = distance_calculated.sum()
        if num_dist_cal > 0:
            return np.less(distance[distance_calculated], self.threshold).sum() * 1.0 / num_dist_cal
        else:
            return -1
