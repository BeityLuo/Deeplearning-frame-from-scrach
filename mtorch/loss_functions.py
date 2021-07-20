import numpy as np

from exceptions.exceptions import ShapeNotMatchException


class SquareLoss:
    def __init__(self, backward_func):
        self.backward_func = backward_func
        self.prediction_gradient = None

    def __call__(self, predictions, targets):
        """
        获取两个矩阵的loss值，公式为0.5 * (prediction - target) ^ 2
        :param predictions:
        :param targets:
        :return:
        """
        """
        predictions.shape = (10, batch_size)
        targets.shape = (1, batch_size)
        """
        targets = self.label_to_matrix(targets, predictions.shape[1])  # 保证targets与predictions的格式相同
        if predictions.shape != targets.shape:
            raise ShapeNotMatchException(self, "__call__: predictions.shape  = " +
                                         str(predictions.shape) + ", targets.shape = " + str(targets.shape))
        temp = predictions - targets
        temp = temp * temp / 2

        self.prediction_gradient = predictions - targets  # 保存预测值的梯度，以后用来反向传播
        return np.sum(temp)

    def backward(self):
        self.backward_func(self.prediction_gradient)

    def label_to_matrix(self, targets, dim):
        batch_size = targets.shape[0]
        targets = targets.astype(np.int)
        targets = targets.reshape(-1).tolist()
        targets = np.eye(dim)[targets]
        targets = np.reshape(targets, (batch_size, dim))
        return targets
