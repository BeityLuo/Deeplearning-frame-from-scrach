import numpy as np

from exceptions.exceptions import ShapeNotMatchException


class SquareLoss:
    def __init__(self, backward_func):
        self.back_func = backward_func
        self.loss = None

    def __call__(self, predictions, targets, transform=False):
        """
        获取两个矩阵的loss值，公式为0.5 * (prediction - target) ^ 2
        :param predictions:
        :param targets:
        :param transform: if transform == True, 那么就认为targets是以label的形式存放
            （比如分类问题用[2]表示第2类，而不是用[0, 0, 1, 0, 0, ...]表示第0类）
        :return:
        """
        """
        predictions.shape = (10, batch_size)
        targets.shape = (1, batch_size)
        """
        if transform:
            targets = self.label_to_matrix(targets, predictions.shape[1])  # 保证targets与predictions的格式相同
        if predictions.shape != targets.shape:
            raise ShapeNotMatchException(self, "__call__: predictions.shape  = " +
                                         str(predictions.shape) + ", targets.shape = " + str(targets.shape))
        temp = predictions - targets
        temp = temp * temp / 2
        loss_value = np.sum(temp)

        prediction_gradient = predictions - targets  # 保存预测值的梯度，以后用来反向传播
        if self.loss is None:
            self.loss = Loss(loss_value, prediction_gradient, self.back_func)
        else:
            self.loss.value = loss_value
            self.loss.prediction_gradient = prediction_gradient
        return self.loss

    @staticmethod
    def label_to_matrix(targets, dim):
        batch_size = targets.shape[0]
        targets = targets.astype(np.int32)
        targets = targets.reshape(-1).tolist()
        targets = np.eye(dim)[targets]
        targets = np.reshape(targets, (batch_size, dim))
        return targets


class Loss:
    def __init__(self, value, prediction_gradient, backward):
        self.back_func = backward
        self.value = value
        self.prediction_gradient = prediction_gradient

    def backward(self):
        self.back_func(self.prediction_gradient)
