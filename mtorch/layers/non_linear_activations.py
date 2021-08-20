import numpy as np

from exceptions.exceptions import ShapeNotMatchException
from mtorch.layers.layer import Layer


class Sigmoid(Layer):
    def __init__(self, in_dim):
        super(Sigmoid, self).__init__(in_dim, in_dim)

    def forward(self, x):
        if not self.match_the_shape(x):
            raise ShapeNotMatchException(self, "Sigmoid::__call__: requied input dimention:"
                                         + str(self.in_dim) + ", and we got " + str(x.shape))
        self.inputs = x
        self.outputs = self.__sigmoid(x)
        return self.outputs

    @staticmethod
    def __sigmoid(x):
        return 1 / (1 + np.exp(-1 * x))

    @staticmethod
    def __sigmoid_derivative(x):
        e = np.exp(-1 * x)
        return e / (1 + e) ** 2

    def backward(self, output_gradiant):
        if output_gradiant.shape != self.outputs.shape:
            # 期望得到(out_dim, batch_size)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.outputs.shape, output_gradiant.shape))
        input_gradient = output_gradiant * self.__sigmoid_derivative(self.inputs)
        pos_cnt = 0
        neg_cnt = 0
        return input_gradient


class Relu(Layer):
    def __init__(self, in_dim):
        super(Relu, self).__init__(in_dim, in_dim)

    def forward(self, x):
        if not self.match_the_shape(x):
            raise ShapeNotMatchException(self, "ReLu::__call__: required input dimension:"
                                         + str(self.in_dim) + ", and we got " + str(x.shape))
        self.inputs = x
        self.outputs = self.__relu(x)
        return self.outputs

    @staticmethod
    def __relu(x):
        return np.where(x < 0, 0, x)

    @staticmethod
    def __relu_derivative(x):
        return np.where(x < 0, 0, 1)

    def backward(self, output_gradiant):
        if output_gradiant.shape != self.outputs.shape:
            # 期望得到(out_dim, batch_size)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.outputs.shape, output_gradiant.shape))

        input_gradiant = output_gradiant * self.__relu_derivative(self.inputs)
        return input_gradiant


class Tanh(Layer):
    def __init__(self, in_dim):
        super(Tanh, self).__init__(in_dim, in_dim)

    def forward(self, x):
        if not self.match_the_shape(x):
            raise ShapeNotMatchException(self, "__call__: required input dimension:"
                                         + str(self.in_dim) + ", and we got " + str(x.shape))
        self.inputs = x
        self.outputs = self.__tanh(x)
        return self.outputs

    @staticmethod
    def __tanh(x):
        return np.tanh(x)

    @staticmethod
    def __tanh_derivative(x):
        # tanh'(x) = sech(x)**2 = 1 - tanh(x)**2 = 4 / (e**(2*x) + e**(-2*x) + 2)
        return 4 / (np.exp(2*x) + np.exp(-2*x) + 2)
        # return 1 - np.tanh(x) ** 2

    def backward(self, output_gradiant):
        if output_gradiant.shape != self.outputs.shape:
            # 期望得到(out_dim, batch_size)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.outputs.shape, output_gradiant.shape))
        # tanh'(x) = sech(x)**2 = 1 - tanh(x)**2
        # input_gradiant = output_gradiant * (1 - self.outputs**2)
        input_gradiant = output_gradiant * (1 - self.outputs ** 2)
        return input_gradiant
