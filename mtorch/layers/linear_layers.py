import numpy as np
from exceptions.exceptions import ShapeNotMatchException
from mtorch.layers.layer import Layer
from mtorch.layers.non_linear_activations import Sigmoid, Relu, Tanh


class Linear(Layer):
    """
    input  : (in_dim, batch_size)
    output : (out_dim, batch_size)
    weights: (out_dim, in_dim)
    bias   : (out_dim, 1)
    """

    def __init__(self, in_dim, out_dim, coe=1):
        super(Linear, self).__init__(in_dim, out_dim)
        self.weights, self.bias = None, None
        self.gradients = {"w": None, "b": None}
        self.VARIANCE_COEFFICIENT = coe
        self.init_weights()  # 初始化权重矩阵

    def forward(self, x):
        if not self.match_the_shape(x):
            raise ShapeNotMatchException(self, "__call__: required input dimension:"
                                         + str(self.in_dim) + ", and we got " + str(x.shape))

        self.outputs = np.matmul(x, self.weights) + self.bias
        self.inputs = x
        return self.outputs

    def backward(self, output_gradiant):
        pass

    def weights_shape(self):
        return self.weights.shape

    def bias_shape(self):
        return self.bias.shape

    @staticmethod
    def transpose(matrix):
        """
        将最后两维转置
        :param matrix:
        :return:
        """
        # %%%%%%%%%%%%%丑死了%%%%%%%%%%%%%%%
        if len(matrix.shape) == 2:
            return matrix.T
        elif len(matrix.shape) == 3:
            return np.transpose(matrix, (0, 2, 1))
        else:
            raise ShapeNotMatchException("transpose: shape {} is not supported yet".format(matrix.shape))

    def step(self, lr):
        self.weights -= lr * self.gradients["w"]
        self.bias -= lr * self.gradients["b"]

    def init_weights(self):
        """
        初始化参数，采用标准正态分布
        :return: None
        """
        self.weights = np.random.randn(self.in_dim, self.out_dim) * np.sqrt(self.VARIANCE_COEFFICIENT / self.in_dim)
        self.bias = np.random.randn(1, self.out_dim)
        # 均匀分布 / uniform distribution
        # self.weights = (np.random.rand(self.in_dim, self.out_dim) - 0.5) * 2 * np.sqrt(self.VARIANCE_COEFFICIENT / self.in_dim)
        # self.bias = (np.random.rand(1, self.out_dim) - 0.5) * 2 * np.sqrt(self.VARIANCE_COEFFICIENT / self.in_dim)

        # self.bias = np.zeros((1, self.out_dim))

    def set_gradients(self, w=None, b=None):
        """
        设置梯度
        :param w:
        :param b:
        :return:
        """
        if w is not None:
            self.gradients["w"] = w
        if b is not None:
            self.gradients["b"] = b


class Linear2(Linear):
    """
    输入是二维的Linear
    """

    def __init__(self, in_dim, out_dim, coe=1):
        super(Linear2, self).__init__(in_dim, out_dim, coe)

    def backward(self, output_gradiant):
        batch_size = output_gradiant.shape[0]
        weights_gradient = np.dot(self.inputs.T, output_gradiant) / batch_size
        bias_gradient = np.sum(output_gradiant, 0) / batch_size
        self.gradients["w"] = weights_gradient
        self.gradients["b"] = bias_gradient
        input_gradient = np.matmul(output_gradiant, self.weights.T) / self.out_dim
        return input_gradient


class Linear3(Linear):
    """
    输入是三维的Linear
    """

    def __init__(self, in_dim, out_dim, coe=1):
        super(Linear3, self).__init__(in_dim, out_dim, coe)

    def backward(self, output_gradiant):
        batch_size = output_gradiant.shape[1]
        seq_len = output_gradiant.shape[0]
        weights_gradient = np.matmul(self.transpose(self.inputs), output_gradiant) / batch_size
        weights_gradient = np.sum(weights_gradient, axis=0) / seq_len
        bias_gradient = np.sum(output_gradiant, axis=(0, 1)) / batch_size / seq_len

        self.gradients["w"] = weights_gradient
        self.gradients["b"] = bias_gradient
        # print("sum(Linear3.w_gradient) = {}".format(np.sum(np.abs(self.gradients["w"]))))
        input_gradient = np.matmul(output_gradiant, self.weights.T) / self.out_dim
        return input_gradient


class Perceptron(Layer):
    activation_dict = {"sigmoid": Sigmoid, "relu": Relu, "tanh": Tanh}

    def __init__(self, in_dim, out_dim, matrix_len=2, activation="sigmoid"):
        """
        :param in_dim: input dimension
        :param out_dim: output dimension
        :param matrix_len: the length of the input and output, like 2 for (batch, dim), 3 for (seq, batch, dim)
        :param activation: among "sigmoid", "relu", "tanh"
        """
        super().__init__(in_dim, out_dim)
        coe = 2 if activation == "relu" else 1
        if matrix_len == 2:
            self.linear = Linear2(in_dim, out_dim, coe)
        elif matrix_len == 3:
            self.linear = Linear3(in_dim, out_dim, coe)
        else:
            raise ShapeNotMatchException(self, "unexpected matrix_len parameter: " + str(matrix_len))

        if activation not in activation:
            raise Exception("wong activation name: " + activation)
        self.activation = Perceptron.activation_dict[activation](out_dim)

    def forward(self, inputs):
        return self.activation(self.linear(inputs))

    def backward(self, output_gradiant):
        return self.linear(self.activation(output_gradiant))
