from exceptions.exceptions import ShapeNotMatchException
import numpy as np

EPSILON = 1e-5


class Layer:
    def __init__(self, in_dim: "int", out_dim: "int"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.output = None
        self.input = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        pass

    def backward(self, output_delta):
        """反向传播，记录自身的权重的改变，输出对输入的期望变化值(梯度)
        :param output_delta: 输出的期望变化值(梯度)
        :return: 输入的期望变化值
        """
        pass

    def match_the_shape(self, x):
        """shape的第0维标志着输入的维数
        :param x: 要检查的矩阵
        :return:
        """
        return x.shape[0] == self.in_dim

    def step(self, lr):
        """
        根据反向传播得到的值，更新自己的权重
        :param lr: 学习率
        :return:
        """
        pass


class Linear(Layer):
    """
    input  : (in_dim, batch_size)
    output : (out_dim, batch_size)
    weights: (out_dim, in_dim)
    bias   : (out_dim, 1)
    """
    VARIANCE_COEFFICIENT = 1

    def __init__(self, in_dim, out_dim):
        super(Linear, self).__init__(in_dim, out_dim)
        self.weights, self.bias = None, None
        self.init_weights(in_dim, out_dim)  # 初始化权重矩阵
        self.gradients = {"weights": None, "bias": None}

    def forward(self, x):
        if not self.match_the_shape(x):
            raise ShapeNotMatchException(self, "__call__: requied input dimention:"
                                         + str(self.in_dim) + ", and we got " + str(x.shape))
        self.output = np.dot(self.weights, x) + self.bias
        self.input = x
        return self.output

    def backward(self, output_delta):
        if output_delta.shape != self.output.shape:
            # 期望得到(out_dim, batch_size)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.output.shape, output_delta.shape))
        batch_size = output_delta.shape[1]

        input_gradient = np.dot(self.weights.T, output_delta) / self.out_dim
        weights_gradient = np.dot(output_delta, self.input.T) / batch_size
        bias_gradient = np.dot(output_delta, np.ones((batch_size, 1))) / batch_size

        self.gradients["weights"] = weights_gradient
        self.gradients["bias"] = bias_gradient

        return input_gradient

    def step(self, lr):
        self.weights -= lr * self.gradients["weights"]
        self.bias -= lr * self.gradients["bias"]

    def init_weights(self, in_dim, out_dim):
        """
        初始化参数，采用标准正态分布
        :param in_dim:
        :param out_dim:
        :return:
        """
        self.weights = np.random.randn(out_dim, in_dim) * np.sqrt(Linear.VARIANCE_COEFFICIENT / in_dim)
        self.bias = np.random.randn(out_dim, 1) * np.sqrt(Linear.VARIANCE_COEFFICIENT / in_dim)


class Sigmoid(Layer):
    def __init__(self, in_dim):
        super(Sigmoid, self).__init__(in_dim, in_dim)

    def forward(self, x):
        if not self.match_the_shape(x):
            raise ShapeNotMatchException("Sigmoid::__call__: requied input dimention:"
                                         + str(self.in_dim) + ", and we got " + str(x.shape))
        self.input = x
        self.output = self.__sigmoid(x)
        return self.output

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))

    def __sigmoid_derivative(self, x):
        e = np.exp(-1 * x)
        return e / (1 + e) ** 2

    def backward(self, output_delta):
        if output_delta.shape != self.output.shape:
            # 期望得到(out_dim, batch_size)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.output.shape, output_delta.shape))
        input_gradient = output_delta * self.__sigmoid_derivative(self.input)
        return input_gradient


class Relu(Layer):
    def __init__(self, in_dim):
        super(Relu, self).__init__(in_dim, in_dim)

    def forward(self, x):
        if not self.match_the_shape(x):
            raise ShapeNotMatchException("ReLu::__call__: required input dimension:"
                                         + str(self.in_dim) + ", and we got " + str(x.shape))
        self.input = x
        self.output = self.__relu(x)
        return self.output

    def __relu(self, x):
        return np.where(x < 0, 0, x)

    def __relu_derivative(self, x):
        return np.where(x < 0, 0, 1)

    def backward(self, output_delta):
        if output_delta.shape != self.output.shape:
            # 期望得到(out_dim, batch_size)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.output.shape, output_delta.shape))

        input_gradiant = output_delta * self.__relu_derivative(self.input)
        return input_gradiant


class Sequential(Layer):
    def __init__(self, list: "list"):
        self.layers = list
        in_dim = list[0].in_dim
        out_dim = list[len(list) - 1].out_dim
        super(Sequential, self).__init__(in_dim, out_dim)

        self.check_dim_sequence()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, output_delta):
        layer_num = len(self.layers)
        delta = output_delta
        for i in range(layer_num - 1, -1, -1):
            # 反向遍历各个层, 将期望改变量反向传播
            delta = self.layers[i].backward(delta)

    def step(self, lr):
        for layer in self.layers:
            layer.step(lr)

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.layers):
            ans = self.layers[self.idx]
            self.idx += 1
            return ans
        else:
            raise StopIteration

    def check_dim_sequence(self):
        last_out_dim = self.layers[0].out_dim
        for i in range(1, len(self.layers)):
            if self.layers[i].in_dim != last_out_dim:
                raise ShapeNotMatchException(self, "check_dim_sequence: from layer{} to layer{} "
                                                   "not match, which is ({}, batch) to ({}, batch)".
                                             format(i - 1, i, last_out_dim, self.layers[i].in_dim))
            last_out_dim = self.layers[i].out_dim
