from exceptions.exceptions import ShapeNotMatchException
import numpy as np

EPSILON = 1e-5


class Layer:
    def __init__(self, in_dim: "int", out_dim: "int"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.outputs = None
        self.inputs = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, *tuples, **dicts):
        pass

    def backward(self, output_gradiant):
        """反向传播，记录自身的权重的改变，输出对输入的期望变化值(梯度)
        :param output_gradiant: 输出的期望变化值(梯度)
        :return: 输入的期望变化值
        """
        pass

    def match_the_shape(self, x):
        """shape的最后一维维标志着输入的维数
        :param x: 要检查的矩阵
        :return:
        """
        return x.shape[len(x.shape) - 1] == self.in_dim

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
        if output_gradiant.shape != self.outputs.shape:
            # 期望得到(batch_size, out_dim)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.outputs.shape, output_gradiant.shape))
        input_gradient = np.matmul(output_gradiant, self.weights.T) / self.out_dim
        return input_gradient

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
        self.bias = np.random.randn(1, self.out_dim) * np.sqrt(self.VARIANCE_COEFFICIENT / self.in_dim)

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
        return super(Linear2, self).backward(output_gradiant)


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
        return super(Linear3, self).backward(output_gradiant)



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
        return input_gradient


class RNN(Layer):
    def __init__(self, in_dim, out_dim):
        super(RNN, self).__init__(in_dim, out_dim)
        self.linear = Linear2(in_dim + out_dim, out_dim, 1)
        self.activation = Tanh(out_dim)
        self.outputs = None
        self.seq_len = None

    def forward(self, inputs):
        """
        :param inputs: input = (h0, x) h0.shape == (batch, out_dim) x.shape == (seq, batch, in_dim)
        :return: outputs: outputs.shape == (seq, batch, out_dim)
        """
        h = inputs[0]  # 输入的inputs由两部分组成
        X = inputs[1]
        if X.shape[2] != self.in_dim or h.shape[1] != self.out_dim:
            # 检查输入的形状是否有问题
            raise ShapeNotMatchException(self, "forward: wrong shape: h0 = {}, X = {}".format(h.shape, X.shape))

        self.seq_len = X.shape[0]  # 时间序列的长度
        self.inputs = X  # 保存输入，之后的反向传播还要用
        output_list = []  # 保存每个时间点的输出
        for x in X:
            # 按时间序列遍历input
            # x.shape == (batch, in_dim), h.shape == (batch, out_dim)
            h = self.activation(self.linear(np.c_[h, x]))
            output_list.append(h)
        self.outputs = np.stack(output_list, axis=0)  # 将列表转换成一个矩阵保存起来
        return self.outputs

    def backward(self, output_gradiant):
        """
        :param output_gradiant: shape == (seq, batch, out_dim)
        :return: input_gradiant
        """
        if output_gradiant.shape != self.outputs.shape:
            # 期望得到(seq, batch, out_dim)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.outputs.shape, output_gradiant.shape))

        input_gradients = []
        # 每个time_step上的虚拟weight_gradient, 最后求平均值就是总的weight_gradient
        weight_gradients = np.zeros(self.linear.weights_shape())
        bias_gradients = np.zeros(self.linear.bias_shape())
        batch_size = output_gradiant.shape[1]

        # total_gradient: 前向传播的时候是将x, h合成为一个矩阵，所以反向传播也先计算这个大矩阵的梯度再拆分为x_grad, h_grad
        total_gradient = np.zeros((batch_size, self.out_dim + self.in_dim))
        h_gradient = None

        # 反向遍历各个时间层，计算该层的梯度值
        for i in range(self.seq_len - 1, -1, -1):
            # 前向传播顺序: x, h -> z -> h
            # 所以反向传播计算顺序：h_grad -> z_grad -> x_grad, h_grad, w_grad, b_grad

            # %%%%%%%%%%%%%%计算平均值的版本%%%%%%%%%%%%%%%%%%%%%%%
            # h_gradient = (output_gradiant[i] + total_gradient[:, 0:self.out_dim]) / 2
            # %%%%%%%%%%%%%%不计算平均值的版本%%%%%%%%%%%%%%%%%%%%%%%
            #  计算h_grad: 这一时间点的h_grad包括输出的grad和之前的时间点计算所得grad两部分
            h_gradient = output_gradiant[i] + total_gradient[:, 0:self.out_dim]

            # w_grad和b_grad是在linear.backward()内计算的，不用手动再计算了
            z_gradient = self.activation.backward(h_gradient)  # 计算z_grad
            total_gradient = self.linear.backward(z_gradient)  # 计算x_grad和h_grad合成的大矩阵的梯度

            # total_gradient 同时包含了h和x的gradient, shape == (batch, out_dim + in_dim)
            x_gradient = total_gradient[:, self.out_dim:]

            input_gradients.append(x_gradient)
            weight_gradients += self.linear.gradients["w"]
            bias_gradients += self.linear.gradients["b"]

        # %%%%%%%%%%%%%%%%%%计算平均值的版本%%%%%%%%%%%%%%%%%%%%%%%
        # self.linear.set_gradients(w=weight_gradients / self.seq_len, b=bias_gradients / self.seq_len)
        # %%%%%%%%%%%%%%%%%%不计算平均值的版本%%%%%%%%%%%%%%%%%%%%%%%
        self.linear.set_gradients(w=weight_gradients, b=bias_gradients)  # 设置梯度值

        list.reverse(input_gradients)  # input_gradients是逆序的，最后输出时需要reverse一下
        print("sum(weight_gradients) = {}".format(np.sum(np.abs(weight_gradients))))
        print("sum(h0_gradients) = {}".format(np.sum(np.abs(total_gradient[:, 0:self.out_dim]))))

        # np.stack的作用是将列表转变成一个矩阵
        return np.stack(input_gradients), h_gradient

    def step(self, lr):
        self.linear.step(lr)
        self.activation.step(lr)


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
        # tanh'(x) = (sech(x))**2 = 4 / (e**2x + 2 + e**-2x)
        return 4 / (np.exp(2 * x) + 2 + np.exp(-2 * x))

    def backward(self, output_gradiant):
        if output_gradiant.shape != self.outputs.shape:
            # 期望得到(out_dim, batch_size)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.outputs.shape, output_gradiant.shape))

        input_gradiant = output_gradiant * self.__tanh_derivative(self.inputs)
        return input_gradiant


class Sequential(Layer):
    def __init__(self, layer_list: "list"):
        self.layers = layer_list
        in_dim = layer_list[0].in_dim
        out_dim = layer_list[len(layer_list) - 1].out_dim
        super(Sequential, self).__init__(in_dim, out_dim)

        self.check_dim_sequence()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

    def backward(self, output_gradiant):
        layer_num = len(self.layers)
        delta = output_gradiant
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
                                                   "not match, which is (batch, {}) to (batch, {})".
                                             format(i - 1, i, last_out_dim, self.layers[i].in_dim))
            last_out_dim = self.layers[i].out_dim
