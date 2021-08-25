import numpy as np

from exceptions.exceptions import ShapeNotMatchException
from mtorch.layers.layer import Layer
from mtorch.layers.linear_layers import Linear2, Perceptron
from mtorch.layers.non_linear_activations import Tanh


class RNN(Layer):
    def __init__(self, in_dim, out_dim):
        super(RNN, self).__init__(in_dim, out_dim)
        self.perceptron = Perceptron(in_dim + out_dim, out_dim, activation="tanh")
        self.outputs = None
        self.seq_len = None

    def forward(self, inputs):
        """
        :param inputs: input = X, h0   h0.shape == (batch, out_dim) x.shape == (seq, batch, in_dim)
        :return: outputs: outputs.shape == (seq, batch, out_dim)
        """
        X = inputs[0]
        batch_size = X.shape[1]
        if inputs[1] is None:
            h = np.zeros((batch_size, self.out_dim))
        else:
            h = inputs[1]
        if X.shape[2] != self.in_dim or h.shape[1] != self.out_dim:
            # 检查输入的形状是否有问题
            raise ShapeNotMatchException(self, "forward: wrong shape: h0 = {}, X = {}".format(h.shape, X.shape))

        self.seq_len = X.shape[0]  # 时间序列的长度
        self.inputs = (X, h)  # 保存输入，之后的反向传播还要用
        output_list = []  # 保存每个时间点的输出
        for x in X:
            # 按时间序列遍历input
            # x.shape == (batch, in_dim), h.shape == (batch, out_dim)
            h = self.perceptron(np.c_[h, x])
            output_list.append(h)
        self.outputs = np.stack(output_list, axis=0)  # 将列表转换成一个矩阵保存起来
        return self.outputs

    def backward(self, output_gradient, inputs=None):
        """
        :param inputs: MUST BE NONE!!!!!
        :param output_gradient: shape == (seq, batch, out_dim)
        :return: input_gradiant
        """
        if output_gradient.shape != self.outputs.shape:
            # 期望得到(seq, batch, out_dim)形状
            raise ShapeNotMatchException(self, "__backward: expected {}, but we got "
                                               "{}".format(self.outputs.shape, output_gradient.shape))

        input_gradients = []
        # 每个time_step上的虚拟weight_gradient, 最后求平均值就是总的weight_gradient
        weight_gradients = np.zeros(self.perceptron.weights().shape)
        bias_gradients = np.zeros(self.perceptron.bias().shape)
        batch_size = output_gradient.shape[1]

        # total_gradient: 前向传播的时候是将x, h合成为一个矩阵，所以反向传播也先计算这个大矩阵的梯度再拆分为x_grad, h_grad
        total_gradient = np.zeros((batch_size, self.out_dim + self.in_dim))
        h_gradient = None
        h0 = self.inputs[1]
        X = self.inputs[0]
        h = self.outputs

        # 反向遍历各个时间层，计算该层的梯度值
        for i in range(self.seq_len - 1, -1, -1):
            # 前向传播顺序: x, h -> h, 所以反向传播计算顺序：h_grad -> x_grad, h_grad, w_grad, b_grad
            # 计算h_grad: 这一时间点的h_grad包括输出的grad和之前的时间点计算所得grad两部分
            h_gradient = (output_gradient[i] + total_gradient[:, 0:self.out_dim])
            # w_grad和b_grad是在linear.backward()内计算的，不用手动再计算了
            # 计算当前时间的input if t = 0 : [h0, X[i]] else [h[i - 1], X[i]]
            input = np.c_[h[i - 1] if i != 0 else h0, X[i]]
            # 计算x_grad和h_grad合成的大矩阵的梯度
            total_gradient = self.perceptron.backward(h_gradient, input)
            # total_gradient 同时包含了h和x的gradient, shape == (batch, out_dim + in_dim)
            x_gradient = total_gradient[:, self.out_dim:]

            input_gradients.append(x_gradient)
            weight_gradients += self.perceptron.linear.gradients["w"]
            bias_gradients += self.perceptron.linear.gradients["b"]

        self.perceptron.set_gradients(w=weight_gradients, b=bias_gradients)  # 设置梯度值
        list.reverse(input_gradients)  # input_gradients是逆序的，最后输出时需要reverse一下
        # np.stack的作用是将列表转变成一个矩阵
        return np.stack(input_gradients), h_gradient

    def step(self, lr):
        self.perceptron.step(lr)


class LSTM(Layer):
    def __init__(self, in_dim, out_dim):
        super(LSTM, self).__init__(in_dim, out_dim)
        # u: update, f: forget, o: output
        self.u_gate = Perceptron(out_dim + in_dim, out_dim, activation="sigmoid", matrix_len=2)
        self.f_gate = Perceptron(out_dim + in_dim, out_dim, activation="sigmoid", matrix_len=2)
        self.o_gate = Perceptron(out_dim + in_dim, out_dim, activation="sigmoid", matrix_len=2)
        self.perceptron = Perceptron(out_dim + in_dim, out_dim, activation="tanh", matrix_len=2)
        self.activation = Tanh(out_dim)

        # forward时记录这些值，backward要用
        self.c_list = []
        self.c_hat_list = []
        self.u_list = []
        self.f_list = []
        self.o_list = []

    def forward(self, inputs):
        """
        :param inputs: input = X, (h0, c0)
        c0.shape == h0.shape = (batch, out_dim) X.shape == (seq, batch, in_dim)
        :return: outputs, (h, c): outputs.shape == (seq, batch, out_dim)
        """
        # 懒得检查shape了

        X = inputs[0]
        batch_size = X.shape[1]
        if inputs[1] is None:
            h = np.zeros((batch_size, self.out_dim))
            c = np.zeros((batch_size, self.out_dim))
        else:
            h = np.zeros((batch_size, self.out_dim)) if inputs[1][0] is None else inputs[1][0]
            c = np.zeros((batch_size, self.out_dim)) if inputs[1][1] is None else inputs[1][1]
        self.inputs = (X, (h, c))
        h_list = []

        for x in X:
            total = np.c_[h, x]
            u = self.u_gate(total)
            f = self.f_gate(total)
            o = self.o_gate(total)
            c_hat = self.perceptron(total)
            c = u * c_hat + f * c
            h = o * self.activation(c)
            h_list.append(h)
            self.c_list.append(c)
            self.c_hat_list.append(c_hat)
            self.u_list.append(u)
            self.f_list.append(f)
            self.o_list.append(o)
        self.outputs = np.stack(h_list, axis=0)
        return self.outputs

    def backward(self, output_gradient, inputs=None):
        """
        :param inputs: MUST BE NONE!!!!
        :param output_gradient: shape == (seq, batch, out_dim)
        :return: input_gradiant: shape == (seq, batch, in_dim)
        """
        in_grad_list = []
        seq_len = output_gradient.shape[0]
        batch_size = output_gradient.shape[1]

        total_gradient = np.zeros((batch_size, self.out_dim + self.in_dim))
        c_gradient = np.zeros((batch_size, self.out_dim))

        X = self.inputs[0]  # self.inputs = X, (h0, c0)
        h0, c0 = self.inputs[1]
        h = self.outputs

        o_w_grad = np.zeros(self.o_gate.weights().shape)
        o_b_grad = np.zeros(self.o_gate.bias().shape)
        f_w_grad = np.zeros(self.f_gate.weights().shape)
        f_b_grad = np.zeros(self.f_gate.bias().shape)
        u_w_grad = np.zeros(self.u_gate.weights().shape)
        u_b_grad = np.zeros(self.u_gate.bias().shape)
        p_w_grad = np.zeros(self.perceptron.weights().shape)
        p_b_grad = np.zeros(self.perceptron.bias().shape)

        # 此部分的原理和RNN相同，不再注释
        for i in range(seq_len - 1, -1, -1):
            h_gradient = output_gradient[i] + total_gradient[:, :self.out_dim]
            if i == seq_len - 1:
                c_gradient = self.o_list[i] * self.activation.backward(h_gradient)
            else:
                c_gradient = self.o_list[i] * self.activation.backward(h_gradient) \
                             + c_gradient * self.f_list[i + 1]
            c_hat_gradient = c_gradient * self.u_list[i]
            o_gradient = h_gradient * self.activation(self.c_list[i])
            u_gradient = c_gradient * self.c_list[i]
            f_gradient = c_gradient * (self.c_list[i - 1] if i != 0 else c0)

            inputs = np.c_[h[i - 1] if i != 0 else h0, X[i]]
            total_gradient = self.o_gate.backward(o_gradient, inputs) + \
                             self.f_gate.backward(f_gradient, inputs) + \
                             self.u_gate.backward(u_gradient, inputs) + \
                             self.perceptron.backward(c_hat_gradient, inputs)
            x_gradient = total_gradient[:, self.out_dim:]
            in_grad_list.append(x_gradient)

            o_w_grad += self.o_gate.w_gradient()
            o_b_grad += self.o_gate.b_gradient()
            f_w_grad += self.f_gate.w_gradient()
            f_b_grad += self.f_gate.b_gradient()
            u_w_grad += self.u_gate.w_gradient()
            u_b_grad += self.u_gate.b_gradient()
            p_w_grad += self.perceptron.w_gradient()
            p_b_grad += self.perceptron.b_gradient()

        self.o_gate.set_gradients(o_w_grad, o_b_grad)
        self.f_gate.set_gradients(f_w_grad, f_b_grad)
        self.u_gate.set_gradients(u_w_grad, u_b_grad)
        self.perceptron.set_gradients(p_w_grad, p_b_grad)
        list.reverse(in_grad_list)
        return np.stack(in_grad_list)

    def step(self, lr):
        self.o_gate.step(lr)
        self.u_gate.step(lr)
        self.f_gate.step(lr)
        self.perceptron.step(lr)

