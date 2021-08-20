import numpy as np

from exceptions.exceptions import ShapeNotMatchException
from mtorch.layers.layer import Layer
from mtorch.layers.linear_layers import Linear2
from mtorch.layers.non_linear_activations import Tanh


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
            # 前向传播顺序: x, h -> z -> h, 所以反向传播计算顺序：h_grad -> z_grad -> x_grad, h_grad, w_grad, b_grad
            # %%%%%%%%%%%%%%计算平均值的版本%%%%%%%%%%%%%%%%%%%%%%%
            # h_gradient = (output_gradiant[i] + total_gradient[:, 0:self.out_dim]) / 2
            #  计算h_grad: 这一时间点的h_grad包括输出的grad和之前的时间点计算所得grad两部分
            h_gradient = (output_gradiant[i] + total_gradient[:, 0:self.out_dim])

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
        self.linear.set_gradients(w=weight_gradients, b=bias_gradients)  # 设置梯度值

        list.reverse(input_gradients)  # input_gradients是逆序的，最后输出时需要reverse一下
        # print("sum(weight_gradients) = {}".format(np.sum(np.abs(weight_gradients))))
        # print("sum(h0_gradients) = {}".format(np.sum(np.abs(total_gradient[:, 0:self.out_dim]))))
        # pos_cnt = 0
        # neg_cnt = 0
        # for i in output_gradiant:
        #     for j in i:
        #         for k in j:
        #             if k > 0:
        #                 pos_cnt += 1
        #             elif k < 0:
        #                 neg_cnt += 1
        # print("output_gradiant has {} pos, {} neg".format(pos_cnt, neg_cnt))


        # np.stack的作用是将列表转变成一个矩阵
        return np.stack(input_gradients), h_gradient

    def step(self, lr):
        self.linear.step(lr)
        self.activation.step(lr)

class LSTM(Layer):
    def __init__(self, in_dim, out_dim):
        super(LSTM, self).__init__(in_dim, out_dim)
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