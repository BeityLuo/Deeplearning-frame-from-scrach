from exceptions.exceptions import ShapeNotMatchException


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
