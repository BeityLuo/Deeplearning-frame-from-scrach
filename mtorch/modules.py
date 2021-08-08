class Module:
    def __init__(self, sequential):
        self.sequential = sequential

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # 这里要转置两次，因为外界统一用(batch_size, dim), 神经网络内部统一用(dim, batch_size)
        return self.sequential(x)

    def backward(self, output_delta):
        """
        执行反向传播
        :param output_delta: 模型输出的期望改变值（loss对module的输出求导）
        :return: None
        """
        self.sequential.backward(output_delta)

    def step(self, lr):
        """
        根据反向传播计算的值更新各个层的参数
        :param lr: 学习率
        :return:
        """
        for layer in self.sequential:
            layer.step(lr)
