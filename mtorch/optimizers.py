"""
只是对pytorch的拙劣模仿，没有任何实际意义 QAQ
"""


class SGD():
    def __init__(self, module, lr, lr_decay=False):
        """
        :param module: module to optimize
        :param lr: learning rate
        :param lr_decay: if true, using learning rate decay
        """
        self.module = module
        self.lr = lr
        self.lr_decay = lr_decay
        if lr_decay:
            self.epoch = 0

    def step(self):
        if self.lr_decay:
            self.epoch += 1
            lr = self.lr / (self.epoch ** 0.5)
        else:
            lr = self.lr
        self.module.step(lr)
