"""
只是对pytorch的拙劣模仿，没有任何实际意义 QAQ
"""
class SGD():
    def __init__(self, module, lr):
        self.module = module
        self.lr = lr

    def step(self):
        self.module.step(self.lr)
