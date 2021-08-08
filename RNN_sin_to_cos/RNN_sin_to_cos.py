# 定义网络
import numpy as np

import chart_drawer
import mtorch.modules
from mtorch import layers, loss_functions, optimizers

INPUT_SIZE = 1
EPOCH_NUM = 10000  # 共训练EPOCH_NUM次 train for EPOCH_NUM times
TEST_STEP = 500  # 每训练TEST_STEP轮就测试一次 test for every TEST_STEP times train
SHOW_CHART_STEP = 10  # 每测试SHOW_CHART_STEP次就输出一次图像 draw a chart for every SHOW_CHART_STEP times test
learning_rate = 0.1  # 学习率
TIME_STEP = 10  # 时间序列长度
HIDDEN_SIZE = 32


class RNNPractice(mtorch.modules.Module):
    def __init__(self):
        super(RNNPractice, self).__init__(layers.Sequential([
            layers.RNN(INPUT_SIZE, HIDDEN_SIZE),
            layers.Linear3(HIDDEN_SIZE, 1),
            layers.Sigmoid(1)
        ]))

    def forward(self, inputs):
        """
        :param inputs: inputs = (h0_state, X)
        :return:
        """
        output = self.sequential.layers[0](inputs)
        h_final = output[len(output) - 1]
        output = self.sequential.layers[1](output)
        output = self.sequential.layers[2](output)
        return output, h_final


rnn = RNNPractice()
loss_func = loss_functions.SquareLoss(rnn.backward)
optimizer = optimizers.SGD(rnn, learning_rate)

drawer = chart_drawer.ChartDrawer()

h_state = np.zeros((1, HIDDEN_SIZE))
for i in range(EPOCH_NUM):
    start = i * 13 / 11 * np.pi
    end = (i + 1) * 13 / 11 * np.pi
    axis = np.linspace(start, end, TIME_STEP)
    inputs = np.sin(axis)[:, np.newaxis, np.newaxis]
    targets = np.cos(axis)[:, np.newaxis, np.newaxis]

    outputs, h_state = rnn((h_state, inputs))
    if i % TEST_STEP == 0:
        print("i = {}, TEST_STEP = {}, i % TEST_STEP = {}".format(i, TEST_STEP, i % TEST_STEP))
        # print("turns = {}, loss now is {}".format(i, loss.value))
        drawer.draw_2lines(axis, np.ndarray.flatten(targets), np.ndarray.flatten(outputs))
    loss = loss_func(outputs, targets)
    loss.backward()
    optimizer.step()

