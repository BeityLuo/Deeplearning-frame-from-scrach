# 定义网络
import numpy as np

import chart_drawer
import mtorch.modules
from mtorch import layers, loss_functions, optimizers

INPUT_SIZE = 1
EPOCH_NUM = 20  # 共训练EPOCH_NUM次 train for EPOCH_NUM times
TEST_STEP = 1  # 每训练TEST_STEP轮就测试一次 test for every TEST_STEP times train
SHOW_CHART_STEP = 10  # 每测试SHOW_CHART_STEP次就输出一次图像 draw a chart for every SHOW_CHART_STEP times test
learning_rate = 0.1  # 学习率
TIME_STEP = 10  # 时间序列长度
HIDDEN_SIZE = 32
batch_size = 15

# 定义网络
class RNNPractice(mtorch.modules.Module):
    def __init__(self):
        super(RNNPractice, self).__init__(layers.Sequential([
            layers.RNN(INPUT_SIZE, HIDDEN_SIZE),
            layers.Linear3(HIDDEN_SIZE, 1),
            layers.Sigmoid(1)
        ]))

    # def forward(self, inputs):
    #     """
    #     :param inputs: inputs = (h0_state, X)
    #     :return:
    #     """
    #     # 由于需要两个输出值，所以就重写了一下，十分丑陋毫无拓展性
    #     output = self.sequential.layers[0](inputs)
    #     h_final = output[len(output) - 1]
    #     output = self.sequential.layers[1](output)
    #     output = self.sequential.layers[2](output)
    #     return output, h_final


rnn = RNNPractice()
loss_func = loss_functions.SquareLoss(rnn.backward)
optimizer = optimizers.SGD(rnn, learning_rate)

drawer = chart_drawer.ChartDrawer()

h_state = np.zeros((batch_size, HIDDEN_SIZE))


def getSample(epoch):
    inputs = []
    targets = []
    for i in range(batch_size):
        # 获取样本，13和11是随便想的两个数字
        start = (epoch * batch_size + i) * 13 / 11 * np.pi
        end = (epoch * batch_size + i + 1) * 13 / 11 * np.pi
        axis = np.linspace(start, end, TIME_STEP)  # 坐标x的值
        inputs.append(np.sin(axis)[:, np.newaxis, np.newaxis])
        targets.append(np.cos(axis)[:, np.newaxis, np.newaxis])

    inputs = np.concatenate(inputs, 1)
    targets = np.concatenate(targets, 1)
    return inputs, targets

for i in range(EPOCH_NUM):
    inputs, targets = getSample(i)
    outputs = rnn((h_state, inputs))  #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    loss = loss_func(outputs, targets)
    loss.backward()
    optimizer.step()

    if i % TEST_STEP == 0:
        start = (i * batch_size) * 13 / 11 * np.pi
        end = (i * batch_size + 1) * 13 / 11 * np.pi
        axis = np.linspace(start, end, TIME_STEP)
        print("i = {}, TEST_STEP = {}, loss = {}".format(
            i, TEST_STEP, loss.value))
    # print("turns = {}, loss now is {}".format(i, loss.value))
        drawer.draw_2lines(axis, np.ndarray.flatten(targets[:,1]), np.ndarray.flatten(outputs[:,1]))