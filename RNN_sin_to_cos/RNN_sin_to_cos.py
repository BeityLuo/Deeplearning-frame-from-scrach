# 定义网络

import numpy as np
import sys
sys.path.append(".") 
import matplotlib.pyplot as plt
import chart_drawer
import mtorch.modules
from mtorch import loss_functions, optimizers



# Hyper parameters
from mtorch.layers.layer import Sequential
from mtorch.layers.linear_layers import Linear3
from mtorch.layers.recurrent_layers import RNN, LSTM

INPUT_SIZE = 1
EPOCH_NUM = 2000  # 共训练EPOCH_NUM次 train for EPOCH_NUM times
TEST_STEP = 100  # 每训练TEST_STEP轮就测试一次 test for every TEST_STEP times train
SHOW_CHART_STEP = 5  # 每测试SHOW_CHART_STEP次就输出一次图像 draw a chart for every SHOW_CHART_STEP times test
learning_rate = 0.1  # 学习率
TIME_STEP = 10  # 时间序列长度
HIDDEN_SIZE = 32
batch_size = 32

# 定义网络
class RNNPractice(mtorch.modules.Module):
    def __init__(self):
        super(RNNPractice, self).__init__(Sequential([
            LSTM(INPUT_SIZE, HIDDEN_SIZE),
            Linear3(HIDDEN_SIZE, 1),
        ]))

    def forward(self, inputs):
        """
        :param inputs: inputs = (h0_state, X)
        :return:
        """
        # 由于需要两个输出值，所以就重写了一下，十分丑陋毫无拓展性
        output = self.sequential.layers[0](inputs)
        h_final = output[len(output) - 1]
        output = self.sequential.layers[1](output)
        return output, h_final


rnn = RNNPractice()
loss_func = loss_functions.SquareLoss(rnn.backward)
optimizer = optimizers.SGD(rnn, learning_rate, lr_decay=False)

drawer = chart_drawer.ChartDrawer()

def getSample(epoch):
    inputs = []
    targets = []
    for i in range(batch_size):
        # 获取样本，13和11是随便想的两个数字
        start = (epoch * batch_size + i) * 13 / 11 * np.pi
        end = (epoch * batch_size + i + 1) * 13 / 11 * np.pi
        axis = np.linspace(start, end, TIME_STEP)  # 坐标x的值, 将[start, end]用TIME_STEP个等间隔的点分开
        inputs.append(np.sin(axis)[:, np.newaxis])
        targets.append(np.cos(axis)[:, np.newaxis])

    inputs = np.stack(inputs, 1)  # transform list to matrix
    targets = np.stack(targets, 1)
    return inputs, targets

plt.figure(1, figsize=(12, 5))
plt.ion()  # continuously plot
losses = []
epochs = []


zero_state = np.zeros((batch_size, HIDDEN_SIZE))  # the shape of one time step of RNN's output is (batch_size, HIDDEN_SIZE)
# h_state = np.random.randn(batch_size, HIDDEN_SIZE)
h_state = zero_state

for i in range(1, EPOCH_NUM + 1):
    inputs, targets = getSample(i)
    outputs, h_state = rnn((inputs, None))
    loss = loss_func(outputs, targets)
    loss.backward()
    optimizer.step()

    if i % TEST_STEP == 0:
        losses.append(loss.value)
        epochs.append(i)
        print("epoch = {}, loss = {}".format(
            i, loss.value))
        if i % (TEST_STEP * SHOW_CHART_STEP) == 0:
            start = (i * batch_size) * 13 / 11 * np.pi
            end = (i * batch_size + 1) * 13 / 11 * np.pi
            axis = np.linspace(start, end, TIME_STEP)
            drawer.draw_2lines(axis, np.ndarray.flatten(targets[:,0]), np.ndarray.flatten(outputs[:,0]))
            plt.plot(axis, targets[:,0], 'b-')
            plt.plot(axis, inputs[:, 0], 'g-')
            plt.plot(axis, np.ndarray.flatten(outputs[:,0]), 'r-')
            plt.draw()
            plt.pause(0.05)
            if i % (TEST_STEP * SHOW_CHART_STEP * 2) == 0:
                plt.plot(epochs, losses)
                plt.draw()
                plt.pause(0.05)

plt.plot(epochs, losses)
plt.draw()
plt.pause(0.05)
plt.ioff()
plt.show()

print("training finished: loss = {}".format(loss.value))
# for i in range(batch_size):
#     start = ((EPOCH_NUM - 1) * batch_size + i) * 13 / 11 * np.pi
#     end = ((EPOCH_NUM - 1) * batch_size + i + 1) * 13 / 11 * np.pi
#     axis = np.linspace(start, end, TIME_STEP)
#
# # print("turns = {}, loss now is {}".format(i, loss.value))
#     drawer.draw_2lines(axis, np.ndarray.flatten(targets[:,i]), np.ndarray.flatten(outputs[:,i]))