import numpy as np
from matplotlib import pyplot as plt


class ChartDrawer:
    """
    画图表的
    """
    def __init__(self, start=None, end=None, step=1):
        """
        :param start: 坐标轴起点
        :param end: 坐标轴重点
        :param step: 绘图步长
        """
        self.losses = []
        self.accuracies = []
        if start is not None and end is not None:
            self.turns = np.linspace(start, end, (end - start) // step)
            for i in range((end - start) // step):
                self.losses.append(None)
                self.accuracies.append(None)
        self.colors = ('b', 'g', 'r', 'c', 'm', 'y', 'k', 'w')

    def draw(self, start=None, end=None, step=1):
        if start is not None and end is not None:
            self.turns = np.linspace(start, end, (end - start) // step)
        plt.clf()
        fig, loss_axis = plt.subplots()
        accuracy_axis = loss_axis.twinx()
        loss_axis.set_ylabel("loss of test set", color="g")
        accuracy_axis.set_ylabel("accuracy of test set", color="b")
        loss_axis.set_xlabel("training turns")
        loss_axis.plot(self.turns, self.losses, "g")
        accuracy_axis.plot(self.turns, self.accuracies, "b")
        plt.show()

    def draw_lines(self, x, y_list):
        plt.clf()
        fig, axis = plt.subplots()
        axis.set_xlabel("x")
        for i in range(len(y_list)):
            y_axis = axis.twinx()
            y_axis.set_ylabel("y{}".format(i), color=self.colors[i % len(self.colors)])  # 最多不超过8种颜色
            y_axis.plot(x, y_list[i], self.colors[i % len(self.colors)])
        plt.show()

    def draw_2lines(self, x, y0, y1):
        plt.clf()
        fig, axis0 = plt.subplots()
        axis1 = axis0.twinx()

        axis0.set_xlabel("x")
        axis0.set_ylabel("y0", color="g")
        axis1.set_ylabel("y1", color="r")
        axis0.plot(x, y0, color="g")
        axis1.plot(x, y1, color="r")
        plt.show()
