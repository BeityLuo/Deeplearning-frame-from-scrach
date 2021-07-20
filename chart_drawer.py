import numpy as np
from matplotlib import pyplot as plt


class ChartDrawer:
    """
    画图表的
    """
    def __init__(self, start, end, step):
        self.turns = np.linspace(start, end, (end - start) // step)
        self.losses = []
        self.accuracies = []
        for i in range((end - start) // step):
            self.losses.append(None)
            self.accuracies.append(None)

    def draw(self):
        plt.clf()
        fig, loss_axis = plt.subplots()
        accuracy_axis = loss_axis.twinx()
        loss_axis.set_ylabel("loss of test set", color="g")
        accuracy_axis.set_ylabel("accuracy of test set", color="b")
        loss_axis.set_xlabel("training turns")
        loss_axis.plot(self.turns, self.losses, "g")
        accuracy_axis.plot(self.turns, self.accuracies, "b")
        plt.show()