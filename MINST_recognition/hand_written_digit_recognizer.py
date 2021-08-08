from chart_drawer import ChartDrawer
from mtorch import layers
from mtorch.layers import Sequential
from mtorch.loss_functions import SquareLoss
from mtorch.modules import Module
from mtorch.optimizers import SGD
from mtorch.utils.dataloader import DataLoader
from mtorch.utils.dataset import FileDataset
from macros import *

EPOCH_NUM = 4000  # 共训练EPOCH_NUM次 train for EPOCH_NUM times
TEST_STEP = 50  # 每训练TEST_STEP轮就测试一次 test for every TEST_STEP times train
SHOW_CHART_STEP = 50  # 每测试SHOW_CHART_STEP次就输出一次图像 draw a chart for every SHOW_CHART_STEP times test
learning_rate = 0.1  # 学习率
batch_size = 20


drawer = ChartDrawer(0, EPOCH_NUM, TEST_STEP)



def show_effect(epoch, module, loss_fun, test_loader, step):
    """
    要输出的：已经循环的轮数、当前的测试集loss、当前的测试集准确率
    :param epoch: 循环的轮数
    :param module: 神经网络模型
    :param loss_fun: 使用的损失函数
    :param test_loader: 测试集的loader
    :return:
    """
    print("——————————————————第{}轮————————————————————".format(epoch))

    test_size = len(test_set)
    total_loss = 0
    total_accuracy = 0
    for data in test_loader:
        imgs, targets = data
        outputs = module(imgs)
        loss = loss_fun(outputs, targets, transform=True)
        total_loss += loss.value
        accuracy = (outputs.argmax(1).reshape(targets.shape) == targets).sum()
        total_accuracy += accuracy

    accuracy = total_accuracy / test_size
    drawer.losses[step] = total_loss
    drawer.accuracies[step] = accuracy
    print("test loss = {}".format(total_loss))
    print("test accuracy = {}%".format(accuracy * 100))
    show_image(test_img)
    if step % SHOW_CHART_STEP == 0:
        drawer.draw()


def show_image(img):
    """
    打印一张灰度图片
    :param img: 一副正则化后的图片
    :return:
    """
    row, column = img.shape
    for i in range(row):
        for j in range(column):
            if img[i][j] > 0:
                print("  ", end="")
            else:
                print("■ ", end="")
        print("")




# 准备数据集 prepare dataset
dataset_path = "../dataset"
train_img_path = dataset_path + "/" + TRAIN_IMG
train_label_path = dataset_path + "/" + TRAIN_LABEL
train_set = FileDataset(train_img_path, train_label_path)

test_img_path = dataset_path + "/" + TEST_IMG
test_label_path = dataset_path + "/" + TEST_LABEL
test_set = FileDataset(test_img_path, test_label_path)

# 加载数据集 load dataset
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
print("Finished loader dataset")


# 定义网络 define neural network
class DigitModule(Module):
    def __init__(self):
        sequential = Sequential([
            layers.Linear2(in_dim=ROW_NUM * COLUM_NUM, out_dim=16, coe=2),
            layers.Relu(16),
            layers.Linear2(in_dim=16, out_dim=16, coe=2),
            layers.Relu(16),
            layers.Linear2(in_dim=16, out_dim=CLASS_NUM, coe=1),
            layers.Sigmoid(CLASS_NUM)

            # layers.Linear(in_dim=ROW_NUM * COLUM_NUM, out_dim=100),
            # layers.Sigmoid(100),
            # layers.Linear(in_dim=100, out_dim=CLASS_NUM),
            # layers.Sigmoid(CLASS_NUM)
        ])
        super(DigitModule, self).__init__(sequential)


module = DigitModule()  # 创建模型 create module
loss_func = SquareLoss(backward_func=module.backward)  # 定义损失函数 define loss function
optimizer = SGD(module, lr=learning_rate)  # 定义优化器 define optimizer

for i in range(EPOCH_NUM):
    trainning_loss = 0
    for data in train_loader:
        """
        imgs.shape = (batch_size, 28 * 28)
        targets = (batch_size, 1)
        """
        imgs, targets = data
        # test_img = imgs[0].reshape((28, 28))
        # show_image(test_img)
        outputs = module(imgs)
        loss = loss_func(outputs, targets, transform=True)  # 计算loss calculate loss
        trainning_loss += loss.value
        loss.backward()  # 通过反向传播计算梯度 calculate gradiant through back propagation
        optimizer.step()  # 通过优化器调整模型参数 adjust the weights of network through optimizer
    if i % TEST_STEP == 0:
        show_effect(i, module, loss_func, test_loader, i // TEST_STEP)
        print("{} turn finished, loss of train set = {}".format(i, trainning_loss))
