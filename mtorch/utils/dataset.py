from MINST_recognition import IDX_reader
import numpy as np


# from macros import ROW_NUM, COLUM_NUM
# import numpy
# import numpy.ctypeslib as ctlib

# from ctypes import *
#
# file_reader = ctlib.load_library("IDX_reader", "../..")
# file_reader.next_train_img.argtypes = [
#     ctlib.ndpointer(dtype=numpy.ubyte, ndim=2, shape=(ROW_NUM, COLUM_NUM), flags="C_CONTIGUOUS")]
# file_reader.next_test_img.argtypes = [
#     ctlib.ndpointer(dtype=numpy.ubyte, ndim=2, shape=(ROW_NUM, COLUM_NUM), flags="C_CONTIGUOUS")]
# file_reader.show_image.argtypes = [
#     ctlib.ndpointer(dtype=numpy.ubyte, ndim=2, shape=(ROW_NUM, COLUM_NUM), flags=("C_CONTIGUOUS"))]
#
# file_reader.load()
#
# item_num = file_reader.get_train_item_num()
# print("There are {} imgs".format(item_num))
# for i in range(5):
#     label = file_reader.next_train_label()
#     data = numpy.zeros((ROW_NUM, COLUM_NUM), dtype=numpy.ubyte)
#     file_reader.next_train_img(data)
#     print("train set {}: label = {}".format(i, label))
#     file_reader.show_image(data)
#
# for i in range(5):
#     label = file_reader.next_test_label()
#     data = numpy.zeros((ROW_NUM, COLUM_NUM), dtype=numpy.ubyte)
#     file_reader.next_test_img(data)
#     print("test set {}: label = {}".format(i, label))
#     file_reader.show_image(data)


class Dataset:
    def __init__(self):
        self.size = None
        pass

    def set_idx(self, idx):
        pass

    def __len__(self):
        return self.size


class FileDataset(Dataset):
    def __init__(self, img_path, label_path):
        super(FileDataset, self).__init__()
        self.images = self.normalize((IDX_reader.decode_idx3_ubyte(img_path)))  # 正则化输入
        self.labels = (IDX_reader.decode_idx1_ubyte(label_path))
        self.image_idx = 0
        self.label_idx = 0
        self.size = self.images.shape[0]
        # print("Dataset size = {}".format(self.size))

    def get_next_labels(self, num):
        """
        获取接下来的num个标签，如果不足num个，就返回剩下的；如果一个都没有了，就返回None
        :param num:
        :return:
        """
        if self.label_idx + num <= self.size:
            slide = self.labels[np.arange(self.label_idx, self.label_idx + num)]
            self.label_idx += num
        elif self.label_idx < self.size:
            slide = self.labels[np.arange(self.label_idx, self.size)]
            self.label_idx = self.size
        else:
            slide = None
        return slide

    def get_next_imgs(self, num):
        if self.image_idx + num <= self.size:
            slide = self.images[np.arange(self.image_idx, self.image_idx + num)]
            self.image_idx += num
        elif self.image_idx < self.size:
            slide = self.images[np.arange(self.image_idx, self.size)]
            self.image_idx = self.size
        else:
            slide = None
        return slide

    def set_idx(self, idx):
        self.image_idx = self.label_idx = idx

    def normalize(self, array):
        aver = np.mean(array, axis=1).reshape((array.shape[0], 1))
        sigma = np.std(array, axis=1).reshape((array.shape[0], 1))
        return (array - aver) / sigma

    def shuffle(self):
        random_state = np.random.get_state()
        np.random.shuffle(self.images)
        np.random.set_state(random_state)
        np.random.shuffle(self.labels)
