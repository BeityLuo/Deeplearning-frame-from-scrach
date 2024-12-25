import numpy as np
import struct#处理二进制文件


#将图片信息从idx3-ubyte文件中解析出来，将标签信息从idx1-ubyte文件中解析出来，其中idx3的3代表是三维，idx1的1代表是一维
# 训练集文件
train_images_idx3_ubyte_file = './dataset/train-images.idx3-ubyte'
# 训练集标签文件
train_labels_idx1_ubyte_file = './dataset/train-labels.idx1-ubyte'

# 测试集文件
test_images_idx3_ubyte_file = './dataset/t10k-images.idx3-ubyte'
# 测试集标签文件
test_labels_idx1_ubyte_file = './dataset/t10k-labels.idx1-ubyte'

#解析idx3文件
def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3文件的通用函数
    :param idx3_ubyte_file: idx3文件路径
    :return: 数据集，图片数量
    """
    # 读取二进制数据
    bin_data = open(idx3_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽，这里未对魔数进行存储，因此struct.unpack_from的第一个返回值为_
    offset = 0
    fmt_header = '>iiii'#这里的i代表整型数据，四个i代表有四个整形数字，>是为了解决对齐问题加入的字符
    #struct.unpack_from(fmt=,buffer=,offfset=)该函数可以将缓冲区buffer中的内容在按照指定的格式fmt='somenformat'，从偏移量为offset=numb的位置开始进行读取。返回的是一个对应的元组tuple
    _, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print('图片数量: %d张, 图片大小: %d*%d' % (num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)#calcsize(fmt) -> integer  计算给定的格式(fmt)占用多少字节的内存
    fmt_image = '>' + str(image_size) + 'B'#读取一张图片需要的字节数
    images = np.empty((num_images, num_rows * num_cols))
    for i in range(num_images):
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows * num_cols))
        offset += struct.calcsize(fmt_image)
    return images

#解析idx1文件
def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1文件的通用函数
    :param idx1_ubyte_file: idx1文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()

    # 解析文件头信息，依次为魔数和标签数，这里的魔数也未存储
    offset = 0
    fmt_header = '>ii'
    _, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    # print('图片数量: %d张' % (num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return np.reshape(labels, (num_images, 1))

#获取训练数据
def load_train_images(idx_ubyte_file=train_images_idx3_ubyte_file):
    """
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)

#获取训练标签
def load_train_labels(idx_ubyte_file=train_labels_idx1_ubyte_file):
    """
    TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

#获取测试数据
def load_test_images(idx_ubyte_file=test_images_idx3_ubyte_file):
    """
    TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  10000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

    :param idx_ubyte_file: idx文件路径
    :return: n*row*col维np.array对象，n为图片数量
    """
    return decode_idx3_ubyte(idx_ubyte_file)

#获取测试标签
def load_test_labels(idx_ubyte_file=test_labels_idx1_ubyte_file):
    """
    TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.

    :param idx_ubyte_file: idx文件路径
    :return: n*1维np.array对象，n为图片数量
    """
    return decode_idx1_ubyte(idx_ubyte_file)

def load_idx1_file(file_path):
    return

def run():
    train_images,train_nums = load_train_images()
    train_labels = load_train_labels()
    test_images,test_nums = load_test_images()
    test_labels = load_test_labels()

    return train_images,train_labels,test_images,test_labels,train_nums,test_nums


