import numpy.ctypeslib as ctlib
import numpy
import ctypes
ROW_NUM, COLUM_NUM = 28, 28
# reader = ctlib.load_library("IDX_reader.dll", ".")
reader = ctypes.cdll.LoadLibrary("./IDX_reader.dll")
reader.print_test()
reader.load()
reader.next_train_img.argtypes = [
        ctlib.ndpointer(dtype=numpy.ubyte, ndim = 2, shape = (ROW_NUM, COLUM_NUM), flags=("C"))]
reader.show_image.argtypes = [
        ctlib.ndpointer(dtype=numpy.ubyte, ndim = 2, shape = (ROW_NUM, COLUM_NUM), flags=("C"))]
data = numpy.zeros((ROW_NUM, COLUM_NUM), dtype=numpy.ubyte)
reader.next_train_img(data)
reader.close()

import numpy as np
a = np.array([1, 2, 3])
w = np.array([1, 2, 3])
np.dot(w, a)




