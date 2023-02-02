from numba import jit
from numba import cuda
from numba import vectorize
import numpy as np
from timeit import default_timer as timer
import math

@jit(nopython=True)
def reshape_output(output):
    #transform input (shape # molecules, # assays, # data) into (# molecules, # assays x # data)
    processed_output = np.zeros((len(output), 229432)) 
    #pdb.set_trace()
    for i in range(len(output)): # for each uneven 2d array in output
        counter = 0
        if (i % 500 == 0):
            print(i)
        for j in range(len(output[i])): #convert into 1d array
            for k in range(len(output[i][j])):
                processed_output[i][counter] = output[i][j][k]
                counter += 1
    return processed_output

reshape_output(np.array([ [ [1, 2], [3, 4]  ],  [ [5, 7], [7, 8] ] ]))


# # To run on CPU
# def func(a):
#     for i in range(10000000):
#         a[i]+= 1
# # To run on GPU
# @jit
# def func2(a):
#     for i in range(10000000):
#         a[i]+= 1
# if __name__=="__main__":
#     n = 10000000
#     a = np.ones(n, dtype = np.float64)
#     start = timer()
#     func(a)
#     print("without GPU:", timer()-start)
#     start = timer()
#     print("here")
#     func2(a)
#     #cuda.profile_stop()
#     print("with GPU:", timer()-start)

# from ctypes import CDLL, POINTER, byref, c_size_t
# import sys

# cudart = CDLL('libcudart.so')

# cudaMemGetInfo = cudart.cudaMemGetInfo
# cudaMemGetInfo.argtypes = [POINTER(c_size_t), POINTER(c_size_t)]

# free = c_size_t()
# total = c_size_t()
# err = cudaMemGetInfo(byref(free), byref(total))

# if (err):
#     print(f"cudaMemGetInfo error is {err}")
#     sys.exit(1)

# print(f"Free: {free.value} bytes")
# print(f"Total: {total.value} bytes")

# print("Importing Numba")

# from numba import cuda

# print("Allocating array")

# cuda.device_array(1)

# print("Finished")