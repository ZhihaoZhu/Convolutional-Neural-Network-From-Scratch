import pickle
import numpy as np

def relu(x):
    sig = lambda p: p if p >= 0 else 0
    sigfunc = np.vectorize(sig)
    res = sigfunc(x)
    return res


a = np.random.rand(4,4,4,4)-0.5
print(a[0,0,:,:])
x = relu(a)
print(x[0,0,:,:])




'''
    read data
'''
# import pickle
# import matplotlib.pyplot as plt
#
# with open("../data/sampledCIFAR10", "rb") as f :
#     data = pickle.load(f)
#     train, val, test = data["train"], data["val"], data["test"]
#     n_samples = train["data"].shape[0]
#     train["data"] = train["data"].reshape(n_samples, 3, 32, 32)
#     print(train["data"].shape)
#     image = train["data"][9,:,:,:].transpose(1,2,0)
#     plt.imshow(image)
#     plt.show()

'''
    test1
'''
# def im2col_sliding(image, block_size, skip=1):
#
#     rows, cols = image.shape
#     horz_blocks = cols - block_size[1] + 1
#     vert_blocks = rows - block_size[0] + 1
#
#     output_vectors = np.zeros((block_size[0] * block_size[1], horz_blocks * vert_blocks))
#     itr = 0
#     for v_b in range(vert_blocks):
#         for h_b in xrange(horz_blocks):
#             output_vectors[:, itr] = image[v_b: v_b + block_size[0], h_b: h_b + block_size[1]].ravel()
#             itr += 1
#
#     return output_vectors[:, ::skip]

'''
    test2
'''
# a = np.array([[1, 2], [3, 4]])
# b = a.repeat(2, axis=0).repeat(2,axis=1)
# print(b)