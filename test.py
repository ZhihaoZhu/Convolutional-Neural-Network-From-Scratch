import pickle
import numpy as np




def im2col(x, kernel_size, padding=1, stride=1):
  x_padded = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
  input_size = x.shape[2]
  C = x.shape[1]
  out_size = int((input_size + 2 * padding - kernel_size) / stride + 1)
  l = np.repeat(np.arange(C), kernel_size * kernel_size).reshape(-1, 1)

  m1 = np.repeat(np.arange(kernel_size), kernel_size)
  m1 = np.tile(m1, C)
  m2 = stride * np.repeat(np.arange(out_size), out_size)

  n1 = np.tile(np.arange(kernel_size), kernel_size * C)
  n2 = stride * np.tile(np.arange(out_size), out_size)

  m = m1.reshape(-1, 1) + m2.reshape(1, -1)
  n = n1.reshape(-1, 1) + n2.reshape(1, -1)

  cols = x_padded[:, l, m, n]
  print(cols.shape)
  cols = cols.transpose(1, 2, 0).reshape(kernel_size * kernel_size * C, -1)
  print(cols.shape)
  return cols

x = np.arange(30000).reshape(100,3,10,10)
col = im2col(x, 3, padding=2, stride=1)
print(col)

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