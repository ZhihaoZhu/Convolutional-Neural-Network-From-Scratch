import numpy as np
import os
import pickle
from nn import *

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
  # First figure out what the size of the output should be
  N, C, H, W = x_shape
  assert (H + 2 * padding - field_height) % stride == 0
  assert (W + 2 * padding - field_height) % stride == 0
  out_height = int((H + 2 * padding - field_height) / stride + 1)
  out_width = int((W + 2 * padding - field_width) / stride + 1)

  i0 = np.repeat(np.arange(field_height), field_width)
  i0 = np.tile(i0, C)
  i1 = stride * np.repeat(np.arange(out_height), out_width)
  j0 = np.tile(np.arange(field_width), field_height * C)
  j1 = stride * np.tile(np.arange(out_width), out_height)
  i = i0.reshape(-1, 1) + i1.reshape(1, -1)
  j = j0.reshape(-1, 1) + j1.reshape(1, -1)

  k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

  return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
  """ An implementation of im2col based on some fancy indexing """
  # Zero-pad the input
  p = padding

  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

  k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                               stride)

  cols = x_padded[:, k, i, j]
  C = x.shape[1]
  cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
  return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
  """ An implementation of col2im based on fancy indexing and np.add.at """
  N, C, H, W = x_shape
  H_padded, W_padded = H + 2 * padding, W + 2 * padding
  x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
  k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
  cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
  cols_reshaped = cols_reshaped.transpose(2, 0, 1)
  np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
  if padding == 0:
    return x_padded
  return x_padded[:, :, padding:-padding, padding:-padding]


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
    cols = cols.transpose(1, 2, 0).reshape(kernel_size * kernel_size * C, -1)
    return cols, out_size

def col2im(cols, x_shape, kernel_size=3, padding=1,stride=1):
    batch_number, C, H, W = x_shape
    x_padded = np.zeros((batch_number, C, H + 2 * padding, W + 2 * padding))

    out_size = int((H + 2 * padding - kernel_size) / stride + 1)
    l = np.repeat(np.arange(C), kernel_size * kernel_size).reshape(-1, 1)

    m1 = np.repeat(np.arange(kernel_size), kernel_size)
    m1 = np.tile(m1, C)
    m2 = stride * np.repeat(np.arange(out_size), out_size)

    n1 = np.tile(np.arange(kernel_size), kernel_size * C)
    n2 = stride * np.tile(np.arange(out_size), out_size)
    m = m1.reshape(-1, 1) + m2.reshape(1, -1)
    n = n1.reshape(-1, 1) + n2.reshape(1, -1)

    cols_reshaped = cols.reshape(C * kernel_size * kernel_size, -1, batch_number)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), l, m, n), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


x = np.arange(64).reshape(4,1,4,4)
padding = 0
MP_stride = 2


def max_pooling(x, MP_stride):
    batch_number, C, W, H = x.shape
    x = x.reshape(batch_number*C,1, W, H)
    cols, out_size = im2col(x, MP_stride, padding=0, stride=MP_stride)
    MP_index = np.argmax(cols, axis=0)
    MP_value = cols[MP_index,range(MP_index.shape[0])]
    MP_value = MP_value.reshape(out_size,out_size,batch_number,C)
    MP_value = MP_value.transpose(2, 3, 0, 1)

    return MP_value

print(x)
print("-----------------------")
dd = max_pooling(x, MP_stride)
print(dd)



# '''
#     my method
# '''
#
# kernel_size = kernels.shape[2]
# cols,out_size = im2col(x, kernel_size, padding=padding, stride=conv_stride)
# kernel_cols = kernels.reshape(kernels.shape[0],-1)
# output = kernel_cols @ cols
# out = output.reshape(kernels.shape[0], out_size, out_size, x.shape[0])
# out = out.transpose(3, 0, 1, 2)
#
#
#
#
# '''
#     others method
# '''
# n_filters, d_filter, h_filter, w_filter = kernels.shape
# n_x, d_x, h_x, w_x = x.shape
# h_out = (h_x - h_filter + 2 * padding) / conv_stride + 1
# w_out = (w_x - w_filter + 2 * padding) / conv_stride + 1
#
# h_out, w_out = int(h_out), int(w_out)
# X_col = im2col_indices(x, h_filter, w_filter, padding=padding, stride=conv_stride)
# W_col = kernels.reshape(n_filters, -1)
#
# out1 = W_col @ X_col
# out1 = out1.reshape(n_filters, h_out, w_out, n_x)
# out1 = out1.transpose(3, 0, 1, 2)
#
#
# print(np.array_equal(out, out1))

