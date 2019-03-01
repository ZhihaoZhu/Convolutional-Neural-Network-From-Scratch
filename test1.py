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
    cols = cols.transpose(1, 2, 0).reshape(kernel_size * kernel_size * C, -1)
    return cols, out_size


def max_pooling_forward(x):
    batch_number, C, _, _ = x.shape
    cols, out_size = im2col(x, 2, padding=0, stride=2)
    print(cols.shape)
    cols_w, cols_h = cols.shape
    cols.reshape(C, cols_w / C, -1)  # 注意im2col的定义，这里的col已经被reshape过了
    MP_index = np.max(cols, axis=1)


a = np.arange(27).reshape(100,3,100,100)
max_pooling_forward(a)



