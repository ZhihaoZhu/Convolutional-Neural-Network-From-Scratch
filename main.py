import numpy as np
import os
import pickle
from nn import *

# Apply the im2col method
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



def convolution(x,kernels,padding,conv_stride, layer):
    kernel_size = kernels.shape[2]
    cols,out_size = im2col(x, kernel_size, padding=padding, stride=conv_stride)
    params[layer]["x_shape"] = x.shape

    params[layer]["cols"] = cols
    kernel_cols = kernels.reshape(kernels.shape[0],-1)
    output = kernel_cols @ cols
    out = output.reshape(kernels.shape[0], out_size, out_size, x.shape[0])
    out = out.transpose(3, 0, 1, 2)

    return out

def relu(x,layer):
    params[layer]["pre_activ"] = x
    sig = lambda p: p if p >= 0 else 0
    sigfunc = np.vectorize(sig)
    res = sigfunc(x)
    return res

def relu_deriv(x):
    sig = lambda p: 1 if p >= 0 else 0
    sigfunc = np.vectorize(sig)
    res = sigfunc(x)
    return res

def max_pooling(x, MP_stride, layer):
    batch_number, C, W, H = x.shape
    params[layer]["pre_MP_size"] = x.shape
    x = x.reshape(batch_number*C,1, W, H)
    cols, out_size = im2col(x, MP_stride, padding=0, stride=MP_stride)
    params[layer]["cols_size"] = cols.shape

    MP_index = np.argmax(cols, axis=0)
    params[layer]["MP_index"] = MP_index
    MP_value = cols[MP_index,range(MP_index.shape[0])]
    MP_value = MP_value.reshape(out_size,out_size,batch_number,C)
    MP_value = MP_value.transpose(2, 3, 0, 1)
    return MP_value

def convolution_forward(x,kernels,padding,conv_stride,MP_stride,layer):
    conv_result = convolution(x,kernels,padding,conv_stride,layer)

    relu_result = relu(conv_result, layer=layer)

    MP_result = max_pooling(relu_result, MP_stride, layer)

    return MP_result

def convolution_backward(delta, kernel, padding, conv_stride, MP_stride, layer):
    # max pooling backward
    input_shape = params[layer]["pre_MP_size"]
    batch_number, C, W, H = input_shape
    cols_shape = params[layer]["cols_size"]
    d_col = np.zeros(cols_shape)
    delta_flat = delta.transpose(2, 3, 0, 1).ravel()
    MP_index = params[layer]["MP_index"]
    d_col[MP_index, range(MP_index.shape[0])] = delta_flat
    dX = col2im(d_col, (batch_number * C, 1, H, W), MP_stride, padding=0, stride=MP_stride).reshape(input_shape)

    # relu backward
    pre_activ = params[layer]["pre_activ"]
    drelu = relu_deriv(pre_activ)
    drelu = drelu*dX

    # convolution backward
    cols = params[layer]["cols"]
    drelu_C = drelu.shape[1]
    drelu_reshaped = drelu.transpose(1, 2, 3, 0).reshape(drelu_C, -1)
    dW = drelu_reshaped @ cols.T
    dW = dW.reshape(kernel.shape)
    kernel_reshape = kernel.reshape(kernel.shape[0], -1)
    dX_col = kernel_reshape.T @ drelu_reshaped
    dX = col2im(dX_col, params[layer]["x_shape"], kernel.shape[2], padding=padding, stride=conv_stride)

    return dW, dX

def MLP_forward(X, W, b, layer, activation=Sigmoid()):
    pre_act = X @ W + b
    post_act = activation.forward(pre_act)
    params[layer] = (X, pre_act, post_act)
    return post_act

def get_random_batches(x,y,batch_size):
    batches = []
    N = x.shape[0]
    rand_index = np.random.permutation(N)
    num_batches = N//batch_size
    for i in range(num_batches):
        index = rand_index[i*batch_size:(i+1)*batch_size]
        x_batch = x[index,:,:,:]
        y_batch = y[index,:]
        batches.append((x_batch,y_batch))
    return batches

def compute_loss_and_acc(y, probs):
    probs = probs.copy()
    log = lambda p: np.log(p)
    log_probs = log(probs)
    cross = y*log_probs
    loss = -np.sum(cross)

    n = y.shape[0]
    index = 0
    y_label = np.argmax(y, axis=1)
    probs_label = np.argmax(probs, axis=1)

    for i in range(n):
        if y_label[i] == probs_label[i]:
            index += 1
    acc = index/n
    return loss, acc


def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def linear_deriv(post_act):
    return np.ones_like(post_act)

def MLP_backwards(delta, W, b, layer, activation_deriv=sigmoid_deriv):
    X, pre_act, post_act = params[layer]
    delta_pre = delta * activation_deriv(post_act)
    grad_W = X.transpose() @ delta_pre
    grad_b = np.sum(delta_pre, axis=0, keepdims=True)
    grad_X = delta_pre @ W.transpose()
    return grad_X, grad_W, grad_b

def get_val_loss(val_X, val_y, val_loss, W1, b1, kernel1, val_size, val_acc):
    # forward
    conv1 = convolution_forward(val_X, kernel1, padding=padding, conv_stride=1, MP_stride=2, layer=1)
    flattened = conv1.reshape(conv1.shape[0], -1)
    p = MLP_forward(flattened, W1, b1, layer=2, activation=Softmax())

    # calculate loss
    loss, acc = compute_loss_and_acc(val_y, p)
    val_loss.append(loss/val_size)
    val_acc.append(acc)

def test_loss_and_accuracy(test_X, test_y, W1, b1, W2, b2, test_size):
    h1 = MLP_forward(test_X, W1, b1, layer=1, activation=Sigmoid())
    p = MLP_forward(h1, W2, b2, layer=2, activation=Softmax())

    loss, acc = compute_loss_and_acc(test_y, p)
    return loss/test_size, acc

def random_kernel_init(input_channel, output_channel, size):
    d0 = output_channel
    d1 = input_channel * size * size
    W = np.random.uniform(-np.sqrt(6) / np.sqrt(d0 + d1),
                          np.sqrt(6) / np.sqrt(d0 + d1), d0 * d1).reshape(output_channel, input_channel, size, size)
    return W




input_size = 256
hidden_size = 10
batch_size = 100
max_iters = 30
stride = 1
padding = 2
learning_rate = 0.01
params = {1:{},2:{}}
Momentum = -0.1
epsilon = 1e-9
output_channel1 = 1

'''
    Get the train/val/test dataset
'''
with open("../data/sampledCIFAR10", "rb") as f :
    data = pickle.load(f)
    train, val, test = data["train"], data["val"], data["test"]
    n_train_samples = train["data"].shape[0]
    n_val_samples = val["data"].shape[0]
    n_test_samples = test["data"].shape[0]

    train_X = train["data"].reshape(n_train_samples, 3, 32, 32)
    val_X = val["data"].reshape(n_val_samples, 3, 32, 32)
    test_X = test["data"].reshape(n_test_samples, 3, 32, 32)
    train_size = train_X.shape[0]
    val_size = val_X.shape[0]
    test_size = test_X.shape[0]

    train_y = train["labels"].astype(int)
    train_Y = np.zeros((train_y.shape[0],10))
    train_Y[np.arange(train_y.shape[0]),train_y] = 1

    val_y = val["labels"].astype(int)
    val_Y = np.zeros((val_y.shape[0], 10))
    val_Y[np.arange(val_y.shape[0]), val_y] = 1

    test_y = test["labels"].astype(int)
    test_Y = np.zeros((test_y.shape[0], 10))
    test_Y[np.arange(test_y.shape[0]), test_y] = 1

    print("successfully load data")


'''
    Initialize the weight
'''


kernel1 = random_kernel_init(3, output_channel1, 5)
W1 = random_normal_weight_init(input_size, hidden_size)
b1 = zeros_bias_init(hidden_size)
print("Weight initialized")

'''
    Get batches
'''
batches = get_random_batches(train_X,train_Y,batch_size)
print("Successfully splited the training data")


'''
    Train the network
'''

train_loss = []
val_loss = []
train_acc = []
val_acc = []
for itr in range(max_iters):
    total_loss = 0
    avg_acc = 0
    i = 0
    for xb, yb in batches:
        print(i)
        i += 1
        # forward
        conv1 = convolution_forward(xb, kernel1, padding=padding, conv_stride=1, MP_stride=2, layer=1)

        flattened = conv1.reshape(conv1.shape[0],-1)

        p = MLP_forward(flattened, W1, b1, layer=2, activation=Softmax())


        # calculate loss
        loss, acc = compute_loss_and_acc(yb, p)
        # print(loss)

        total_loss += loss
        avg_acc += acc

        # backward
        delta1 = p - yb
        delta2, grad_W1, grad_b1 = MLP_backwards(delta1, W1, b1, layer=2, activation_deriv=linear_deriv)
        delta2 = delta2.reshape(100,1,16,16)
        dW,_ = convolution_backward(delta2, kernel1, padding=padding, conv_stride=1, MP_stride=2, layer=1)
        # print("dW", dW)
        # Update the weight
        '''
            Without Momentum
        '''
        W1 -= learning_rate * grad_W1
        b1 -= learning_rate * grad_b1
        kernel1 -= learning_rate * dW


    if itr % 2 == 0:

        avg_acc /= len(batches)
        train_acc.append(avg_acc)
        print("Training epoch:", itr, "training accuracy:", avg_acc)
        train_loss.append(total_loss/train_size)
        get_val_loss(val_X, val_Y, val_loss, W1, b1, kernel1, val_size, val_acc)



'''
    Show the error rate
'''
x_axis = np.arange(1,len(train_loss)+1)
import matplotlib.pyplot as plt
plt.plot(x_axis*2, train_loss, 'r', label='Training Loss')
plt.legend()
plt.plot(x_axis*2, val_loss, 'g', label='Validation Loss')
plt.legend()
plt.xlabel('Training epoch')
plt.ylabel('Loss')
plt.show()
plt.close()

'''
    Show the accuracy rate
'''
x_axis = np.arange(1,len(train_acc)+1)
import matplotlib.pyplot as plt
plt.plot(x_axis*2, train_acc, 'r', label='Training Accuracy')
plt.legend()
plt.plot(x_axis*2, val_acc, 'g', label='Validation Accuracy')
plt.legend()
plt.xlabel('Training epoch')
plt.ylabel('Accuracy')
plt.grid()
plt.show()


