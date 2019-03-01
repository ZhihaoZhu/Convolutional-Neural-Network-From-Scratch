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

def convolution_forward(x,kernels,padding,conv_stride,params,layer):
    kernel_size = kernels.shape[2]
    cols,out_size = im2col(x, kernel_size, padding=padding, stride=conv_stride)
    kernel_cols = kernels.reshape(kernels.shape[0],-1)
    output = kernel_cols @ cols
    out = output.reshape(kernels.shape[0], out_size, out_size, x.shape[0])
    out = out.transpose(3, 0, 1, 2)
    param_name = "conv_output" + layer
    params[param_name] = out
    return out

def relu(x):
    sig = lambda p: p if p > 0 else 0
    sigfunc = np.vectorize(sig)
    res = sigfunc(x)
    return res

def max_pooling_forward(x):
    batch_number, C, _, _ = x.shape
    cols, out_size = im2col(x, 2, padding=0, stride=2)
    cols_w,cols_h = cols.shape
    cols.reshape(C, cols_w/C, -1)  #注意im2col的定义，这里的col已经被reshape过了
    MP_index = np.max(cols, axis=1)


def forward_pass(X, W, b, layer, activation=Sigmoid()):

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
        x_batch = x[index,:]
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

def backwards(delta, W, b, layer, activation_deriv=sigmoid_deriv):
    X, pre_act, post_act = params[layer]
    delta_pre = delta * activation_deriv(post_act)
    print("yes")
    print(X.shape)
    print(delta.shape)
    print(delta_pre.shape)

    grad_W = X.transpose() @ delta_pre
    grad_b = np.sum(delta_pre, axis=0, keepdims=True)
    grad_X = delta_pre @ W.transpose()
    return grad_X, grad_W, grad_b

def get_val_loss(val_X, val_y, val_loss, W1, b1, W2, b2, val_size, val_acc):
    # forward
    h = forward_pass(val_X, W1, b1, layer=1, activation=Sigmoid())
    p = forward_pass(h, W2, b2, layer=2, activation=Softmax())

    # calculate loss
    loss, acc = compute_loss_and_acc(val_y, p)
    val_loss.append(loss/val_size)
    val_acc.append(acc)

def test_loss_and_accuracy(test_X, test_y, W1, b1, W2, b2, test_size):
    h1 = forward_pass(test_X, W1, b1, layer=1, activation=Sigmoid())
    p = forward_pass(h1, W2, b2, layer=2, activation=Softmax())

    loss, acc = compute_loss_and_acc(test_y, p)
    return loss/test_size, acc

def random_kernel_init(input_channel, output_channel, size):
    return np.random.normal(loc = 0, scale = 1.0, size = (output_channel, input_channel, size*size))


input_size = 1568
hidden_size = 100
output_size = 19
batch_size = 100
max_iters = 30
stride = 1
padding = 2
learning_rate = 0.01
params = {1:{},2:{}}
Momentum = -0.1
epsilon = 1e-9

'''
    Get the train/val/test dataset
'''
train = np.loadtxt('../data/data/train.txt',delimiter= ',',unpack= False)
train_X = train[:,:-1]
train_Y = train[:,-1].astype(int)
train_y = np.zeros((train_Y.shape[0],19))
train_y[np.arange(train_Y.shape[0]),train_Y] = 1
print("Successfully loaded training data")

val = np.loadtxt('../data/data/val.txt',delimiter= ',',unpack= False)
val_X = val[:,:-1]
val_Y = val[:,-1].astype(int)
val_y = np.zeros((val_Y.shape[0],19))
val_y[np.arange(val_Y.shape[0]),val_Y] = 1
print("Successfully loaded val data")

test = np.loadtxt('../data/data/test.txt',delimiter= ',',unpack= False)
test_X = test[:,:-1]
test_Y = test[:,-1].astype(int)
test_y = np.zeros((test_Y.shape[0],19))
test_y[np.arange(test_Y.shape[0]),test_Y] = 1
print("Successfully loaded test data")

train_size = train_X.shape[0]
val_size = val_X.shape[0]
test_size = test_X.shape[0]
print("train_size:", train_size)
print("val_size:", val_size)
print("test_size:", test_size)



'''
    Initialize the weight
'''


W1 = random_normal_weight_init(input_size, hidden_size)
b1 = zeros_bias_init(hidden_size)
W2 = random_normal_weight_init(hidden_size, output_size)
b2 = zeros_bias_init(output_size)

M_W1, M_W2, M_b1, M_b2 = W1, W2, b1, b2
print("Weight initialized")


'''
    Get batches
'''
batches = get_random_batches(train_X,train_y,batch_size)
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
    for xb, yb in batches:
        # forward
        h = forward_pass(xb, W1, b1, layer=1, activation=Sigmoid())
        p = forward_pass(h, W2, b2, layer=2, activation=Softmax())

        # calculate loss
        loss, acc = compute_loss_and_acc(yb, p)
        # print(loss)

        total_loss += loss
        avg_acc += acc

        # backward
        delta1 = p - yb
        delta2, grad_W2, grad_b2 = backwards(delta1, W2, b2, layer=2, activation_deriv=linear_deriv)
        _, grad_W1, grad_b1 = backwards(delta2, W1, b1, layer=1, activation_deriv=sigmoid_deriv)

        # Update the weight
        '''
            Without Momentum
        '''
        # W1 -= learning_rate * grad_W1
        # W2 -= learning_rate * grad_W2
        # b1 -= learning_rate * grad_b1
        # b2 -= learning_rate * grad_b2

        '''
            With Momentum
        '''
        M_W1 = Momentum * M_W1 - (1+Momentum) * learning_rate * grad_W1
        W1 += M_W1
        M_W2 = Momentum * M_W2 - (1+Momentum) * learning_rate * grad_W2
        W2 += M_W2
        M_b1 = Momentum * M_b1 - (1+Momentum) * learning_rate * grad_b1
        b1 += M_b1
        M_b2 = Momentum * M_b2 - (1+Momentum) * learning_rate * grad_b2
        b2 += M_b2

    if itr % 2 == 0:
        # np.save("./saved_model/W1_%d.npy" % itr, W1)
        # np.save("./saved_model/W2_%d.npy" % itr, W2)
        # np.save("./saved_model/b1_%d.npy" % itr, b1)
        # np.save("./saved_model/b2_%d.npy" % itr, b2)
        # print("model saved for epoch:", itr)

        avg_acc /= len(batches)
        train_acc.append(avg_acc)
        print("Training epoch:", itr, "training accuracy:", avg_acc)
        train_loss.append(total_loss/train_size)
        get_val_loss(val_X, val_y, val_loss, W1, b1, W2, b2, val_size, val_acc)



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

'''
    Show the error and accuracy for test
'''

test_size = test_X.shape[0]
test_loss, test_accuracy = test_loss_and_accuracy(test_X, test_y, W1, b1, W2, b2, test_size)
print(test_loss, test_accuracy)