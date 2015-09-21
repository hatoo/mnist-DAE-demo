#!/usr/bin/env python
"""Chainer example: train a multi-layer perceptron on MNIST

This is a minimal example to write a feed-forward net. It requires scikit-learn
to load MNIST dataset.

"""
import argparse

import numpy as np
import six

import chainer
from chainer import computational_graph as c
from chainer import cuda
import chainer.functions as F
from chainer import optimizers

from autoencoder import Autoencoder
from sigmoid_cross_entropy_float import sigmoid_cross_entropy_float

import data
import pickle

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()
if args.gpu >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if args.gpu >= 0 else np

batchsize = 100
n_epoch = 25
n_units = 500

# Prepare dataset
print('load MNIST dataset')
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N = 60000
x_train, x_test = np.split(mnist['data'],   [N])
y_train, y_test = np.split(mnist['target'], [N])
N_test = y_test.size

# Prepare multi-layer perceptron model
model = chainer.FunctionSet(ae1=Autoencoder(784, n_units))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()


def forward(x_data, t_data, train=True):
    x = chainer.Variable(x_data)
    t = chainer.Variable(t_data)
    y = model.ae1(x)
    loss = sigmoid_cross_entropy_float(y, t)

    return loss


def ZeroMask(data, ratio=0.90):
    return data * (xp.random.random_sample(data.shape) > ratio)

def SP(data, ratio=0.50):
    n = int(data.size*ratio)
    perm = np.random.permutation(data.size)[:n]
    ret = cuda.to_cpu(data).flatten()
    ret[perm] = (np.random.rand(n) > 0.5)
    ret = ret.reshape(data.shape)
    return xp.asarray(ret)

def GN(data, var=0.75):
    noise = xp.random.normal(scale=var, size=data.shape, dtype=xp.float32)
    return data+noise

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)

# Learning loop
for epoch in six.moves.range(1, n_epoch + 1):
    print('epoch', epoch)

    # training
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N, batchsize):
        x_batch = xp.asarray(x_train[perm[i:i + batchsize]])
        noised = GN(x_batch)

        optimizer.zero_grads()
        loss = forward(noised, x_batch)
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * batchsize

    print('train mean loss={}'.format(
        sum_loss / N))

    # evaluation
    sum_accuracy = 0
    sum_loss = 0
    for i in six.moves.range(0, N_test, batchsize):
        x_batch = xp.asarray(x_test[i:i + batchsize])
        noised = GN(x_batch)
        loss = forward(noised, x_batch, train=False)
        sum_loss += float(loss.data) * batchsize

    print('test  mean loss={}'.format(
        sum_loss / N_test))

print 'serialize ...'
ae1 = model.ae1

model_data = {'w': cuda.to_cpu(ae1.W), 'b1':cuda.to_cpu(ae1.b1), 'b2':cuda.to_cpu(ae1.b2)}
f = open("ae1.pickle", "w")
pickle.dump(model_data, f)
f.close()
