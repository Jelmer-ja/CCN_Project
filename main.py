import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, training
from chainer import Link, Chain, ChainList
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import math
import random
from ANN import *
from utils import *

def main():
    epoch = 100
    train_data, test_data = get_mnist(n_train=1000,n_test=100,with_label=False,classes=[0])
    #train = datasets.get_mnist(withlabel=False, ndim=2)

    batch_size = 32
    gen = Generator(10,(32,32))
    dis = Discriminator((32,32))
    iterator = iterators.SerialIterator(train_data, batch_size=batch_size)
    g_optimizer = optimizers.MomentumSGD(0.01)
    g_optimizer.setup(gen)
    d_optimizer = optimizers.MomentumSGD(0.01)
    d_optimizer.setup(dis)
    loss = run_network(epoch,batch_size,gen,dis,iterator,g_optimizer,d_optimizer)
    showImages(gen,batch_size)
    plot_loss(loss,epoch,batch_size)

def run_network(epoch,batch_size,gen,dis,iterator,g_optimizer,d_optimizer):
    losses = [[],[]]
    for i in range(0, epoch):
        # for j in range (0,batch_size) THEY USED K=1 IN THE PAPER SO SO DO WE
        print(i)

        batch = iterator.next()
        dis.cleargrads()
        gen.cleargrads()
        noise = randomsample(batch_size)
        g_sample = gen(noise)
        disc_gen = dis(g_sample)
        disc_data = dis(np.reshape(batch, (batch_size, 1, 28, 28), order='F'))
        L1 = F.sigmoid_cross_entropy(disc_gen, np.zeros((batch_size, 1)).astype('int32'))
        L2 = F.sigmoid_cross_entropy(disc_data, np.ones((batch_size, 1)).astype('int32'))
        #L1 = F.sum(F.softplus(disc_data)) / batch_size
        #L2 = F.sum(F.softplus(-disc_gen)) / batch_size
        loss = L1 + L2
        loss.backward()
        d_optimizer.update()
        losses[0].append(loss.data)

        noise = randomsample(batch_size)
        gn = gen(noise)
        gloss = F.sigmoid_cross_entropy(dis(gn), np.ones((batch_size, 1)).astype('int32'))
        #gloss = F.sum(F.softplus(gn)) / batch_size
        gloss.backward()
        g_optimizer.update()
        losses[1].append(gloss.data)
    return losses

def randomsample(batch_size):
    return np.random.uniform(-1, 1, (batch_size, 10, 1, 1)).astype(np.float32)

def plot_loss(loss,epoch,batch_size):
    plt.plot(np.array(range(0, epoch)), np.array(loss[0]), label='Discriminator Loss')
    plt.plot(np.array(range(0, epoch)), np.array(loss[1]), label='Generator Loss')
    plt.legend()
    plt.show()

def showImages(gen,batch_size):
    batch_size = 1
    noise = randomsample(batch_size)
    images = gen(noise)
    for i in images:
        plt.imshow(np.reshape(i.data[:,], (28, 28), order='F'))
        plt.show()

if(__name__ == "__main__"):
    main()