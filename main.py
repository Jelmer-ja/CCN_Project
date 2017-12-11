import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
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
    noise_size = 128
    train_data, test_data = get_mnist(n_train=1000,n_test=100,with_label=False,classes=[0])
    batch_size = 32
    gen = Generator()
    dis = Discriminator()
    iterator = RandomIterator(train_data, batch_size=batch_size)
    g_optimizer = optimizers.SGD()
    g_optimizer.setup(gen)
    d_optimizer = optimizers.SGD()
    d_optimizer.setup(dis)
    loss = run_network(epoch,batch_size,gen,dis,iterator,g_optimizer,d_optimizer,noise_size)
    showImages(gen,batch_size,noise_size)
    plot_loss(loss,epoch)

def run_network(epoch,batch_size,gen,dis,iterator,g_optimizer,d_optimizer,noise_size):
    losses = [[],[]]
    for i in range(0,epoch):
        #for j in range (0,batch_size) THEY USED K=1 IN THE PAPER SO SO DO WE
        k = 0

        for batch in iterator:
            k += 1
            dis.cleargrads(); gen.cleargrads()
            noise = randomsample(noise_size, batch_size)
            g_sample = gen(noise)
            disc_gen = dis(g_sample)
            disc_data = dis(np.reshape(batch,(batch_size,1,28,28),order='F'))
            softmax1 = F.sigmoid_cross_entropy(disc_gen,np.zeros((batch_size,1)).astype('int32'))
            softmax2 = F.sigmoid_cross_entropy(disc_data,np.ones((batch_size,1)).astype('int32'))
            loss = softmax1 + softmax2
            loss.backward()
            d_optimizer.update()
            losses[0].append(loss.data)
            if(k >= 1):
                break

        noise = randomsample(noise_size, batch_size)
        gn = gen(noise)
        loss = F.sigmoid_cross_entropy(dis(gn),np.ones((batch_size,1)).astype('int32'))
        loss.backward()
        g_optimizer.update()
        losses[1].append(loss.data)
    return losses

def randomsample(size, batch_size):
    return np.random.uniform(-1, 1, (batch_size, 100, 1, 1)).astype(np.float32)

def plot_loss(loss,epoch):
    plt.plot(np.array(range(1, epoch + 1)), np.array(loss[0]), label='Discriminator Loss')
    plt.plot(np.array(range(1, epoch + 1)), np.array(loss[1]), label='Generator Loss')
    plt.legend()
    plt.show()

def showImages(gen,batch_size,noise_size):
    batch_size = 1
    noise = randomsample(noise_size, batch_size)
    images = gen(noise)
    for i in images:
        plt.imshow(np.reshape(i.data[:,], (19, 19), order='F'))
        plt.show()

if(__name__ == "__main__"):
    main()