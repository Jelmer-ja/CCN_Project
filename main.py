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
    epoch = 3
    train_data, test_data = get_mnist(n_train=1000,n_test=100,with_label=False,classes=[0])
    batch_size = 32
    gen = Generator()
    dis = Discriminator()
    iterator = iterators.SerialIterator(train_data, batch_size=batch_size)
    g_optimizer = optimizers.Adam()
    g_optimizer.setup(gen)
    d_optimizer = optimizers.Adam()
    d_optimizer.setup(dis)
    loss = run_network(epoch,batch_size,gen,dis,iterator,g_optimizer,d_optimizer)
    showImages(gen,batch_size)
    plot_loss(loss,epoch,batch_size)

def run_network(epoch,batch_size,gen,dis,iterator,g_optimizer,d_optimizer):
    losses = [[],[]]
    for i in range(0, epoch):
        # for j in range (0,batch_size) THEY USED K=1 IN THE PAPER SO SO DO WE
        print(i)

        #always start epoch loss at zero
        gloss_epoch = np.float32(0)
        dloss_epoch = np.float32(0)

        for j in range(0, 1000, batch_size):
            batch = iterator.next()
            noise = randomsample(batch_size)
            g_sample = gen(noise)

            disc_gen = dis(g_sample) # others call this y_fake
            disc_data = dis(np.reshape(batch, (batch_size, 1, 28, 28), order='F')) # others call this y_real

            L1 = F.sigmoid_cross_entropy(disc_gen, np.zeros((batch_size, 1)).astype('int32')) # compare y_fake to zeros
            L2 = F.sigmoid_cross_entropy(disc_data, np.ones((batch_size, 1)).astype('int32')) # compare y_real to ones
            #L1 = F.sum(F.softplus(disc_data)) / batch_size
            #L2 = F.sum(F.softplus(-disc_gen)) / batch_size
            dloss = L1 + L2
            dloss/= 2
            #losses[0].append(dloss.data)

            #noise = randomsample(batch_size)
            #gn = gen(noise)
            gloss = F.sigmoid_cross_entropy(disc_gen, np.ones((batch_size, 1)).astype('int32')) # result of discriminator on generated image should be close to one
            #gloss = F.sum(F.softplus(gn)) / batch_size
            #losses[1].append(gloss.data)

            dis.cleargrads()
            dloss.backward()
            d_optimizer.update()

            gen.cleargrads()
            gloss.backward()
            g_optimizer.update()

            gloss_epoch += gloss.data
            dloss_epoch += dloss.data

        # every epoch, append average loss per item
        losses[0].append(dloss_epoch / 1000)
        losses[1].append(gloss_epoch / 1000)


        #     generator_epoch_loss += generator_loss.data
        #     discriminator_epoch_loss += discriminator_loss.data
        #
        # generator_avg_loss = generator_epoch_loss / train_size
        # discriminator_avg_loss = discriminator_epoch_loss / train_size

    return losses

def randomsample(batch_size):
    return np.random.uniform(-1, 1, (batch_size, 100, 1, 1)).astype(np.float32)

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