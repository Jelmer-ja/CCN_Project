import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers, training
from chainer import Link, Chain, ChainList
from multiprocessing.dummy import Pool
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import math
import random
from scipy import misc
import glob
from ANN import *
from utils import *

def main():
    #SET PARAMETERS
    epoch = 100
    data_size = 6000
    feature_matching = True #Color images = 3
    #train_data, test_data = get_mnist(n_train=1000,n_test=100,with_label=False,classes=[0])
    cats = getCats()
    #cats = grayscale(cats)
    data = np.asarray(cats[0:data_size])
    train_data = chainer.datasets.TupleDataset(data,np.asarray(range(0,data_size)))
    #showTrain(train_data)
    batch_size = 32

    #CREATE CLASSES
    gen = TestGenerator()
    dis = TestDiscriminator(False)
    iterator = RandomIterator(train_data,batch_size) #iterators.SerialIterator(train_data, batch_size=batch_size)
    g_optimizer = optimizers.Adam(alpha=0.0002,beta1=0.5)
    g_optimizer.setup(gen)
    d_optimizer = optimizers.Adam(alpha=0.0002,beta1=0.5)
    d_optimizer.setup(dis)

    #RUN NETWORKS
    loss = run_network(epoch, batch_size, gen, dis, iterator, g_optimizer, d_optimizer, feature_matching)
    showImages(gen,batch_size)
    plot_loss(loss,epoch,batch_size)

def run_network(epoch,batch_size,gen,dis,iterator,g_optimizer,d_optimizer, feature_matching):
    losses = [[],[]]
    for i in range(0,epoch):
        #for j in range (0,batch_size) THEY USED K=1 IN THE PAPER SO SO DO WE
        dloss_all = 0
        gloss_all = 0
        with chainer.using_config('train', True):
            for batch in iterator:
                #clear gradients
                dis.cleargrads()
                gen.cleargrads()
                #Run the discriminator on fake and real data
                input = np.reshape(batch[0], (batch_size, 3, 28, 28), order='F')
                input = input.astype('float32')
                noise = randomsample(batch_size)
                if(feature_matching):
                    disc_act,disc_data = dis.activation_call(input)
                    gen_act,g_sample = gen.activation_call(noise)
                else:
                    disc_data = dis(input)
                    g_sample = gen(noise)
                disc_gen = dis(g_sample)

                #Calculate loss and update discriminator
                softmax1 = F.sigmoid_cross_entropy(disc_gen,np.zeros((batch_size,)).astype('int32'))
                softmax2 = F.sigmoid_cross_entropy(disc_data,np.ones((batch_size,)).astype('int32'))
                dloss = softmax1 + softmax2
                dloss.backward()
                d_optimizer.update()


                #Calculate loss and update generator
                gloss = F.sigmoid_cross_entropy(disc_gen, np.ones((batch_size,)).astype('int32'))
                if(feature_matching):
                    disc_act[-1].unchain_backward()
                    gloss += F.mean_squared_error(disc_act[-1],gen_act[-1])
                gloss.backward()
                g_optimizer.update()

                dloss_all +=gloss.data
                gloss_all +=dloss.data
            losses[0].append(dloss_all)
            losses[1].append(gloss_all)
            print('Epoch ' + str(i) + ' finished')
    return losses

def randomsample(batch_size):
    return np.random.uniform(-1, 1, (batch_size, 100, 1, 1)).astype(np.float32)

def plot_loss(loss,epoch,batch_size):
    plt.plot(np.array(range(0, epoch)), np.array(loss[0]), label='Discriminator Loss')
    plt.plot(np.array(range(0, epoch)), np.array(loss[1]), label='Generator Loss')
    plt.legend()
    plt.show()

def showImages(gen,batch_size):
    f,axes = plt.subplots(2,5)
    noise = randomsample(batch_size)
    with chainer.using_config('train', False):
        images = gen(noise)
    for i in range(0,10):
        if(i % 2 == 0):
            x = 0
        else:
            x = 1
        y = int(round(i/2,0))
        axes[x][y].imshow(np.reshape(images[i].data[:,], (28, 28, 3), order='F'))
    plt.show()

def showTrain(train):
    f,axes = plt.subplots(2,5)
    for i in range(0,10):
        if (i % 2 == 0):
            x = 0
        else:
            x = 1
        y = int(round(i / 2, 0))
        image = train[i]
        axes[x][y].imshow(image[0])
    plt.show()

def getCats():
    images = []
    for image_path in glob.glob("cropped_catfaces/*.jpg"):
        image = misc.imread(image_path)
        images.append(image)
        #print image.shape
    return images

def grayscale(images):
    output = []
    for i in images:
        r, g, b = i[:, :, 0], i[:, :, 1], i[:, :, 2]
        gray = np.float32(0.2989) * r + np.float32(0.5870) * g + np.float32(0.1140) * b
        gray2 = [float(item) for sublist in gray for item in sublist]
        output.append(gray2)
    return output

if(__name__ == "__main__"):
    main()