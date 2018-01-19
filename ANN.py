import chainer.functions as F
import numpy as np
import chainer.links as L
from chainer import Link, Chain, ChainList, report

class Generator(Chain):
    def __init__(self):
        super(Generator, self).__init__()
        self.ch = 512 #Channels
        with self.init_scope():
            # # the size of the inputs to each layer will be inferred
            # TODO: pad = ???
            self.l1 = L.Linear(in_size=100, out_size=3*3*self.ch)
            self.l2 = L.BatchNormalization(self.ch)  # n_units -> n_out       BATCH NORMALIZATION
            self.l3 = L.Deconvolution2D(in_channels=self.ch, out_channels=self.ch / 2, ksize=2, stride=2, pad=1)  # DECONVOLUTION
            self.l4 = L.BatchNormalization(self.ch / 2)  # n_units -> n_out       BATCH NORMALIZATION
            self.l5 = L.Deconvolution2D(in_channels=self.ch / 2, out_channels=self.ch / 4, ksize=2, stride=2, pad=1)  # DECONVOLUTION
            self.l6 = L.BatchNormalization(self.ch / 4)  # n_units -> n_out       BATCH NORMALIZATION
            self.l7 = L.Deconvolution2D(in_channels=self.ch / 4, out_channels=self.ch / 8, ksize=2, stride=2, pad=1)  # DECONVOLUTION
            self.l8 = L.BatchNormalization(self.ch / 8)  # n_units -> n_out       BATCH NORMALIZATION
            self.l9 = L.Deconvolution2D(in_channels=self.ch / 8, out_channels=1, ksize=3, stride=3, pad=1)  # DECONVOLUTION

            #self.l10 = L.BatchNormalization(3 * 64 ** 2)  # n_units -> n_out       BATCH NORMALIZATION
            #self.l11 = L.Deconvolution2D(in_channels=3, out_channels=1, ksize=5, stride=2, pad=1,outsize=(64,64))  # DECONVOLUTION

    def __call__(self, x):
        # Reshape image
        h1 = F.relu(self.l1(x))
        h2 = F.reshape(h1, (len(x), self.ch, 3, 3))
        h22 = F.relu(self.l2(h2))
        h3 = F.relu(self.l4(self.l3(h22)))
        h5 = F.relu(self.l6(self.l5(h3)))
        h7 = F.relu(self.l8(self.l7(h5)))
        y = F.sigmoid(self.l9(h7))
        #h10 = F.relu(self.l10(h9))
        #y = F.tanh(self.l11(h10))
        return y

class Discriminator(Chain):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.ch = 512
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            # TODO: How to split up input into 16 channels with no changes
            #28 to 32 for cats/bedrooms
            self.l2 = L.Convolution2D(in_channels=1, out_channels=self.ch/8, ksize=3, stride=3, pad=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3 = L.Convolution2D(in_channels=self.ch / 8, out_channels=self.ch / 4, ksize=3, stride=3, pad=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3a = L.BatchNormalization(self.ch / 4)
            self.l4 = L.Convolution2D(in_channels=self.ch / 4, out_channels=self.ch / 2, ksize=3, stride=3, pad=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l4a = L.BatchNormalization(self.ch / 2)
            self.l5 = L.Convolution2D(in_channels=self.ch / 2, out_channels=self.ch, ksize=3, stride=3, pad=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l5a = L.BatchNormalization(self.ch)
            self.l6 = L.Linear(None, 1)    # n_units -> n_out

    def __call__(self, x):
        h = F.leaky_relu(self.l2(x))
        h = F.leaky_relu(self.l3a(self.l3(h)))
        h = F.leaky_relu(self.l4a(self.l4(h)))
        h = F.leaky_relu(self.l5a(self.l5(h)))
        y = F.sigmoid(self.l6(h))
        return y

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01

class LilGenerator(Generator):
    def __init__(self, n_units):
        super(Generator, self).__init__()
        self.mnist_dim = 28
        self.n_units = n_units
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(n_units * self.mnist_dim **2)  # n_in -> n_units        INPUT LAYER
            self.l_a = L.BatchNormalization(n_units * self.mnist_dim ** 2)    # n_units -> n_out       BATCH NORMALIZATION
            self.l_b = L.Deconvolution2D(in_channels=n_units, out_channels=2, ksize=2, stride=1,pad=1)                   #  DECONVOLUTION
            self.l2 = L.BatchNormalization(n_units * self.mnist_dim ** 2)    # n_units -> n_out       BATCH NORMALIZATION
            self.l3 = L.Deconvolution2D(in_channels=n_units, out_channels=1, ksize=2, stride=1,pad=1,outsize=(28,28))                   #  DECONVOLUTION

    def __call__(self, x):
        h1 = self.l1(x)
        ha = F.relu(self.l_a(h1))
        hb = F.relu(self.l_b(ha))
        h2 = F.relu(self.l2(ha))
        h = F.reshape(h2, [-1,self.n_units,self.mnist_dim,self.mnist_dim])
        y = F.sigmoid(self.l3(h))
        return y

class LilDiscriminator(Discriminator):
    def __init__(self, n_units):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l2 = L.Convolution2D(in_channels=None, out_channels=n_units,ksize=3,stride=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l_a = L.Convolution2D(in_channels=None, out_channels=n_units,ksize=3,stride=1)  # n_units -> n_units  CONVOLUTINAL LAYER
            self.l3 = L.Linear(None, 1)    # n_units -> n_out

    def __call__(self, x):
        h2 = F.relu(self.l2(x))
        y = F.squeeze(self.l3(h2))
        return y
