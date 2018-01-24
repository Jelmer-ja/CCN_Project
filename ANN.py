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
    def __init__(self):
        super(Generator, self).__init__()
        self.mnist_dim = 28
        self.n_units = 10
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(self.n_units * self.mnist_dim ** 2)  # n_in -> n_units        INPUT LAYER
            self.l2 = L.BatchNormalization(self.n_units * self.mnist_dim ** 2)  # n_units -> n_out       BATCH NORMALIZATION
            self.l3 = L.Deconvolution2D(in_channels=self.n_units, out_channels=1, ksize=3, stride=1, pad=1,
                                        outsize=(28, 28))  #  D              #  DECONVOLUTION

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = F.relu(self.l2(h1))
        h = F.reshape(h2, [-1, self.n_units, self.mnist_dim, self.mnist_dim])
        y = F.sigmoid(self.l3(h))
        return y

class LilDiscriminator(Discriminator):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_units = 10
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l2 = L.Convolution2D(in_channels=None, out_channels=self.n_units,ksize=3,stride=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3 = L.Linear(None, 1)    # n_units -> n_out

    def __call__(self, x):
        h2 = F.relu(self.l2(x))
        y = F.squeeze(self.l3(h2))
        return y

class LilColorGenerator(Generator):
    def __init__(self):
        super(Generator, self).__init__()
        self.mnist_dim = 28
        self.n_units = 10
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(self.n_units * 3 * self.mnist_dim ** 2)  # n_in -> n_units        INPUT LAYER
            self.l2 = L.BatchNormalization((self.n_units,3,self.mnist_dim,self.mnist_dim)) # n_units -> n_out       BATCH NORMALIZATION
            self.l3 = L.Deconvolution2D(in_channels=self.n_units, out_channels=1, ksize=3, stride=1, pad=1,
                                        outsize=(28, 28))  #  DECONVOLUTION

    def __call__(self, x):
        h1 = self.l1(x)
        hx = np.reshape(h1,[-1,self.n_units,28,28], order='F').astype('float32')
        h2 = F.relu(self.l2(hx))
        y = F.sigmoid(self.l3(h2))
        return y

class LilColorDiscriminator(Discriminator):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_units = 10
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l2 = L.Convolution2D(in_channels=None, out_channels=self.n_units,ksize=3,stride=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3 = L.Linear(None, 1)    # n_units -> n_out

    def __call__(self, x):
        h2 = F.relu(self.l2(x))
        y = F.squeeze(self.l3(h2))
        return y

class TestGenerator(Generator):
    def __init__(self):
        super(Generator, self).__init__()
        self.mnist_dim = 28
        self.n_units = 10
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(self.n_units * 4 ** 2)  # n_in -> n_units        INPUT LAYER
            self.l2 = L.BatchNormalization(self.n_units * 4 ** 2)    # n_units -> n_out       BATCH NORMALIZATION
            self.l3 = L.Deconvolution2D(in_channels=self.n_units, out_channels=10, ksize=2, stride=2,pad=1, outsize=(7,7))
            self.l4 = L.BatchNormalization(self.n_units * (self.mnist_dim/4) ** 2)  # n_units -> n_out       BATCH NORMALIZATION
            self.l5 = L.Deconvolution2D(in_channels=self.n_units, out_channels=10, ksize=4, stride=2, pad=1, outsize=(14,14))
            self.l6 = L.BatchNormalization(self.n_units * (self.mnist_dim / 2) ** 2)
            self.l7 = L.Deconvolution2D(in_channels=self.n_units, out_channels=1, ksize=3, stride=2,pad=1,outsize=(28,28))
            #  DECONVOLUTION

    def __call__(self, x):
        h1 = self.l1(x)
        h2 = F.relu(self.l2(h1))
        hx = F.reshape(h2, [-1,self.n_units,4,4])
        hx2 = self.l3(hx)
        hx3 = F.reshape(hx2, [32,490])
        h3 = F.relu(self.l4(hx3))
        h4 = F.reshape(h3, [-1,self.n_units,7,7])
        h5 = self.l5(h4)
        h5x = F.reshape(h5,[32,1960])
        h6 = F.relu(h5x)
        h6x = F.reshape(h6,[-1,self.n_units,14,14])
        y = F.sigmoid(self.l7(h6x))
        return y

class TestDiscriminator(Discriminator):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_units = 10
        self.mnist_dim = 28
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l2 = L.Convolution2D(in_channels=None, out_channels=self.n_units,ksize=3,stride=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3 = L.BatchNormalization(self.n_units * 26 ** 2)
            self.l4 = L.Convolution2D(in_channels=None, out_channels=self.n_units,ksize=4,stride=2)
            self.l5 = L.BatchNormalization(self.n_units * 12 ** 2)
            self.lx = L.Convolution2D(in_channels=None, out_channels=self.n_units,ksize=2,stride=2)
            self.ly = L.BatchNormalization(self.n_units * 6 ** 2)
            self.l6 = L.Linear(None, 1)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.reshape(self.l2(x),[32,6760])
        h2 = F.relu(self.l3(h1))
        h3 = F.reshape(h2,[-1,self.n_units,26,26])
        h4 = self.l4(h3)
        h5 = F.reshape(h4, [32,1440])
        h6 = F.relu(self.l5(h5))
        y = F.squeeze(self.l6(h6))
        return y