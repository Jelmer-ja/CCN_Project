import chainer.functions as F
import numpy as np
import chainer.links as L
from scipy import misc
import glob
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
            self.l1 = L.Linear(124 * 4 ** 2)  # n_in -> n_units        INPUT LAYER
            self.l2 = L.BatchNormalization((124,4,4) )   # n_units -> n_out       BATCH NORMALIZATION
            self.l3 = L.Deconvolution2D(in_channels=124, out_channels=64, ksize=2, stride=2,pad=1, outsize=(7,7))
            self.l4 = L.BatchNormalization((64,7,7))  # n_units -> n_out       BATCH NORMALIZATION
            self.l5 = L.Deconvolution2D(in_channels=64, out_channels=32, ksize=4, stride=2, pad=1, outsize=(14,14))
            self.l6 = L.BatchNormalization((32,14,14))
            self.l7 = L.Deconvolution2D(in_channels=32, out_channels=3, ksize=3, stride=2,pad=1,outsize=(28,28))
            #  DECONVOLUTION

    def __call__(self, x):
        h1 = F.reshape(self.l1(x),[32,124,4,4])
        h2 = F.relu(self.l2(h1))
        #hx = F.reshape(h2, [32,self.n_units,4,4])
        hx2 = self.l3(h2)
        #hx3 = F.reshape(hx2, [32,self.n_units * 7 ** 2])
        h3 = F.relu(self.l4(hx2))
        #h4 = F.reshape(h3, [-1,self.n_units,7,7])
        h5 = self.l5(h3)
        #h5x = F.reshape(h5,[32,self.n_units * 14 ** 2])
        h6 = F.relu(h5)
        #h6x = F.reshape(h6,[-1,self.n_units,14,14])
        y = F.sigmoid(self.l7(h6))
        return y

class TestDiscriminator(Discriminator):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_units = 15
        self.mnist_dim = 28
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l2 = L.Convolution2D(in_channels=None, out_channels=64,ksize=3,stride=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3 = L.BatchNormalization((64,26,26))
            self.l4 = L.Convolution2D(in_channels=64, out_channels=32,ksize=4,stride=2)
            self.l5 = L.BatchNormalization((32,12,12))
            self.l6 = L.Convolution2D(in_channels=32, out_channels=16,ksize=2,stride=2)
            self.l7 = L.BatchNormalization((16,6,6))
            self.l8 = L.Linear(576, 1)    # n_units -> n_out

    def __call__(self, x):
        #h1 = F.reshape(self.l2(x),[32,self.n_units * 3 * 26 ** 2])
        h1 = self.l2(x)
        h2 = F.relu(self.l3(h1))
        #h3 = F.reshape(h2,[-1,self.n_units,3,26,26])
        h4 = self.l4(h2)
        #h5 = F.reshape(h4, [32,self.n_units * 3 * 12 ** 2])
        h6 = F.relu(self.l5(h4))
        #h6x = F.reshape(h6,[-1,self.n_units,3,12,12])
        h7 = self.l6(h6)
        #h7x = F.reshape(h7,[32,self.n_units * 3 * 6 ** 2])
        h8 = F.relu(self.l7(h7))
        y = F.squeeze(self.l8(h8))
        return y


class HighResGenerator(Generator):
    def __init__(self):
        super(Generator, self).__init__()
        self.mnist_dim = 28
        self.n_units = 10
        with self.init_scope():
            self.l2 = L.BatchNormalization((self.n_units, 28, 28))  # n_units -> n_out       BATCH NORMALIZATION
            self.l3 = L.Deconvolution2D(in_channels=self.n_units, out_channels=self.n_units, ksize=4, stride=2, pad=1, outsize=(48, 48))
            self.l4 = L.BatchNormalization((self.n_units, 48, 48))  # n_units -> n_out       BATCH NORMALIZATION
            self.l5 = L.Deconvolution2D(in_channels=self.n_units, out_channels=self.n_units, ksize=4, stride=1, pad=0, outsize=(51, 51))
            self.l6 = L.BatchNormalization((self.n_units, 51, 51))
            self.l7 = L.Deconvolution2D(in_channels=self.n_units, out_channels=3, ksize=2, stride=2, pad=1, outsize=(100, 100))

    def __call__(self, x):
        h1 = F.relu(self.l2(x))
        h2 = F.relu(self.l4(self.l3(h1)))
        h3 = F.relu(self.l6(self.l5(h2)))
        y = F.sigmoid(self.l7(h3))
        return y

class HighResDiscriminator(Discriminator):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_units = 15
        self.mnist_dim = 28
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l2 = L.Convolution2D(in_channels=None, out_channels=64,ksize=3,stride=1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3 = L.BatchNormalization((64,26,26))
            self.l4 = L.Convolution2D(in_channels=64, out_channels=32,ksize=4,stride=2)
            self.l5 = L.BatchNormalization((32,12,12))
            self.l6 = L.Convolution2D(in_channels=32, out_channels=16,ksize=2,stride=2)
            self.l7 = L.BatchNormalization((16,6,6))
            self.l8 = L.Linear(576, 1)    # n_units -> n_out

    #def __call__(self, x):
    #TODO: Implement call and fix layer parameters


class Indices():
    def __init__(self):
        self.url = '/home/jelmer/Github/CCN_Project/catfaces/'

    def getIndices(self):
        indices = []
        for image_path in glob.glob("/home/jelmer/Github/CCN_Project/catfaces/*.jpg"):
            image = misc.imread(image_path)
            width, heigth, dummy = np.shape(image)
            tokens = image_path.split('/')
            if (width == 48 and heigth == 48):
                indices.append(tokens[-1])
        return indices