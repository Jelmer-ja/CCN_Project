import chainer.functions as F
import F.batch_normalization as bn
import numpy as np
import chainer.links as L
from chainer import Link, Chain, ChainList, report

class Generator(Chain):
    def __init__(self, n_units):
        super(Generator, self).__init__()
        #self.dim = 32 #IMAGE SIZE
        with self.init_scope():
            # # the size of the inputs to each layer will be inferred
            # TODO: pad = ???
            self.l1 = L.Linear(1 * 100 ** 2)  # n_in -> n_units        INPUT LAYER
            self.l2 = L.BatchNormalization(1024 * 4 ** 2)  # n_units -> n_out       BATCH NORMALIZATION
            self.l3 = L.Deconvolution2D(in_channels=1024, out_channels=512, ksize=1, stride=2, pad=1,outsize=(4,4))  # DECONVOLUTION
            self.l4 = L.BatchNormalization(512 * 8 ** 2)  # n_units -> n_out       BATCH NORMALIZATION
            self.l5 = L.Deconvolution2D(in_channels=512, out_channels=256, ksize=5, stride=2, pad=1,outsize=(8,8))  # DECONVOLUTION
            self.l6 = L.BatchNormalization(256 * 16 ** 2)  # n_units -> n_out       BATCH NORMALIZATION
            self.l7 = L.Deconvolution2D(in_channels=256, out_channels=128, ksize=5, stride=2, pad=1,outsize=(16,16))  # DECONVOLUTION
            self.l8 = L.BatchNormalization(128 * 32 ** 2)  # n_units -> n_out       BATCH NORMALIZATION
            self.l9 = L.Deconvolution2D(in_channels=128, out_channels=3, ksize=5, stride=2, pad=1,outsize=(32,32))  # DECONVOLUTION
            self.l10 = L.BatchNormalization(3 * 64 ** 2)  # n_units -> n_out       BATCH NORMALIZATION
            self.l11 = L.Deconvolution2D(in_channels=3, out_channels=1, ksize=5, stride=2, pad=1,outsize=(64,64))  # DECONVOLUTION

    def __call__(self, x):
        # Reshape image
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        h5 = F.relu(self.l5(h4))
        h6 = F.relu(self.l6(h5))
        h7 = F.relu(self.l7(h6))
        h8 = F.relu(self.l8(h7))
        h9 = F.relu(self.l9(h8))
        h10 = F.relu(self.l10(h9))
        y = F.tanh(self.l11(h10))
        return y

class Discriminator(Chain):
    def __init__(self, n_units):
        super(Discriminator, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            # TODO: How to split up input into 16 channels with no changes
            self.l1 = L.Linear(1, 32)  # n_in -> n_units            INPUT LAYER
            self.l2 = L.Convolution2D(32, 16)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l3 = L.Convolution2D(16, 8)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l4 = L.Convolution2D(8, 4)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l5 = L.Convolution2D(4, 2)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l6 = L.Convolution2D(2, 1)  # n_units -> n_units  CONVOLUTIONAL LAYER
            self.l7 = L.Linear(None, 1)    # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(bn(self.l2(h1)))
        h3 = F.relu(bn(self.l3(h2)))
        h4 = F.relu(bn(self.l4(h3)))
        h5 = F.relu(bn(self.l5(h4)))
        h6 = F.relu(bn(self.l6(h5)))
        y = F.relu(bn(self.l7(h6)))
        return y

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01
