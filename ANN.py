import chainer.functions as F
import numpy as np
import chainer.links as L
from chainer import Link, Chain, ChainList, report

"""
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
        y = F.tanh(self.l9(h7))
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
"""
def lindim(dims, scale, n):
    d = map(lambda x: x // scale, dims)
    d = reduce(lambda x, y: x * y, d)
    return d * n

def convdim(dims, scale, n):
    return (n, dims[0] // scale, dims[1] // scale)

class Generator(Chain):
    def __init__(self, n_z, out_shape):
        super(Generator, self).__init__(
            fc0=L.Linear(n_z, lindim(out_shape, 2**4, 256)),
            dc1=L.Deconvolution2D(256, 128, 4, stride=2, pad=1),
            dc2=L.Deconvolution2D(128, 64, 4, stride=2, pad=1),
            dc3=L.Deconvolution2D(64, 32, 4, stride=2, pad=1),
            dc4=L.Deconvolution2D(32, 1, 4, stride=2, pad=1),
            bn0=L.BatchNormalization(lindim(out_shape, 2**4, 256)),
            bn1=L.BatchNormalization(128),
            bn2=L.BatchNormalization(64),
            bn3=L.BatchNormalization(32)
        )
        self.out_shape = out_shape

    def __call__(self, z, test=False):
        h = F.relu(self.bn0(self.fc0(z)))
        h = F.reshape(h, ((z.shape[0],) + convdim(self.out_shape, 2**4, 256)))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.sigmoid(self.dc4(h))
        return h

class Discriminator(Chain):
    def __init__(self, in_shape):
        super(Discriminator, self).__init__(
            c0=L.Convolution2D(1, 32, 4, stride=2, pad=1),
            c1=L.Convolution2D(32, 64, 4, stride=2, pad=1),
            c2=L.Convolution2D(64, 128, 4, stride=2, pad=1),
            c3=L.Convolution2D(128, 256, 4, stride=2, pad=1),
            fc4=L.Linear(lindim(in_shape, 2**4, 256), 512),
            #mbd=MinibatchDiscrimination(512, 32, 8),
            fc5=L.Linear(512, 512+32),  # Alternative to minibatch discrimination
            fc6=L.Linear(512+32, 2),
            bn1=L.BatchNormalization(64),
            bn2=L.BatchNormalization(128),
            bn3=L.BatchNormalization(256)
        )

    def __call__(self, x, minibatch_discrimination=True, test=False):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.bn1(self.c1(h)))
        h = F.leaky_relu(self.bn2(self.c2(h)))
        h = F.leaky_relu(self.bn3(self.c3(h)))
        h = self.fc4(h)

        #if minibatch_discrimination:
            #h = self.mbd(h)
        #else:
        h = F.leaky_relu(self.fc5(h))
        h = self.fc6(h)
        return h

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        return np.linspace(-self.range, self.range, N) + \
            np.random.random(N) * 0.01
