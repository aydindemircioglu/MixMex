#!/usr/bin/python

#
## copyright © 2014 by aydin demircioglu <mixmax /at/ cloned.de>
## License: MIT (rough WTFPL <http://sam.zoy.org/wtfpl> equivalent)
##   0. You just DO WHAT THE FUCK YOU WANT TO.
#

import numpy
import theano.tensor as T
from pylearn2.models.model import Model
from pylearn2.space import VectorSpace
from pylearn2.utils import sharedX


class MoE(Model):
    def __init__(self, nvis, nclasses):
#        super(LogisticRegression, self).__init__()

        self.nvis = nvis
        self.nclasses = nclasses

        W_value = numpy.random.uniform(size=(self.nvis, self.nclasses))
        self.W = sharedX(W_value, 'W')
        b_value = numpy.zeros(self.nclasses)
        self.b = sharedX(b_value, 'b')
        self._params = [self.W, self.b]

        self.input_space = VectorSpace(dim=self.nvis)
        self.output_space = VectorSpace(dim=self.nclasses)

    def mixtureOfExperts (self, inputs):
        return T.nnet.softmax(T.dot(inputs, self.W) + self.b)
