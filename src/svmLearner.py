
import numpy


class SVMLearner:

    def __init__(self, C, gamma, 
            params=None):

        self.C= C
        self.gamma = gamma

        if params is None:
            self.alpha = numpy.zeros(3)
            self.bias = 0
        else:
            self.alpha = params[0]
            self.bias = params[1]

        print self

    def __str__(self):
        rval  = '%s\n' % self.__class__.__name__
        rval += '\tC = %i\n' % self.C
        rval += '\tgamma = %i\n' % self.gamma
#        rval += '\talpha = %s\n' % str(self.activation_fn)
        rval += '\tmean std(alpha) = %.2f\n' % self.alpha.std(axis=0).mean()
        rval += '\tbias = %s\n' % str(self.bias)
        return rval

    def save(self, fname):
        fp = open(fname, 'w')
        pickle.dump([self.C, self.gamma, self.alpha, self.bias], fp)
        fp.close()

