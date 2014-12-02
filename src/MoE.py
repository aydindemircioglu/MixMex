class MoE:

    def __init__(self, nvis, nhid, iscale=0.1,
            activation_fn=numpy.tanh,
            params=None):

        self.nvis = nvis
        self.nhid = nhid
        self.activation_fn = activation_fn

        if params is None:
            self.W = iscale * numpy.random.randn(nvis, nhid)
            self.bias_vis = numpy.zeros(nvis)
            self.bias_hid = numpy.zeros(nhid)
        else:
            self.W = params[0]
            self.bias_vis = params[1]
            self.bias_hid = params[2]

        print self

    def __str__(self):
        rval  = '%s\n' % self.__class__.__name__
        rval += '\tnvis = %i\n' % self.nvis
        rval += '\tnhid = %i\n' % self.nhid
        rval += '\tactivation_fn = %s\n' % str(self.activation_fn)
        rval += '\tmean std(weights) = %.2f\n' % self.W.std(axis=0).mean()
        return rval

    def save(self, fname):
        fp = open(fname, 'w')
        pickle.dump([self.W, self.bias_vis, self.bias_hid], fp)
        fp.close()
