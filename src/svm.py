#!/usr/bin/python

"""
SVM model
"""
__authors__ = "Aydin Demircioglu"
__copyright__ = "Copyright 2014, Ruhr-Universitaet Bochum, Germany"
__credits__ = ["Aydin Demircioglu"]
__license__ = "MIT, WTFPL"


## copyright (C) 2014 by aydin demircioglu <mixmax /at/ cloned.de>
## License: WTFPL <http://sam.zoy.org/wtfpl>
##   0. You just DO WHAT THE FUCK YOU WANT TO.
##
## License: MIT (rough WTFPL equivalent)


import logging
import numpy

from pylearn2.models.model import Model
from sklearn import svm


logger = logging.getLogger(__name__)
# logger.debug("MLP changing the recursion limit.")



class SVM(Model):

	"""
	Class for an SVM.

	Parameters
	----------
	kwargs : dict
		Passed on to the superclass.
	k : int
		Number of clusters
	nvis : int
		Dimension of input
	convergence_th : float, optional
		Threshold of distance to clusters under which k-means stops
		iterating.
	max_iter : int, optional
		Maximum number of iterations. Defaults to infinity.
	verbose : bool
		WRITEME

	Notes
	-----
	"""

	def __init__(self, C = 1.0, 
				kernel = 'rbf',
				gamma = 1.0,
				cache_size = 1024,
				params=None,
				verbose = False):

		Model.__init__(self)

		self.C= C
		self.gamma = gamma
		self.cache_size = cache_size
		self.verbose = verbose
		self.kernel = kernel
#		self.input_space = VectorSpace()

		if params is None:
			# create dummies
			self.bias = 0
			self.alpha = numpy.zeros(1)
		else:
			self.bias = params[0]
			self.alpha = params[1]

		# this way we query predictions without saving anything 
		self.classifier = None
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



	def train_all(self, dataset, mu=None):
		"""
		Process kmeans algorithm on the input to localize clusters.

		Parameters
		----------
		dataset : WRITEME
		mu : WRITEME

		Returns
		-------
		rval : bool
			WRITEME
		"""

		X = dataset.X #get_design_matrix()
		y = dataset.y

		clf = svm.SVC(C = self.C, cache_size = self.cache_size, gamma = self.gamma, kernel = self.kernel,
				class_weight = None, coef0=0.0, degree=3,
				max_iter = -1, probability=False, random_state=None,
				shrinking = True, tol=0.001, verbose=False)
		clf.fit (X, y)
		
		# save the last classifier
		self.classifier = clf

		# get model parameter
		self.alpha = clf.dual_coef_
		self.bias = clf.intercept_
		# TODO
		self.params = clf.get_params (deep=True)


		#TODO:self.mu = sharedX(mu)?
		self._params = [self.alpha, self.bias]
		print(self._params)


#	@wraps(Model.continue_learning)
	def continue_learning(self):
		# One call to train_all currently trains the model fully,
		# so return False immediately.
		return False



	def get_params(self):
		"""
		.. todo::

			WRITEME
		"""
		return [param for param in self._params]



	def __call__(self, X):
		"""
		Compute for each sample its probability to belong to a cluster.

		Parameters
		----------
		X : numpy.ndarray
			Matrix of sampless of shape (n, d)

		Returns
		-------
		WRITEME
		"""
		
		if (self.classifier == None):
			# reinit
			clf = svm.SVC(C = self.C, cache_size = self.cache_size, gamma = self.gamma, kernel = self.kernel,
				class_weight = None, coef0=0.0, degree=3,
				max_iter = -1, probability=False, random_state=None,
				shrinking = True, tol=0.001, verbose=False)
			clf.set_params(self.params)
			self.classifier = clf
			
		predictions = self.classifier.predict(X)
		return predictions

	# Use version defined in Model, rather than Block (which raises
	# NotImplementedError).
#	get_input_space = Model.get_input_space
#	get_output_space = Model.get_output_space

