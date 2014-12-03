
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_svmlight_file
from sklearn import preprocessing
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

import numpy as np
import os



class DataSet:
	def __init__(self, name = None, X = None, y = None, X_test = None, y_test = None, verbose = True):
		self.name = name
		self.X = X
		self.y = y
		self.X_test = X_test
		self.y_test = y_test
		self.verbose = verbose

		if X != None and y != None and X_test == None and y_test == None:
			self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size=0.3)
			
			
	def splitTrainTest (self, test_size = 0.3):
		# TODO: first merge everything? not to loose data
		# TODO: add validation set 
		self.X, self.X_test, self.y, self.y_test = train_test_split(self.X, self.y, test_size = test_size)


	def normalize(self):
		self.X = normalize(X)
		self.X_test = normalize(X_test)

		
	def scale(self):
		# FIXME: this cannot work this way, scaling must be done with
		# the joined set.
		if (self.X != None):
			self.X = preprocessing.scale(self.X)
		if (self.X_test != None):
			self.X_test = preprocessing.scale(self.X_test)


	def get_cv_performance(self, clf):
		X = self.X
		y = self.y
		scores = cross_validation.cross_val_score(clf, X, y, cv=cross_validation.KFold(n=X.shape[0], n_folds=10))#cv=10)
		print("%s error rate: %0.2f (+/- %0.2f)" % (self.name, 1.-scores.mean(), scores.std() * 2))


	def load(self, dataset = None, data_dir = "/home/drunkeneye/lab/data", verbose = None):
		if verbose == None:
			verbose = self.verbose
			
		if dataset == None:
			dataset = self.name
		# first try to load the data 'directly'
		try:
			filePath = os.path.join(data_dir, dataset, dataset)
			if verbose:
				print("  Trying to load data set from {}". format(filePath))
			self.X, self.y = load_svmlight_file(filePath)
			self.X = np.asarray(self.X.todense())
			if verbose:
				print ("    Loaded from {}". format( filePath))
			return
		except:
			pass
		
		# next try
		try:
			filePath = os.path.join(data_dir, dataset, dataset + ".combined.scaled")
			if verbose:
				print("  Trying to load data set from {}". format(filePath))
			X, y = load_svmlight_file(filePath)
			X = np.asarray(X.todense())
			if verbose:
				print ("    Loaded from {}". format( filePath))
			return 
		except:
			pass

#    y = y.reshape(y.shape[0], 1)
#			return DenseDesignMatrix (X=X, y=y)
#			return DataSet.DataSet(X, y, dataset)

#split (XX)

#getPartitionByIndex

#getPartition(5, 6)



