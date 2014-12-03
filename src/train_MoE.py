#!/usr/bin/python

#from pylearn2.datasets.sparse_dataset import SparseDataset

import os
import numpy as np

import DataSet

from sklearn.datasets import load_svmlight_file
from sklearn.metrics import f1_score
from sklearn import svm, grid_search, datasets

numberOfExperts = 32
maxIter = 128



def expertWeight (x, i):
	w = 0
	p = w * x
	return p


def expertOutput (i):
	d = 0
	return d


def MoEOutput (x):
	output = 0
	for i in range(numberOfExperts):
		output += expertWeight (i) * expertOutput(x, i) 
	output = transferFunction (output)
	return output


def costForPoint (x, y):
	C = MoEOutput(x) - y
	return C

def cost ():
	# for all data set points
	C = 0
	for i in range(data):
		C += costForPoint (data(x), y)**2
	return (C)


def main ():
	# divide data set into M random sets
	
	validationErrorGoesUp = False
	while (iter < maxIter) and (validationErrorGoesUp):
		# train svm i on dataset i
		
		# train gater minimizing cost function on WHOLE dataset
		
		# reconstruct subsets:
		
		# get array of weights
		# sort array
		weights = []
		
		# reset the partitions to be empty
		
		# reassign data point to expert
		for d in datasets:
			for i in weights:
				if examples(expert) < N/M + c:
					assign (d, i)
		
	
	
	
print ("Mixture of SVM Experts")

D = getDataset ("australian")

from sklearn import svm
svc_rbf = svm.SVC(kernel='rbf', gamma=1e2)
svc_rbf.fit(D.X, D.y) 
score = svc_rbf.score(D.X, D.y)
print(score)
score = svc_rbf.score(D.Xtest, D.ytest)
print(score)


tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

scores = [('f1', f1_score)]

for score_name, score_func in scores:
    print "# Tuning hyper-parameters for %s" % score_name
    print

    clf = grid_search.GridSearchCV( svm.SVC(), tuned_parameters, score_func=score_func, n_jobs=-1, verbose=2 )
    clf.fit(D.X, D.y)

    print "Best parameters set found on development set:"
    print
    print clf.best_estimator_
    print
    print "Grid scores on development set:"

    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params)
    print

    print "Detailed classification report:"
    print
    print "The model is trained on the full development set."
    print "The scores are computed on the full evaluation set."
    print
    y_true, y_pred = Y_test, clf.predict(X_test)
    print cross_validation.classification_report(y_true, y_pred)
    print
