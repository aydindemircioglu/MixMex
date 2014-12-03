#!/usr/bin/python

import pylearn2.config.yaml_parse as yaml_parse
from sklearn.metrics import f1_score
from sklearn import svm, grid_search, datasets
from sklearn.metrics import classification_report

import DataSet

fp = open('example_SVM.yaml')
model = yaml_parse.load(fp)
print model
fp.close()


print ("Mixture of SVM Experts")

D = DataSet.DataSet ("australian")
D.load()
D.scale()
D.splitTrainTest(0.5)

from sklearn import svm
svc_rbf = svm.SVC(kernel='rbf', gamma=1e2)
svc_rbf.fit(D.X, D.y) 
score = svc_rbf.score(D.X, D.y)
print(score)
score = svc_rbf.score(D.X_test, D.y_test)
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
    y_true, y_pred = D.y_test, clf.predict(D.X_test)
    print classification_report(y_true, y_pred)
    print
