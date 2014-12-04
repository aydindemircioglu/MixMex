#!/usr/bin/python

## copyright (C) 2014 by aydin demircioglu <mixmax /at/ cloned.de>
## License: WTFPL <http://sam.zoy.org/wtfpl>
##   0. You just DO WHAT THE FUCK YOU WANT TO.
##
## License: MIT (rough WTFPL equivalent)


import pylearn2.config.yaml_parse as yaml_parse
import DataSet
import svm 
import numpy

fp = open('example_SVM.yaml')
model = yaml_parse.load(fp)
#print model
fp.close()

D = DataSet.DataSet(verbose = True)
D.load("spektren")
D.normalize()

S = svm.SVM(C = 3, gamma = 0.1, kernel = "rbf")
S.train_all (D)
P = S(D.X)
d = P - D.y
print(numpy.mean(d)/2)
#print(S)
#print(P)
