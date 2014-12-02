#!/usr/bin/python

import pylearn2.config.yaml_parse as yaml_parse

fp = open('example_SVM.yaml')
model = yaml_parse.load(fp)
print model
fp.close()

