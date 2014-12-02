import pylearn2.config.yaml_parse as yaml_parse

fp = open('example1.yaml')
model = yaml_parse.load(fp)
print model
fp.close()

