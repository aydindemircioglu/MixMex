import os
import pylearn2
path = os.path.join(pylearn2.__path__[0], 'scripts', 'tutorials', 'multilayer_perceptron', 'mlp_tutorial_part_2.yaml')
with open(path, 'r') as f:
    train = f.read()
hyper_params = {'train_stop' : 50000,
                'valid_stop' : 60000,
                'dim_h0' : 500,
                'max_epochs' : 10000,
                'save_path' : '.'}
train = train % (hyper_params)
print train

train = yaml_parse.load(train)
train.main_loop()
