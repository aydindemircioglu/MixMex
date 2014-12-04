#!/usr/bin/python

## copyright © 2014 by aydin demircioglu <mixmax /at/ cloned.de>
## License: WTFPL <http://sam.zoy.org/wtfpl>
##   0. You just DO WHAT THE FUCK YOU WANT TO.
##
## License: MIT (rough WTFPL equivalent)

import os
import pylearn2

path = os.path.join('example_MLP.yaml')
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
