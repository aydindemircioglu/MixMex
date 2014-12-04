#!/usr/bin/python

## copyright © 2014 by aydin demircioglu <mixmax /at/ cloned.de>
## License: WTFPL <http://sam.zoy.org/wtfpl>
##   0. You just DO WHAT THE FUCK YOU WANT TO.
##
## License: MIT (rough WTFPL equivalent)

from pylearn2.costs.cost import Cost, DefaultDataSpecsMixin

class MoECost (Cost, DefaultDataSpecsMixin):
    # Here it is assumed that we are doing supervised learning
    supervised = True

    def expr(self, model, data, **kwargs):
        space, source = self.get_data_specs(model)
        space.validate(data)

        inputs, targets = data
        outputs = model.logistic_regression(inputs)
        loss = -(targets * T.log(outputs)).sum(axis=1)
        return loss.mean()

