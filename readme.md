
MixMex
======

Mixture of SVM Experts,


Goal
====

the goal of the hackathon was to get something along the lines of 
collobert+bengio+bengio.
basically it should be a use case for my local directory, so that i
can check whether i can really use that for experiments.

from the many python packages that do support neural networks,
i chose pylearn2. the reason for that is twofold: it has quite good
support, many developers and is actively developed. this is not
true for every package. furthermore, it is able to run on the gpu
and has advanced features like dropout. so it is naturally a good 
choice, even over caffe.


Paper
====

It became clear that the paper 'Scaling Large Learning Problems
with Hard Parallel Mixtures' is an updated version of the old one.
(In case you need to find this, it is written down in a few lines at
the end of the introduction, without any citation)-- but the new one
more or less states that replacing the SVM experts with MLPs yields
better results. This is actually contrary to what i would like to have.
This better is not really accuracy-wise, but time-wise. One need
still to compare the results with the comment on the old paper,
where the whole setup is just randomized trees. The authors there
claim to be better then Mixture of Experts.
While searching i found also 
'Fast modular network implementation for support vector machines'
by GB Huang, KZ Mao, CK Siew, which might also be very interesting.

Also see the thesis of Collobert for maybe some more details-- and
now, that i browsed through it, one can read:

	The generalization performance of a hard mixture with a local gater were
in all cases worst than what we obtained with a global gater, especially with
SVM experts. Obviously, a local gater is much less powerful than a global
gater for re-distributing examples to the experts. Moreover even if we were
able to obtain reasonable generalization results using the hard mixture with a
local gater in a short time, we must admit that tuning the Gaussian Mixture
Models in the local gater is very tricky compared to the tuning of the MLP
global gater. In the end, the global gater seems to be the way to go...

what the?...



Some words on PyLearn2
===================

- There seem to exist a SVM model under pylearn2/models/svm.py.
i did only see that towards the end of the day, so i ignore it basically.f


Problems
=======

In the paper, the authors state that 'the same hyper-parameters
were selected for all iterations of the algorithm and for all the
SVM experts'. But how to choose them? We cannot just take the
whole data set and do some tuning on them. Even on a sub-sample
tuning is a very cost-intensive process, and here, in difference to
normal SVM ensembles, there is also a neural network in the loop,
complicating the search for the best parameters. 


Datasets
======

An artificial toy example was created, which can be seen in the 
figures in the paper. 

On the other hand, covertype was used, and the LIBSVM data set
has the very same multiclass to binary division. Both seem to 
differ in scaling, the LIBSVM dataset is not scaled, while the paper
says that each feature was scaled to [0,1]. 

Numbers95 were used, but this is another proprietary data set,
which needs some kind of agreement to be used.



Parameter Tuning
=============

As already mentioned under Problems, it is not clear, how to find
the 'correct' parameters. There are the following questions, one
should actually answer:
	-Does the Mixture of Experts method 'need' weak classifiers?
		(in this case, 
	-If parameter tuning is done on one of the partitions, is this 
		near 'enough' to the optimal C,g for the whole dataset? 
	-Will averaging over all C,g over all partitions yield a good C,g?
	
	

Implementation Details
=================

-The gater can be warm-started by using the weights obtained in
the previous iteration.
-The SVMs could also be warm-started, though by reshuffling the
data probably this is not the best idea?


Other Knowledge
=============

There are couple of other Python NN software Implementations:

* Hebel (based on Theano, seems to need GPU?)
* PDNN (also based on Theano, but seems to be smaller)
* PyBrain (no modern things like Dropout, said to be very slow)
* Caffe (only real alternative to Theano, as it seems, matter of taste)
* breze
	
For parameter-tuning the following libraries were found:

* Hyperopt and its wrapper for sklearn, Hyperopt-sklearn
* HPOlib
* Spearmint
* BayesOpt
* SMAC
* REMBO
	
Some minor infos are on FastML.com?



Large Dataset sites
==============

A list of pages that might have a lot of different LARGE data sets (or infos):

- https://www.quandl.com/
- http://aws.amazon.com/datasets
- https://www.google.com/cse/publicurl?cx=002720237717066476899:v2wv26idk7m
- http://academictorrents.com/
- http://commoncrawl.org/
- http://lemire.me/blog/archives/2012/03/27/publicly-available-large-data-sets-for-database-research/
- http://www.kdnuggets.com/datasets/index.html
- http://www.datawrangling.com/some-datasets-available-on-the-web/
- http://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public
- https://delicious.com/tag/dataset
- http://labrosa.ee.columbia.edu/millionsong/
- http://grouplens.org/datasets/
- http://www.quora.com/Where-can-I-find-large-datasets-open-to-the-public/answers/784181
	
	
	