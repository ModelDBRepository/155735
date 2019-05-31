This directory contains files to reproduce the simulations presented
in the paper:

Couto, J., Linaro, D., De Schutter, E. and Giugliano, M.  "On the
Firing Rate Dependency Of the Phase Response Curves of rat Purkinje
Neurons In Vitro". PLOS Comp Biol 2015

The subdirectories contain the following items.

1. mod-files - contains the mod files that can be used in NEURON to
simulate the Khaliq-Raman Purkinje cell model using stochastic
representations of the ion channels (as well as the original mod
files) and the mod files for the NEURON version of the De Schutter and
Bower Purkinje Cell model.

2 python - contains the python scripts to perform the simulations in
Fig. 6 (KR folder) and Figs S1, S3 and S4 (DSB folder).

3. matlab - contains the minimal scripts to compute the PRC using the
direct method. Both the traditional and corrected (Phoka et al. 2010)
methods are available.

4. lcg - contains the configuration files that can be used to
reproduce the experiments with frequency clamp. Read the README.txt
file in this directory.

5. brian - contains BRIAN script to perform network simulations of
non-leaky LIF neurons that approximate the flat PRC profile described
for low firing rates.

For any question, please contact: 
	Joao Couto - jpcouto@gmail.com
	Daniele Linaro - daniele.linaro@uantwerpen.be
	Erik De Schutter - erik@tnb.ua.ac.be 
	Michele Giugliano - michele.giugliano@uantwerpen.be

