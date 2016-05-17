### JS Neural Network Algorithm

A javascript implementation of one of Michael Nielsen's neural network algorithms http://neuralnetworksanddeeplearning.com/.


The original purpose of this code was to experiment with classifying MNIST digits
in the browser in a reasonable amount of time and with low memory.
I ended up deciding it was not worth the value, people would not
want to wait even 30 seconds per run let alone up to an hour.


Running a network with 30 hidden nodes took this program almost 55 minutes,
while Michael Nielsen's algorithm took less then a minute using Python and Numpy.
Numpy handles matrices in an high performant way using a C (or C++) implementation.
This JS library includes some element-wise matrix functions which may be the bottleneck
in the algorithm, although more testing would be necessary.


This code served as a nice learning experience, but should not be used
by anyone.
