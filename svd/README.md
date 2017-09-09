# Singular Value Decomposition Example

In this example, Singular Value Decomposition (SVD) is used for compactizing a neural network. With SVD, one could approximate an affine matrix with two matricesi.e., $W = USV$. this only makes the computational complexity higher and memory space larger. However, if one uses a row-rank approximation, i.e., $W = US'V$, where S' is the diagonal matrix which contains eigen values in the higher order, but a few values are uesd, then the computational complexity and memory space are reduced.

The follows show the workflow when one would like to compactize a network by using `SVD`.

1. Train a reference network 
2. Train a small network after applying SVD to an affine layer

`classification.py` corresponds to the first one, and `classification_svd.py` does the second.

This example uses a relatively fat network called LeNet which contains two affine.




