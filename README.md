# nmf: FPA for NMF with the KL divergence

This package implements a gradient descent method for non-negative matrix factorization (NMF) with the Kullback-Leibler (KL) divergence. Because of the lack of smoothness of the KL loss, we use a first-order primal-dual algorithm (FPA) based on the [Chambolle-Pock algorithm](https://hal.archives-ouvertes.fr/hal-00490826/document). We provide an efficient heuristic way to select step-sizes, and all required computations may be obtained in closed form.

## References

Felipe Yanez, and Francis Bach. [Primal-Dual Algorithms for Non-negative Matrix Factorization with the Kullback-Leibler Divergence](https://arxiv.org/abs/1412.1788), arXiv:1412.1788, 2014.

Felipe Yanez, and Francis Bach. [Primal-Dual Algorithms for Non-negative Matrix Factorization with the Kullback-Leibler Divergence](http://ieeexplore.ieee.org/abstract/document/7952558/?reload=true), IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), New Orleans, LA, USA, 2017.
