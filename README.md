FFTKit
======

This package provides a bunch of tools for building and understanding FFT implementations. For example:

- Implementations of several common small butterflies
- An algebraic representation of FFT implementations in terms of stages, and functions for simulating or manipulating the reprepresented transforms, as well as for extracting information about the transform that would be used to realize the implementation such as twiddle generation parameters, input and output sample orders, etc.
- Conversion of that algebraic representation to various other representations, such as square or lower-triangular matrix form. (TODO)
- Metrics for estimating costs of FFT implementations (numbers of various types of operation needed per sample) (Needs more work)

This is a concept sketch / work-in-progress. I have some ideas about a significant rework of the basis of the system, which I may or may not get around to any time soon. Depends on availability of free time.

References
----------

The code in this library uses the mathematical insight of several papers, interspersed with a bit of novel work to generalize it.

Non-exhaustive list of references containing ideas used or expanded upon or perhaps just mentioned in this codebase:

* F. Qureshi and O. Gustafsson, "Generation of All Radix-2 Fast Fourier Transform Algorithms Using Binary Trees", in 2011 20th European Conference on Circuit Theory and Design (ECCTD), pp. 677-680.
* Mario Garrido Galvez, "A New Representation of FFT Algorithms Using Triangular Matrices," IEEE Transactions on Circuits and Systems Part 1, 2016. 63(10), pp. 1735-1745.

