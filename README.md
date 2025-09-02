FFTKit
======

This package provides a bunch of tools for building and understanding FFT implementations. For example:

- Implementations of several common small butterflies
- An algebraic representation of FFT implementations in terms of stages, and functions for simulating or manipulating the reprepresented transforms, as well as for extracting information about the transform that would be used to realize the implementation such as twiddle generation parameters, input and output sample orders, etc.
- Conversion of that algebraic representation to various other representations, such as square or lower-triangular matrix form. (TODO)
- Metrics for estimating costs of FFT implementations (numbers of various types of operation needed per sample) (Needs more work)

This is a concept sketch / work-in-progress. I have some ideas about a significant rework of the basis of the system, which I may or may not get around to any time soon. Depends on availability of free time.
