
# TODO
#  Not sure whether they should go here or with ffts, but some xcorr components:
#     * zero-padded fft (FFT where first stage is a SDF but some subset of inputs is empty)
#     * truncated fft (FFT where last stage is a SDF but some subset of outputs are discarded)
#     * overlapped FFT (structure using zero-padded FFT and delay lines to implement equivalent of overlap-save with less resources)

# the basic concepts are easy to understand with reference to a transform's flow-graph, but I'm
# not sure how much work they will be to extend the existing FFTPlan structure to understand.
# 
# The zero-padded and truncated versions are dual of each other. In both cases, we can consider
# the operation to be "trivializing" some subset of inputs or outputs. In the flowgraph, we just
# delete edges from trivialized inputs/outputs. We can then iteratively delete any butterflies,
# twiddles, or reorders, propagating the trivialization across the graph.

struct Trivialize
  mask :: BitVector
end

# The overlap case is a bit more complex. The general concept is to take a zero-padded transform
# where the overlap portion is zeroed, then delay that output, feeding that output into the final
# twiddles and butterflies.
# 
# It turns out that there's a pretty simple variant in the case we care most about: A forward
# transform in normal (natural-in, bitrev-out) order, with overlap equal to half the transform
# size.. This is the situation on the signal input of a standard FFT-based cross-correlator.
# Set up a N-point zero-padded transform, and a N/2 point delay line. The overlapped output
# is the sum of the non-delayed path plus a twiddled version of the delayed path (in particular,
# the delayed path is twiddled by [+1, -1] - every other sample is negated. This implements a
# 180-degree phase shift due to the N/2 sample delay).

struct OverlapDelay
  depth :: Int
end

# take the inputs, divide them into 2 parts, multiply them point-wise,
# and produce just one output stream.
struct DotProduct
  depth :: Int
end

# conjugate the indicated lanes
struct Conjugation
  mask :: BitVector
end

# TODO: better handling of the fact that cross-correlation combines forward and inverse transforms in
#       a single structure?
# probably just drop FFTMode "inverse", use only backward, and add a Scale primitive to FFTStage. Then
# you can either run a transform as inverse/backward/etc or invert it
struct InvFFT
  plan :: FFTPlan
end

# Together, these primitives together with the existing FFT primitives allow construction of
# a wide variety of cross-correlator/convolution engines.
# 
# TODO: work toward integrating some these into the FFTStage type. For now, keeping them separate
#       and just experimenting with them on an ad-hoc basis here.
const XCorrStage = Union{FFTStage, Trivialize, OverlapDelay, DotProduct}

