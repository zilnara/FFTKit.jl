
module FFTKit

using AbstractFFTs
using Distributions
using FFTW
using LinearAlgebra
using Permutations
using Primes
using Random
using Statistics

# TODO: better way to package things to avoid needing to bring in some of these dependencies, which in some
#       cases bring in lots of stale upstream stuff
using Symbolics
using SymbolicUtils
# using TikzPictures
using Unitful

include("fft/metrics.jl")

include("fft/butterflies.jl")
include("fft/btree.jl")
include("fft/index.jl")
include("fft/rotation.jl")

include("fft/algebraic.jl")
include("fft/matrix_fft.jl")

include("xcorr/algebraic.jl")

end

