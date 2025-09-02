
# TODO: triangular rep based on:
# Garrido, M. (2016). A New Representation of FFT Algorithms Using Triangular Matrices.
# IEEE Transactions on Circuits and Systems, I, 63(10), 1737-1745

function fft_matrix(x::Union{FFTStage, FFTPlan}, n::Int = transform_length(x))
  # TODO: represent these things in a way that explicitly captures their sparsity: diagonals, block-diagonals, permutations, etc
  # TODO: infer simplest type that captures the spirit
  basis = diagm(repeat(ComplexF64[1], n))
  for i in 1:n
    row = @view basis[:, i]
    mul!(row, x, row)
  end
  basis
end

function fft_matrices(plan::FFTPlan)
  n = transform_length(plan)
  [fft_matrix(s, n) for s in plan.stages]
end

function dft_matrix(n::Int)
  basis = diagm(repeat([1], n))
  fft(basis, 1)
end
