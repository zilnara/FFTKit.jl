
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

function function_matrix(n::Int, linear_function)
  basis = I(n)
  if applicable(linear_function, basis, dims=1)
    linear_function(basis, dims=1)
  elseif applicable(linear_function, basis, 1)
    linear_function(basis, 1)
  else
    col_1 = linear_function(basis[:,1])
    output = similar(col_1, length(col_1), n)
    output[:,1] = col_1

    for i in 2:n
      output[:,i] = linear_function(basis[:,i])
    end

    output
  end
end

function dft_matrix(n::Int)
  basis = diagm(repeat([1], n))
  fft(basis, 1)
end
