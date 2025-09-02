
struct Conj
  mask :: BitVector

  # TODO: normalizing constructor (remove cycles)
end

Conj() = Conj([1])

Base.:(==)(x::Conj, y::Conj) = x.mask == y.mask

cost(::Conj, n::Int=1) = Cost()

prefer_inplace(::Conj) = true
radix(::Conj) = Int[]
transform_length(::Conj) = 1
dual(x::Conj) = x # TODO: I don't actually know what it would mean to take the dual of a conjugation stage
Base.repeat(x::Conj; inner::Int=1, outer::Int=1) = Conj(repeat(x.mask; inner))

output_order(::Conj, in_order::AbstractVector=1:1) = in_order
input_order(::Conj, out_order::AbstractVector=1:1) = out_order

function LinearAlgebra.mul!(dst::AbstractVector, x::Conj, src::AbstractVector; mode::FFTMode=forward)
  n = length(x.mask)
  @simd for i in 1:length(src)
    j = 1+mod(i-1, n)
    dst[i] = x.mask[j] ? conj(src[i]) : src[i]
  end
  dst
end

Base.inv(x::Conj) = x

Base.:(*)(x::Conj, y::AbstractArray) = mul!(similar(y), x, y)

# TODO: permute_cyclic

function fft_rewrite_delete_identities(x::Conj)
  if all(iszero, x.mask)
    []
  else
    false
  end
end
