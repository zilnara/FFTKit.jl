
function r2(x0::Number, x1::Number; mode::FFTMode=forward)
    a0 = x0 + x1
    a1 = x0 - x1

    if mode == inverse
      a0 *= 0.5
      a1 *= 0.5
    end

    a0, a1
end

# TODO: should the assertions here maybe just be warnings?

function r2!(dest::AbstractVector, x::AbstractVector{<:Number}; mode::FFTMode=forward)
  n = length(x)
  @assert iseven(n) && length(dest) >= length(x)

  @simd for i in 1:2:n
    dest[i], dest[i+1] = @inline r2(x[i], x[i+1]; mode)
  end
  dest
end

function r2!(dest::AbstractArray, x::AbstractArray{<:Number}; mode::FFTMode=forward)
  @assert iseven(size(x, 1))

  result = dest
  if ndims(dest) > 1
    @assert iseven(size(dest, 1))
    dest = reshape(dest, :)
  end

  if ndims(x) > 1
    x = reshape(x, :)
  end

  r2!(dest, x; mode)
  result
end

function r2!(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r2!(x, x; mode)
end

function r2(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r2!(similar(x), x; mode)
end

cost(::Union{typeof(r2), typeof(r2!)}, n::Number=1) = Cost(
  COMPLEX_SUM     => n,
)

