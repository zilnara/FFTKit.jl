
function r4(x0::Number, x1::Number, x2::Number, x3::Number; mode::FFTMode=forward)
  a0 = x0 + x2; a1 = x1 + x3
  a2 = x0 - x2; a3 = x1 - x3; a3 *= im

  if mode == inverse
    a0 *= 0.25; a1 *= 0.25
    a2 *= 0.25; a3 *= 0.25
  end

  b0 = a0 + a1; b1 = a2 - a3
  b2 = a0 - a1; b3 = a2 + a3

  if mode != forward
    b1, b3 = b3, b1
  end

  (b0, b1, b2, b3)
end

function r4!(dest::AbstractVector, x::AbstractVector{<:Number}; mode::FFTMode=forward)
  n = length(x)
  @assert mod(n, 4) == 0 && length(dest) >= length(x)

  @simd for i in 1:4:n
    dest[i], dest[i+1], dest[i+2], dest[i+3] = @inline r4(x[i], x[i+1], x[i+2], x[i+3]; mode)
  end
  dest
end

function r4!(dest::AbstractArray, x::AbstractArray{<:Number}; mode::FFTMode=forward)
  @assert mod(size(x, 1), 4) == 0

  result = dest
  if ndims(dest) > 1
    @assert mod(size(dest, 1), 4) == 0
    dest = reshape(dest, :)
  end

  if ndims(x) > 1
    x = reshape(x, :)
  end

  r4!(dest, x; mode)
  result
end

function r4!(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r4!(x, x; mode)
end

function r4(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r4!(similar(x, promote_type(eltype(x), Complex{Int})), x; mode)
end

cost(::Union{typeof(r4), typeof(r4!)}, n::Number=1) = Cost(
  COMPLEX_SUM     => 2n,
  TRIVIAL_CMULT   => n / 4,
)


