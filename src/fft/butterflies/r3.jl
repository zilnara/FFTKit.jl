
# TODO: select and compute these in correct type?
const R_3 = 1/3

# TODO: use AbstractIrrational to support computing this at any requested precision
const R3_K = -sin(2π/3)

function r3(x0::Number, x1::Number, x2::Number; mode::FFTMode=forward)
  a0 = x0
  a1 = x1 + x2
  a2 = x1 - x2

  b0 = a0 + a1
  b1 = a0 - 0.5a1
  b2 = R3_K * a2

  if mode == backward
    b2 = -b2
  elseif mode == inverse
    b0 =  R_3 * b0
    b1 =  R_3 * b1
    b2 = -R_3 * b2
  end

  c0 = b0
  c1 = b1 + b2 * im
  c2 = b1 - b2 * im

  c0, c1, c2
end

function r3!(dest::AbstractVector, x::AbstractVector{<:Number}; mode::FFTMode=forward)
  n = length(x)
  @assert mod(n, 3) == 0 && length(dest) >= length(x)

  @simd for i in 1:3:n
    dest[i], dest[i+1], dest[i+2] = @inline r3(x[i], x[i+1], x[i+2]; mode)
  end
  dest
end

function r3!(dest::AbstractArray, x::AbstractArray{<:Number}; mode::FFTMode=forward)
  @assert mod(size(x, 1), 3) == 0

  result = dest
  if ndims(dest) > 1
    @assert mod(size(dest, 1), 3) == 0
    dest = reshape(dest, :)
  end

  if ndims(x) > 1
    x = reshape(x, :)
  end

  r3!(dest, x; mode)
  result
end

function r3!(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r3!(x, x; mode)
end

function r3(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r3!(similar(x, promote_type(eltype(x), Complex{Float64})), x; mode)
end

cost(::Union{typeof(r3), typeof(r3!)}, n::Number=1) = Cost(
  COMPLEX_SUM     => 2n,
  COMPLEX_SCALE   => n / 3,
  TRIVIAL_CSCALE  => n / 3,
  TRIVIAL_CMULT   => 2n / 3,
)

