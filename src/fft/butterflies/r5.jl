
const R_5 = 1/5

# TODO: use AbstractIrrational to support computing these at any requested precision
const R5_K0 = cos(4π/5)
const R5_K10 = -sin(4π/5)
const R5_K11 = -sin(2π/5) + sin(4π/5)
const R5_K12 =  sin(2π/5) + sin(4π/5)

function r5(x0::Number, x1::Number, x2::Number, x3::Number, x4::Number; mode::FFTMode=forward)
  a0 = x0; a1 = x1 + x4; a2 = x2 + x3; a3 = x3 - x2; a4 = x1 - x4
  b0 = a0; b1 = a1; b2 = a2; b3 = a3; b4 = a4; b5 = a1 - a2; b6 = a3 + a4
  c0 = b0; c1 = b0 - 0.5b1; c2 = b0 - 0.5b2; c3 = R5_K11 * b3; c4 = R5_K12 * b4; c5 = R5_K0 * b5; c6 = R5_K10 * b6; c7 = b1 + b2
  d0 = c0 + c7; d1 = c1 - c5; d2 = c2 + c5; d3 = (c3 + c6) * im; d4 = (c4 + c6) * im

  if mode == backward
    d3 = -d3
    d4 = -d4
  elseif mode == inverse
    d0 =  R_5 * d0
    d1 =  R_5 * d1
    d2 =  R_5 * d2
    d3 = -R_5 * d3
    d4 = -R_5 * d4
  end

  (d0, d1-d4, d2+d3, d2-d3, d1+d4)
end

function r5!(dest::AbstractVector, x::AbstractVector{<:Number}; mode::FFTMode=forward)
  n = length(x)
  @assert mod(n, 5) == 0 && length(dest) >= length(x)

  # TODO: handle non-one-based indices or determine that it's not necessary
  @inbounds @simd for i in 1:5:n
    dest[i], dest[i+1], dest[i+2], dest[i+3], dest[i+4] = @inline r5(x[i], x[i+1], x[i+2], x[i+3], x[i+4]; mode)
  end
  dest
end

function r5!(dest::AbstractArray, x::AbstractArray{<:Number}; mode::FFTMode=forward)
  @assert mod(size(x, 1), 5) == 0

  result = dest
  if ndims(dest) > 1
    @assert mod(size(dest, 1), 5) == 0
    dest = reshape(dest, :)
  end

  if ndims(x) > 1
    x = reshape(x, :)
  end

  r5!(dest, x; mode)
  result
end

function r5!(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r5!(x, x; mode)
end

function r5(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r5!(similar(x, promote_type(float(eltype(x)), Complex{Int})), x; mode)
end

cost(::Union{typeof(r5), typeof(r5!)}, n::Number=1) = Cost(
  COMPLEX_SUM     => 18n / 5,
  COMPLEX_SCALE   => 4n / 5,
  TRIVIAL_CMULT   => 2n / 5,
  TRIVIAL_CSCALE  => 2n / 5,
)

