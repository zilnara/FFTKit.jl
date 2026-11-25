
const R_8 = 1/8

const R8_K0 = sqrt(0.5)

function r8(x0::Number, x1::Number, x2::Number, x3::Number, x4::Number, x5::Number, x6::Number, x7::Number; mode::FFTMode=forward)
  (a0, a2, a4, a6) = r4(x0, x2, x4, x6; mode)
  (a1, a3, a5, a7) = r4(x1, x3, x5, x7; mode)

  b0 = a0; b1 = a1
  b2 = a2; b3 = a3 * R8_K0
  b4 = a4; b5 = a5         * -im
  b6 = a6; b7 = a7 * R8_K0

  if mode == backward || mode == inverse
    b5 = -b5
  end

  c0 = b0; c1 = b1; c2 = b2; c3 = b3 * ( 1 - 1im)
  c4 = b4; c5 = b5; c6 = b6; c7 = b7 * (-1 - 1im)

  if mode == backward || mode == inverse
    c3 =  c3 * im
    c7 = -c7 * im
  end

  (d0, d1) = r2(c0, c1; mode)
  (d2, d3) = r2(c2, c3; mode)
  (d4, d5) = r2(c4, c5; mode)
  (d6, d7) = r2(c6, c7; mode)

  (d0, d2, d4, d6, d1, d3, d5, d7)
end

function r8!(dest::AbstractVector, x::AbstractVector{<:Number}; mode::FFTMode=forward)
  n = length(x)
  @assert mod(n, 8) == 0 && length(dest) >= length(x)

  # TODO: handle non-one-based indices or determine that it's not necessary
  @inbounds @simd for i in 1:8:n
    dest[i], dest[i+1], dest[i+2], dest[i+3], dest[i+4], dest[i+5], dest[i+6], dest[i+7] =
      @inline r8(x[i], x[i+1], x[i+2], x[i+3], x[i+4], x[i+5], x[i+6], x[i+7]; mode)
  end
  dest
end

function r8!(dest::AbstractArray, x::AbstractArray{<:Number}; mode::FFTMode=forward)
  @assert mod(size(x, 1), 8) == 0

  result = dest
  if ndims(dest) > 1
    @assert mod(size(dest, 1), 8) == 0
    dest = reshape(dest, :)
  end

  if ndims(x) > 1
    x = reshape(x, :)
  end

  r8!(dest, x; mode)
  result
end

function r8!(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r8!(x, x; mode)
end

function r8(x::AbstractArray{<:Number}; mode::FFTMode=forward)
  r8!(similar(x, promote_type(float(eltype(x)), Complex{Int})), x; mode)
end

# TODO extend cost calculation system to understand multi-lane structures?
cost(::Union{typeof(r8), typeof(r8!)}, n::Number=1) = sum([
  cost(r4, n),
  cost(r2, n),
  Cost(
    COMPLEX_SCALE => 2n / 8,
    TRIVIAL_CMULT => 1n / 8,
  )
])

