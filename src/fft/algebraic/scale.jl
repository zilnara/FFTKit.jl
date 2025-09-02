
# just uniformly scale every term
struct Scale
  scale :: Number
end

Base.:(==)(x::Scale, y::Scale) = (x.scale == y.scale)

function cost(x::Scale, n::Int=1)
  x = x.scale
  if x isa Rotation
    return cost(x, n)
  end

  if x isa Complex
    if iszero(x.im)
      # pure real, just drop the imaginary part and continue
      x = x.re
    elseif iszero(x.re)
      # pure imaginary
      if isone(abs(x.im))
        return Cost(TRIVIAL_CMULT => n)
      elseif ispow2(abs(x.im))
        return Cost(TRIVIAL_CMULT => n, TRIVIAL_CSCALE => n)
      else
        return Cost(TRIVIAL_CMULT => n, COMPLEX_SCALE => n)
      end
    else
      return Cost(COMPLEX_MULT => n)
    end
  end

  if isone(abs(x))
    Cost()
  elseif ispow2(abs(x))
    Cost(TRIVIAL_CSCALE => n)
  else
    Cost(COMPLEX_SCALE => n)
  end
end

prefer_inplace(::Scale) = true
radix(::Scale) = Int[]
transform_length(::Scale) = 1
dual(x::Scale) = x
Base.repeat(x::Scale; inner::Int=1, outer::Int=1) = x

output_order(::Scale, in_order::AbstractVector=1:1) = in_order
input_order(::Scale, out_order::AbstractVector=1:1) = out_order

function LinearAlgebra.mul!(dst::AbstractVector, x::Scale, src::AbstractVector; mode::FFTMode=forward)
  dst .= src * x.scale
end

Base.:(*)(x::Scale, y::Scale) = Scale(x.scale, y.scale)
Base.:(*)(x::Scale, y::AbstractArray) = x.scale * y
Base.inv(x::Scale) = Scale(inv(x.scale))

permute_cyclic(::Permutation, x::Scale) = x

function fft_rewrite_push_scales_right(x::Scale, y::Scale)
  x * y
end

# TODO: add other linear stages here if relevant
function fft_rewrite_push_scales_right(x::Scale, y::Union{Reorder,SDF,Twiddle})
  [y, x]
end

function fft_rewrite_delete_identities(x::Scale)
  if isone(x.scale)
    []
  else
    false
  end
end
