
# This is a WEIRD primitive but I kinda like it.
# The dot-product operation of a convolution, paired with the pointwise division operator of a naive deconvolution,
# can be considered a special case of a more general concept: an adjoint of a radix-2 butterfly under a logarithm.
#
# That is, if we have:
#     y1 = x1 * x2
#     y2 = x1 / x2
#
# Then we can equivalently view it as:
#     ln y1 = ln x1 + ln x2
#     ln y2 = ln x1 - ln x2
#
# This gives a nice symmetrical structure which can also be pseudo-inverted (it's not truly an inverse because of
# the multiplicity of complex logarithms (or, equivalently, the double-cover of z -> z^2), but it's an inverse in
# the sense that given f::AdjointR2, the composition f⁻¹ ∘ f is idempotent: f⁻¹ ∘ f ∘ f⁻¹ ∘ f ≡ f⁻¹ ∘ f)
#     ln x1 ~= (ln y1 + ln y2) / 2
#     ln x2 ~= (ln y1 - ln y2) / 2
#
# Or, equivalently:
#     x1^2 ~= sqrt(y1 * y2)
#     x2^2 ~= sqrt(y1 / y2)
#
# We could generalize this further, and maybe someday I'd like to explore that idea, but I don't see a particular
# use for it right now. But I'm gonna go ahead and implement this weird little primitive for now because it's an
# interesting idea that happens to fit into the overall structural framework I've got going for FFTs already.
#
# We can then use and manipulate this primitive similar to a R2 SDF, for the purpose of constructing a convolution.
# For example, see the definions of convolution and cross-correlation plans in test.jl

struct AdjointR2
  depth :: Int
  inverse :: Bool
end

AdjointR2(depth::Int) = AdjointR2(depth, false)

Base.:(==)(x::AdjointR2, y::AdjointR2) = (x.depth == y.depth) && (x.inverse == y.inverse)

function cost(s::AdjointR2, n::Int=1)
  c = Cost(COMPLEX_MULT => n/2, COMPLEX_DIV => n/2)
  if s.inverse
    c[COMPLEX_SQRT] = n
  end
  c
end

radix(x::AdjointR2) = [2] # TODO: does this make sense?
transform_length(x::AdjointR2) = [2] # TODO: does this make sense?
dual(x::AdjointR2) = x
Base.repeat(x::AdjointR2; inner::Int=1, outer::Int=1) = AdjointR2(inner * s.depth, s.inverse)
prefer_inplace(::AdjointR2) = true

function LinearAlgebra.mul!(dst::AbstractVector, x::AdjointR2, src::AbstractVector; mode::FFTMode=forward)
  # TODO: should `mode` influence this? how?

  n = length(src)
  d = x.depth
  if x.inverse
    @simd for i in 1:2d:n
      @simd for j in 0:d-1
        loc = i + j .+ (0:d)
        y0, y1 = src[loc]
        x0 = sqrt(y0 * y1)
        x1 = sqrt(y0 / y1)
        dst[loc] = [x0, x1]
      end
    end
  else
    @simd for i in 1:2d:n
      @simd for j in 0:d-1
        loc = i + j .+ (0:d)
        x0, x1 = src[loc]
        y0 = x0 * x1
        y1 = x0 / x1
        dst[loc] = [y0, y1]
      end
    end
  end
  dst
end

Base.:(*)(x::AdjointR2, y::AbstractVector) = mul!(similar(y), x, y)

Base.inv(x::AdjointR2) = AdjointR2(x.depth, !x.inverse)

