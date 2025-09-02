
struct Reorder
  perm :: Permutation

  function Reorder(perm::Permutation)
    data = perm.data
    reduced = false

    # normalize length by removing cycles
    for (p, pow) in factor(length(data))
      while pow > 0
        x = copy(reshape(data, :, p))
        q = size(x, 1)
        for i in 1:size(x, 2)
          x[:,i] .-= (i-1) * q
        end
        if all(x .== x[:,1])
          data = x[:,1]; reduced = true
          pow -= 1
        else
          break
        end
      end
    end

    if reduced
      perm = Permutation(data)
    end

    new(perm)
  end
end

Base.:(==)(x::Reorder, y::Reorder) = (x.perm == y.perm)

Base.show(io::IO, r::Reorder) = print(io, "Reorder($(r.perm.data .- 1))")

function Reorder(perm:: Vector{Int})
  if perm isa Vector
    perm = Permutation(perm .+ (1 - minimum(perm)))
  end
  Reorder(perm)
end

function extend_perm(perm::Vector{Int}, n::Int)
  p = length(perm)
  (q, r) = divrem(n, p, RoundDown)
  @assert r == 0
  new_perm = Array{Int}(undef, n)
  for i in 0:p:n-1
    new_perm[i .+ (1:p)] .= i .+ perm
  end
  new_perm
end

extend_perm(perm::Permutation, n::Int) = Permutation(extend_perm(perm.data, n))

function permutation(r::Reorder, n::Int=length(r.perm); inv::Bool=false)
  perm = r.perm
  inv && (perm = Base.inv(perm))
  perm = perm.data
  length(perm) == n ? perm : extend_perm(perm, n)
end

prefer_inplace(::Reorder) = false
cost(r::Reorder, n::Int=1) = Cost() # not currently modeling costs associated with reordering
radix(::Reorder) = Int[]
transform_length(::Reorder) = 1 # No inherent length; length comes from butterflies
dual(r::Reorder) = Reorder(inv(r.perm))

output_order(r::Reorder, in_order::AbstractVector=1:length(r.perm)) = permute_cyclic(r.perm, in_order)
input_order(r::Reorder, out_order::AbstractVector=1:length(r.perm)) = permute_cyclic(inv(r.perm), out_order)

function Base.repeat(p::Permutation; inner::Int=1, outer::Int=1)
  if inner == 1 && outer == 1
    return p
  end

  p = p.data
  if inner != 1
    q = reshape(1:inner*length(p), inner, :)
    p = (vec(q[:, p]))
  end

  if outer != 1
    p = extend_perm(p, length(p) * outer)
  end

  Permutation(p)
end

function Base.repeat(r::Reorder; inner::Int=1, outer::Int=1)
  p = r.perm.data
  q = reshape(1:inner*length(p), inner, :)
  Reorder(Permutation(vec(q[:, p])))
end

function LinearAlgebra.mul!(dst::AbstractVector, r::Reorder, src::AbstractVector; mode::FFTMode=forward)
  permute_cyclic!(dst, r.perm, src)
end

Base.:(*)(r::Reorder, x::AbstractVector) = mul!(similar(x), r, x)
Base.:(*)(r1::Reorder, r2::Reorder) = Reorder(permute_cyclic(r1.perm, r2.perm))
Base.one(::Type{Reorder}) = Reorder(Permutation(1))
Base.inv(r::Reorder) = r

function permute_cyclic!(dst::AbstractVector, permutation::Permutation, src::AbstractVector)
  perm = permutation.data

  @boundscheck @assert length(dst) == length(src)

  n = length(src); p = length(perm); q = n ÷ p
  @boundscheck if n != p * q
    error("Cannot reorder $n elements with reordering of length $p")
  end

  if Base.mightalias(src, dst)
    perm_src = reshape(src, p, q)
    perm_dst = reshape(dst, p, q)
    perm_dst .= perm_src[perm, :]
  else
    for i in 0:p:n-1
      @simd for j in 1:p
        @inbounds dst[i+j] = src[i+perm[j]]
      end
    end
  end

  dst
end

function permute_cyclic(permutation::Permutation, x::AbstractVector)
  permute_cyclic!(similar(x), permutation, x)
end

function permute_cyclic(p1::Permutation, p2::Permutation)
  # compute p1 * p2, but with each extended cyclically
  p = length(p1); q = length(p2)
  n = lcm(p, q)

  new_perm = Array{Int}(undef, n)
  for i in 1:p:n
    j = i + p - 1
    new_perm[i:j] .= (i:j)[p1.data]
  end
  for i in 1:q:n
    j = i + q - 1
    new_perm[i:j] .= new_perm[i:j][p2.data]
  end

  Permutation(new_perm)
end

function fft_rewrite_merge_reorders(r1::Reorder, r2::Reorder)
  Reorder(permute_cyclic(r1.perm, r2.perm))
end

function fft_rewrite_delete_identities(r::Reorder)
  isone(r.perm) ? [] : false
end

function fft_rewrite_delete_initial_reorder(stages::Vector, pos::UnitRange)
  if pos[1] == 1 && all(x -> x isa Reorder, stages[pos])
    splice!(stages, pos)
    true
  else
    false
  end
end

function fft_rewrite_delete_final_reorder(stages::Vector, pos::UnitRange)
  if pos[end] == length(stages) && all(x -> x isa Reorder, stages[pos])
    splice!(stages, pos)
    true
  else
    false
  end
end
