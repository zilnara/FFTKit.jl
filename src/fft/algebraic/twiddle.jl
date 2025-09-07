
struct Twiddle
  order       :: Int
  indexes     :: Vector{Int}
  # TODO: support inner repeat count as a field?

  function Twiddle(order::Int, indexes::Vector{Int})
    # normalize order and indexes by removing common factors
    c = gcd(order, gcd(indexes))
    if c > 1
      order /= c
      indexes ./= c
    end

    # normalize length by removing cycles
    for (p, pow) in factor(length(indexes))
      while pow > 0
        x = reshape(indexes, :, p)
        if all(x .== x[:,1])
          indexes = x[:,1]
          pow -= 1
        else
          break
        end
      end
    end

    new(order, mod.(indexes, order))
  end
end

function Twiddle(indexes::AbstractVector{<:Integer})
  if !(indexes isa Vector{Int})
    indexes = convert(Vector{Int}, indexes)
  end
  Twiddle(length(indexes), indexes)
end

function Twiddle(m::Vector{Int}, p::Vector{Int}, q::Vector{Int}, o::Vector{Int})
  # ignore M, since it just determines how many times the twiddles repeat.
  # The "Twiddle" stage implicitly repeats infinitely.
  P = prod(p); Q = prod(q); O = prod(o)
  P_perm = inv(permute_mixed_digits(p, length(p):-1:1))

  tw = Array{Int}(undef,O,Q,P)
  for i in 0:Q-1, j in 0:P-1
    tw[:,i+1, j+1] .= i * (P_perm[j+1]-1)
  end

  Twiddle(P * Q, vec(tw))
end

Base.:(==)(x::Twiddle, y::Twiddle) = (x.order == y.order) && (x.indexes == y.indexes)

function Base.show(io::IO, tw::Twiddle)
  if tw.order == length(tw.indexes)
    print(io, "Twiddle($(tw.indexes))")
  else
    print(io, "Twiddle($(tw.order), $(tw.indexes))")
  end
end

function twiddle_indexes(tw::Twiddle, n::Int = length(tw.indexes), order::Int=tw.order)
  out = Array{Int}(undef, n)
  m = length(tw.indexes)

  if order == tw.order
    k = 1
  else
    @assert mod(order, tw.order) == 0
    k = order ÷ tw.order
  end

  for i in 1:n
    if i <= m
      out[i] = k * tw.indexes[i]
    else
      out[i] = out[i - m]
    end
  end
  out
end

function rotations(tw::Twiddle, n::Int = length(tw.indexes))
  out = Array{Rotation}(undef, n)
  m = length(tw.indexes)
  for i in 1:n
    if i <= m
      out[i] = Rotation(tw.indexes[i] // tw.order)
    else
      out[i] = out[i - m]
    end
  end
  out
end

function cost(t::Twiddle, n::Int=1)
  # TODO: differentiate between parallel and serial stages?
  if t.order <= 2
    Cost()
  elseif t.order == 4
    Cost(TRIVIAL_CMULT => n)
  else
    Cost(COMPLEX_MULT => n)
  end
end

prefer_inplace(::Twiddle) = true
radix(::Twiddle) = Int[]
transform_length(::Twiddle) = 1 # No inherent length; length comes from butterflies
dual(tw::Twiddle) = tw
Base.repeat(tw::Twiddle; inner::Int=1, outer::Int=1) = Twiddle(tw.order, repeat(tw.indexes; inner))

output_order(r::Twiddle, in_order::AbstractVector=1:length(tw.indexes)) = in_order
input_order(r::Twiddle, out_order::AbstractVector=1:length(tw.indexes)) = out_order

function LinearAlgebra.mul!(dst::AbstractVector, tw::Twiddle, src::AbstractVector; mode::FFTMode=forward)
  # TODO: support pre-calculating twiddle values
  r = rotations(tw)
  n = length(r)
  for i in eachindex(src)
    j = 1+mod(i-1, n)
    rotation = r[j]
    if mode == forward
      rotation = conj(rotation)
    end
    dst[i] = src[i] * rotation
  end

  dst
end

Base.inv(tw::Twiddle) = Twiddle(tw.order, tw.order .- tw.indexes)

Base.:(*)(r::Reorder, tw::Twiddle) = permute_cyclic(r.perm, tw)

function twiddle_schedules(radix_tree::BTree{Int})
  stages = NTuple{4, Vector{Int}}[]

  function f(
    l::BTree{Int},
    r::BTree{Int};
    context_left::Union{Nothing, BTree{Int}},
    context_right::Union{Nothing, BTree{Int}})

    M = isnothing(context_left) ? Int[] : collect(context_left)
    P = collect(l)
    Q = collect(r)
    O = isnothing(context_right) ? Int[] : collect(context_right)
    push!(stages, (M, P, Q, O))
  end

  foreach_node_with_context(f, radix_tree)

  stage_number(s) = length(s[1]) + length(s[2])
  stage_depth(s) = prod(s[3]) * prod(s[4])
  Dict(sort([stage_depth(s) => s for s in stages], rev=true))
end

function permute_cyclic(permutation::Permutation, tw::Twiddle)
  p = length(tw.indexes)
  n = lcm(length(permutation), p)
  q = n ÷ p
  indexes = repeat(tw.indexes, outer=q)
  Twiddle(tw.order, permute_cyclic(permutation, indexes))
end

function fft_rewrite_push_reorders_right(r::Reorder, tw::Twiddle)
  tw = permute_cyclic(inv(r.perm), tw)
  [tw, r]
end

function fft_rewrite_delete_identities(tw::Twiddle)
  iszero(tw.indexes) ? [] : false
end

function fft_rewrite_merge_twiddles(tw1::Twiddle, tw2::Twiddle)
  order_1 = tw1.order; indexes_1 = tw1.indexes
  order_2 = tw2.order; indexes_2 = tw2.indexes
  @info "orig" order_1 order_2 
  @show indexes_1 indexes_2

  order = lcm(order_1, order_2)
  if order_1 != order
    k = order ÷ order_1
    order_1 = order; indexes_1 = k * indexes_1
  end
  if order_2 != order
    k = order ÷ order_2
    order_2 = order; indexes_2 = k * indexes_2
  end
  @info "matched order" order order_1 order_2 
  @show indexes_1 indexes_2

  len_1 = length(indexes_1); len_2 = length(indexes_2)
  len = lcm(len_1, len_2)
  if len_1 != len
    k = len ÷ len_1
    indexes_1 = repeat(indexes_1, outer = k)
  end
  if len_2 != len
    k = len ÷ len_2
    indexes_2 = repeat(indexes_2, outer = k)
  end
  @info "matched length" order len len_1 order_1 len_2 order_2
  @show indexes_1 indexes_2 indexes_1 + indexes_2

  Twiddle(order, indexes_1 + indexes_2)
end

