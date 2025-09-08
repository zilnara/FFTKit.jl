
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

# Simplified form of the M,P,Q,O twiddle gen from [Qureshi & Gustafsson]
# TODO: forms of the Twiddle type that can track the intent behind the construction
# (i.e., the algorithm used to generate them and parameters for the algorithm)
function twiddle_gen(p::Vector{Int}, Q::Int, O::Int)
  P_perm = digit_rev_perm(p)

  indexes = Int[]

  N = length(p)
  counters = zeros(Int,N)
  counter_stride = cumprod([1; p])[1:N]

  P = prod(p)
  for i in 0:P-1
    i_perm = sum(counters .* counter_stride)

    carry = 1
    for i in N:-1:1
      counters[i] += carry
      if counters[i] >= p[i]
        counters[i] = 0
        carry = 1
      else
        carry = 0
      end
    end

    for j in 0:Q-1
      for k in 0:O-1
        push!(indexes, j * i_perm)
      end
    end
  end

  Twiddle(P * Q, indexes)
end

# this should work for more general cases than the M,P,Q,O method above, I think. It's similar, but
# the digits are not assumed to be contiguous OR in order.
# 
# for example, for the test transform PLAN_256PT_MDC_R4, the twiddle stages are given by:
#   twiddle_gen_v2([2,2,2,2,2,2,2,2], 7:8, 1:6, [1,6,5,4,3,2,8,7])
#   twiddle_gen_v2([2,2,2,2,2,2,2,2], 5:6, 1:4, [1,7,8,4,3,2,6,5])
#   twiddle_gen_v2([2,2,2,2,2,2,2,2], 3:4, 1:2, [1,7,8,5,6,2,4,3])
#
# These can be derived by either computing the bit order based on the MDC permutations as above, or
# equivalently by describing the subsets of bit positions relative to the current bit order, as follows:
#   twiddle_gen_v2([2,2,2,2,2,2,2,2], [8,7], [1,6,5,4,3,2])
#   twiddle_gen_v2([2,2,2,2,2,2,2,2], [8,7], [1,6,5,4])
#   twiddle_gen_v2([2,2,2,2,2,2,2,2], [8,7], [1,6])
#
# I find the first way more natural, but it may optimize better in hardware to remap it to the second
# (which can be done in a simple automated way; just apply the inverse of the permutation to the P and
# Q digit indexes). In the radix-2^n case especially, it may more reliably infer carry chains across
# the whole counter structure.
#
# I do not know whether this does all the correct bookkeeping for transforms where the reordering
# changes the radix list, but transforms like that are not typical for MDC hardware, and I'm already
# fairly confident in the mixed-radix handling for non-reordering cases (as typical in SDF/MDF transforms)
#
# This function should be pretty straightforward to map to hardware, I think. To generalize to
# multiple lanes there are a lot of possible approaches but even a fairly naive implementation will
# probably generate acceptably-small logic, especially for radix-2^n cases.
function twiddle_gen_v2(radix::AbstractVector{Int}, p_digits::AbstractVector{Int}, q_digits::AbstractVector{Int}, digit_order::AbstractVector{Int}=1:length(radix))
  N = length(radix)
  n = prod(radix)

  counters = zeros(Int,N)
  p_stride = cumprod([1; radix[p_digits]])[1:end-1]
  q_stride = cumprod([1; radix[q_digits]])[1:end-1]

  indexes = Int[]

  for _ in 1:n
    p = sum(counters[p_digits] .* p_stride)
    q = sum(counters[q_digits] .* q_stride)
    push!(indexes, p*q)

    # TODO: consider rewriting this to use little-endian order with carries forward
    carry = 1
    for i in reverse(digit_order)
      counters[i] += carry
      if counters[i] >= radix[i]
        counters[i] = 0
        carry = 1
      else
        carry = 0
      end
    end
  end

  P = prod(radix[p_digits])
  Q = prod(radix[q_digits])
  Twiddle(P * Q, indexes)
end

# there are more straightforward simplifications that can be applied to twiddle_gen_v2 parameters. This function calculates some:
#   * discharges the "digit_order" by reordering the other arrays
#   * coalesces reverse-consecutive digits in P and Q
#   * coalesces any consecutive digits that are not referenced by either P or Q
#
# Under these simplifications, the twiddles from PLAN_256PT_MDC_R4 become:
#     twiddle_gen_v2([2,32,4], [3], [1,2])
#     twiddle_gen_v2([2,4,8,4], [4], [1,3])
#     twiddle_gen_v2([2,16,2,4], [4], [1,3])
# Which can each be implemented with a very simple tapped counter.
#
# I don't know if this is enough to achieve a "normal form" but it might be.
function simplify_twiddlegen_spec(radix::AbstractVector{Int}, p_digits::AbstractVector{Int}, q_digits::AbstractVector{Int}, digit_order::AbstractVector{Int}=Int[])
  # fold digit ordering into spec
  if !isempty(digit_order)
    radix    = radix[digit_order]

    perm = invperm(digit_order)
    p_digits = perm[p_digits]
    q_digits = perm[q_digits]
  else
    radix = copy(radix)
    p_digits = copy(p_digits)
    q_digits = copy(q_digits)
  end

  # coalesce reverse-consecutive P digits
  function coalesce(segment)
    lo = minimum(segment)

    splice!(radix, segment, [prod(radix[segment])])
    p_digits[p_digits .> lo] .-= 1
    q_digits[q_digits .> lo] .-= 1

    unique!(p_digits)
    unique!(q_digits)
  end

  function coalesce_digits(digits)
    while length(digits) > 1
      i = findfirst(diff(digits) .== -1)
      if isnothing(i)
        break
      end

      lo = digits[i+1]
      hi = digits[i]
      coalesce(lo:hi)
    end
  end

  # coalesce digits used consecutively in P or Q
  coalesce_digits(p_digits)
  coalesce_digits(q_digits)

  # coalesce consecutive unused digits
  changed = true
  while changed
    changed = false

    prev_unused = false
    for i in 1:length(radix)
      unused = !(i in p_digits) && !(i in q_digits)
      if prev_unused && unused
        coalesce(i-1:i)
        changed = true
        break
      end

      prev_unused = unused
    end
  end

  radix, p_digits, q_digits
end

# TODO: another version that constructs twiddles from the "triangular matrix" representation in [Garrido] for length-2^n transforms
#       also can we generalize that to true mixed radix? I'm sure the constraints on rotator movement get screwy, but
#       it'd be interesting to try anyway.
# Is there actually any benefit to using a representation with that much freedom though? They never really give examples of
# situations where a useful transform is representable with the "triangular matrix" form but not by the b-tree form.

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

