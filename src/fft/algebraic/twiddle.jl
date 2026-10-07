
struct StandardTwiddle
  p::Vector{Int}
  q::Int
  o::Int
  conj::Bool

  function StandardTwiddle(p::Vector{Int}, q::Int, o::Int, conj::Bool=false)
    @assert all(p .>= 0) && q >= 0 && o >= 0
    P = prod(p)
    if P <= 1 || q <= 1
      # extremely trivial case
      new(Int[], 0, 0)
    elseif P * q <= 2 # not necessarily trivial but self inverse
      new(p, q, o, false)
    else
      new(p, q, o, conj)
    end
  end
end

twiddle_order(tw::StandardTwiddle) = prod(tw.p) * tw.q
twiddle_inner_repeat_lenth(tw::StandardTwiddle) = tw.o
function twiddle_indexes(tw::StandardTwiddle)
  P = prod(tw.p); Q = tw.q; O = tw.o;
  P_perm = inv(permute_mixed_digits(tw.p, length(tw.p):-1:1))

  conj = tw.conj
  order = P * Q

  tw = Array{Int}(undef,O,Q,P)
  for i in 0:Q-1, j in 0:P-1
    k = i * (P_perm[j+1]-1)
    if conj && k != 0
      k = - k
    end
    # TODO: mod necessary? can it be avoided by smarter construction?
    tw[:,i+1, j+1] .= mod(k, order)
  end

  vec(tw)
end

# This method of specifying the sequence of twiddles for a stage is based on [Qureshi 2011], section III
# ("Twiddle factor index generation). The scheme described there is somewhat vague, and only covers power
# of two transform sizes, but that entire paper (including this section) can be generalized by considering
# the B-tree node values to be ordered lists of primitive radixes (or in many cases, the products of such
# lists, by analogy considering the version in the paper to be the sums of the base-2 logarithms).
#
# The full structure including the ordering of factors is only actually relevant in the "p" term, corresponding
# to the order of the radixes in the decomposition of the left child of the focused B-tree node (these are
# the radix groups that will be digit-reversed, which in the original paper is not sensitive to this order
# but when generalizing to mixed-radix transforms, becomes sensitive to it).
function StandardTwiddle(m::Vector{Int}, p::Vector{Int}, q::Vector{Int}, o::Vector{Int})
  # ignore M, since it just determines how many times the twiddles repeat.
  # our treatment of twiddle generators implicitly repeats infinitely.
  StandardTwiddle(p, prod(q), prod(o))
end

function Base.show(io::IO, tw::StandardTwiddle)
  if tw.conj
    print(io, "inv(Twiddle($(tw.p), $(tw.q), $(tw.o)))")
  else
    print(io, "Twiddle($(tw.p), $(tw.q), $(tw.o))")
  end
end

Base.repeat(tw::StandardTwiddle; inner::Int=1, outer::Int=1) = Twiddle(tw.p, tw.q, inner * tw.o)
Base.inv(tw::StandardTwiddle) = StandardTwiddle(tw.p, tw.q, tw.o, !tw.conj)

# This works for more general cases than the StandardTwiddle type. It's similar, but the digits are not
# assumed to be contiguous OR in order.
# 
# for example, for the test transform PLAN_256PT_MDC_R4, the twiddle stages are given by:
#   ExtendedTwiddle([2,2,2,2,2,2,2,2], 7:8, 1:6, [1,6,5,4,3,2,8,7])
#   ExtendedTwiddle([2,2,2,2,2,2,2,2], 5:6, 1:4, [1,7,8,4,3,2,6,5])
#   ExtendedTwiddle([2,2,2,2,2,2,2,2], 3:4, 1:2, [1,7,8,5,6,2,4,3])
#
# These can be derived by either computing the bit order based on the MDC permutations as above, or
# equivalently by describing the subsets of bit positions relative to the current bit order, as follows:
#   ExtendedTwiddle([2,2,2,2,2,2,2,2], [8,7], [1,6,5,4,3,2])
#   ExtendedTwiddle([2,2,2,2,2,2,2,2], [8,7], [1,6,5,4])
#   ExtendedTwiddle([2,2,2,2,2,2,2,2], [8,7], [1,6])
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
# This twiddle generation scheme should be pretty straightforward to map to hardware, I think. To generalize to
# multiple lanes there are a lot of possible approaches but even a fairly naive implementation will
# probably generate acceptably-small logic, especially for radix-2^n cases.
struct ExtendedTwiddle
  radix::Vector{Int}
  p_digits::Vector{Int}
  q_digits::Vector{Int}
  conj::Bool

  function ExtendedTwiddle(radix::AbstractVector{Int}, p_digits::AbstractVector{Int}, q_digits::AbstractVector{Int}, conj::Bool=false; simplify::Bool=true)
    if simplify
      radix, p_digits, q_digits = simplify_extended_twiddle(radix, p_digits, q_digits)

      # TODO simplify more redundancies in the representation
      if all(isone, diff(sort(p_digits))) # P is contiguous (maybe also empty)
        if all(isone, diff(sort(q_digits))) # same for Q
          if isempty(p_digits) || isempty(q_digits) || minimum(q_digits) == 1 + maximum(p_digits)
            # representable as a "standard twiddle"
            radix = Vector{Int}(radix)
            p = radix[p_digits]
            q = prod(radix[q_digits])

            o_start = 1 
            if !isempty(p_digits)
              o_start = 1 + maximum(p_digits)
            end
            if !isempty(q_digits)
              o_start = 1 + maximum(q_digits)
            end
            o = prod(radix[o_start:end])

            return StandardTwiddle(p, q, o, conj)
          end
        end
      end
    end

    new(radix, p_digits, q_digits, conj)
  end
end

function twiddle_order(tw::ExtendedTwiddle)
  P = prod(tw.radix[tw.p_digits])
  Q = prod(tw.radix[tw.q_digits])
  P * Q
end

function twiddle_indexes(tw::ExtendedTwiddle)
  N = length(tw.radix)
  n = prod(tw.radix)

  counters = zeros(Int,N)
  p_stride = cumprod([1; tw.radix[tw.p_digits]])[1:end-1]
  q_stride = cumprod([1; tw.radix[tw.q_digits]])[1:end-1]

  indexes = Int[]
  order = twiddle_order(tw)

  for _ in 1:n
    p = sum(counters[tw.p_digits] .* p_stride)
    q = sum(counters[tw.q_digits] .* q_stride)
    k = p * q
    if tw.conj && k != 0
      k = order - k
    end
    push!(indexes, k)

    # TODO: consider rewriting this to use little-endian order with carries forward
    carry = 1
    for i in N:-1:1
      counters[i] += carry
      if counters[i] >= tw.radix[i]
        counters[i] = 0
        carry = 1
      else
        carry = 0
      end
    end
  end

  # TODO: mod necessary? can it be avoided by smarter construction?
  mod.(indexes, order)
end

function twiddle_inner_repeat_lenth(tw::ExtendedTwiddle)
  max_used = 0
  max_used = maximum(tw.p_digits, init=max_used)
  max_used = maximum(tw.q_digits, init=max_used)
  prod(tw.radix[max_used+1:end])
end

function Base.show(io::IO, tw::ExtendedTwiddle)
  if tw.conj
    print(io, "inv(Twiddle($(tw.radix), $(tw.p_digits), $(tw.q_digits)))")
  else
    print(io, "Twiddle($(tw.radix), $(tw.p_digits), $(tw.q_digits))")
  end
end

function Base.repeat(tw::ExtendedTwiddle; inner::Int=1, outer::Int=1)
  if inner == 0
    return tw
  end

  ExtendedTwiddle(vcat(tw.radix, [inner]), tw.p_digits, tw.q_digits, tw.conj)
end

Base.inv(tw::ExtendedTwiddle) = ExtendedTwiddle(tw.radix, tw.p_digits, tw.q_digits, !tw.conj)

function twiddle_gen_v2(radix::AbstractVector{Int}, p_digits::AbstractVector{Int}, q_digits::AbstractVector{Int}, digit_order::AbstractVector{Int}=1:length(radix))
  radix, p_digits, q_digits = simplify_extended_twiddle(radix, p_digits, q_digits, digit_order)
  ExtendedTwiddle(radix, p_digits, q_digits)
end

# there are more straightforward simplifications that can be applied to ExtendedTwiddle parameters. This function calculates some:
#   * discharges the "digit_order", if given, by reordering the other arrays
#   * coalesces reverse-consecutive digits in P and Q
#   * coalesces any consecutive digits that are not referenced by either P or Q
#   * drop all unused leading digits, they generate outer repeats of the index list
#
# Under these simplifications, the twiddles from PLAN_256PT_MDC_R4 become:
#     ExtendedTwiddle([2,32,4], [3], [1,2])
#     ExtendedTwiddle([2,4,8,4], [4], [1,3])
#     ExtendedTwiddle([2,16,2,4], [4], [1,3])
# Which can each be implemented with a very simple tapped counter.
#
# I don't know if this is enough to achieve a "normal form" but it might be.
function simplify_extended_twiddle(radix::AbstractVector{Int}, p_digits::AbstractVector{Int}, q_digits::AbstractVector{Int}, digit_order::AbstractVector{Int}=Int[])
  # handle a few especially trivial cases
  if prod(radix[p_digits]) <= 1 || prod(radix[q_digits]) <= 1
    return Int[], Int[], Int[]
  end

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
  function coalesce(segment::UnitRange{Int})
    to_drop = length(segment) - 1
    if to_drop < 1
      return
    end

    lo = minimum(segment)

    splice!(radix, segment, [prod(radix[segment])])
    for i in 1:length(p_digits)
      if p_digits[i] > lo
        p_digits[i] = max(lo, p_digits[i] - to_drop)
      end
    end
    for i in 1:length(q_digits)
      if q_digits[i] > lo
        q_digits[i] = max(lo, q_digits[i] - to_drop)
      end
    end

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

  # drop trivial radixes (radix-1 is stupid but in some automatic processing it can occur)
  while true
    i = findfirst(radix .== 1)
    if isnothing(i)
      break
    end

    splice!(radix, i:i, Int[])
    p_digits = p_digits[p_digits .!= i]
    p_digits[p_digits .> i] .-= 1
    q_digits = q_digits[q_digits .!= i]
    q_digits[q_digits .> i] .-= 1
  end

  # delete unused leading digits
  min_used = length(radix) + 1
  min_used = minimum(p_digits, init=min_used)
  min_used = minimum(q_digits, init=min_used)
  n_unused = min_used - 1
  if n_unused > 0
    indexes = twiddle_indexes(ExtendedTwiddle(radix, p_digits, q_digits, simplify=false))
    @info "deleting leading digits" n_unused radix=repr(radix) p_digits=repr(p_digits) q_digits=repr(q_digits) indexes=repr(indexes)

    radix = radix[min_used:end]
    p_digits .-= n_unused
    q_digits .-= n_unused

    indexes = twiddle_indexes(ExtendedTwiddle(radix, p_digits, q_digits, simplify=false))
    @info "post deletion" radix=repr(radix) p_digits=repr(p_digits) q_digits=repr(q_digits) indexes=repr(indexes)
  end

  radix, p_digits, q_digits
end

struct ArbitraryTwiddle
  order       :: Int
  indexes     :: Vector{Int}
  # TODO: support inner repeat count as a field?

  function ArbitraryTwiddle(order::Int, indexes::Vector{Int})
    # normalize order and indexes by removing common factors
    c = gcd(order, gcd(indexes))
    if c > 1
      order ÷= c
      indexes .÷= c
    end

    # normalize length by removing cycles
    indexes = remove_outer_cycles(mod.(indexes, order))
    new(order, indexes)
  end
end

twiddle_order(tw::ArbitraryTwiddle) = tw.order
twiddle_indexes(tw::ArbitraryTwiddle) = tw.indexes

function ArbitraryTwiddle(indexes::AbstractVector{<:Integer})
  if !(indexes isa Vector{Int})
    indexes = convert(Vector{Int}, indexes)
  end
  ArbitraryTwiddle(length(indexes), indexes)
end

function ArbitraryTwiddle(tw::Union{StandardTwiddle, ExtendedTwiddle})
  ArbitraryTwiddle(twiddle_order(tw), twiddle_indexes(tw))
end

function Base.show(io::IO, tw::ArbitraryTwiddle)
  if tw.order == length(tw.indexes)
    print(io, "Twiddle($(tw.indexes))")
  else
    print(io, "Twiddle($(tw.order), $(tw.indexes))")
  end
end

function inner_repeat_length(xs::AbstractVector{Int})
  for n in reverse(divisors(length(xs)))
    if n == 1
      break
    end

    grouped = reshape(xs, n, :)
    if all(grouped .== grouped[1:1, :])
      return n
    end
  end

  return 1
end

function outer_repeat_length(xs::AbstractVector{Int})
  for n in reverse(divisors(length(xs)))
    if n == 1
      break
    end

    grouped = reshape(xs, :, n)
    if all(grouped .== grouped[:, 1:1])
      return n
    end
  end

  return 1
end

function remove_outer_cycles(xs::AbstractVector{Int})
  for n in reverse(divisors(length(xs)))
    if n == 1
      break
    end

    grouped = reshape(xs, :, n)
    if all(grouped .== grouped[:, 1:1])
      return grouped[:, 1]
    end
  end

  xs
end

function twiddle_inner_repeat_lenth(tw::ArbitraryTwiddle)
  inner_repeat_length(tw.indexes)
end

Base.repeat(tw::ArbitraryTwiddle; inner::Int=1, outer::Int=1) = Twiddle(tw.order, repeat(tw.indexes; inner))
Base.inv(tw::ArbitraryTwiddle) = Twiddle(twiddle_order(tw), twiddle_order(tw) .- twiddle_indexes(tw))

const Twiddle = Union{StandardTwiddle, ExtendedTwiddle, ArbitraryTwiddle}

Twiddle(m::Vector{Int}, p::Vector{Int}, q::Vector{Int}, o::Vector{Int}) = StandardTwiddle(m, p, q, o)
Twiddle(p::Vector{Int}, q::Int, o::Int) = StandardTwiddle(p, q, o)
Twiddle(radix::Vector{Int}, p_digits::Vector{Int}, q_digits::Vector{Int}) = ExtendedTwiddle(radix, p_digits, q_digits)
Twiddle(order::Int, indexes::Vector{Int}) = ArbitraryTwiddle(order, indexes)
Twiddle(indexes::Vector{Int}) = ArbitraryTwiddle(indexes) # TODO attempt to recognize standard twiddles from index list

Base.:(==)(x::Twiddle, y::Twiddle) = (twiddle_order(x) == twiddle_order(y)) && _eq_cyclic(twiddle_indexes(x), twiddle_indexes(y))

function _eq_cyclic(x::AbstractVector{Int}, y::AbstractVector{Int})
  nx = length(x); ny = length(y)
  if nx == ny
    return x == y
  end

  if nx == 0
    return iszero(y)
  elseif ny == 0
    return iszero(x)
  end

  j = 1; k = 1
  for i in 1:max(nx, ny)
    if x[j] != y[k]
      return false
    end

    j += 1; k += 1;
    if j > nx
      j -= nx
    end
    if k > ny
      k -= ny
    end
  end
  true
end

function twiddle_indexes(tw::Twiddle, n::Int, order::Int=tw.order)
  tw_order = twiddle_order(tw)
  tw_indexes = twiddle_indexes(tw)

  out = Array{Int}(undef, n)
  m = length(tw_indexes)

  if order == tw_order
    k = 1
  else
    @assert mod(order, tw_order) == 0
    k = order ÷ tw_order
  end

  for i in 1:n
    if i <= m
      out[i] = k * tw_indexes[i]
    else
      out[i] = out[i - m]
    end
  end
  out
end

function rotations(tw::Twiddle, n::Union{Missing,Int} = missing)
  tw_order = twiddle_order(tw)
  tw_indexes = twiddle_indexes(tw)

  if ismissing(n)
    n = length(tw_indexes)
  end

  out = Array{Rotation}(undef, n)
  m = length(tw_indexes)
  for i in 1:n
    if i <= m
      out[i] = Rotation(tw_indexes[i] // tw_order)
    else
      out[i] = out[i - m]
    end
  end
  out
end

function cost(t::Twiddle, n::Int=1)
  # TODO: differentiate between parallel and serial stages?
  order = twiddle_order(t)
  if order <= 2
    Cost()
  elseif order == 4
    Cost(TRIVIAL_CMULT => n)
  else
    Cost(COMPLEX_MULT => n)
  end
end

prefer_inplace(::Twiddle) = true
radix(::Twiddle) = Int[]
transform_length(::Twiddle) = 1 # No inherent length; length comes from butterflies
dual(tw::Twiddle) = tw

output_order(r::Twiddle, in_order::AbstractVector=1:length(twiddle_indexes(tw))) = in_order
input_order(r::Twiddle, out_order::AbstractVector=1:length(twiddle_indexes(tw))) = out_order

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

# TODO: another version that constructs twiddles from the "triangular matrix" representation in [Garrido] for length-2^n transforms
#       also can we generalize that to true mixed radix? I'm sure the constraints on rotator movement get screwy, but
#       it'd be interesting to try anyway.
# Is there actually any benefit to using a representation with that much freedom though? They never really give examples of
# situations where a useful transform is representable with the "triangular matrix" form but not by the b-tree form.

function permute_cyclic(permutation::Permutation, tw::Twiddle)
  if isone(permutation)
    return tw
  end

  if length(permutation) < twiddle_inner_repeat_lenth(tw)
    return tw
  end

  # TODO try to identify other permutation structures that can preserve structured twiddle generation, especially ones
  #      that reorder digits and can be represented by reordering digits in an ExtendedTwiddle structure
  if !(tw isa ArbitraryTwiddle)
    tw = ArbitraryTwiddle(tw)
  end

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
  iszero(twiddle_indexes(tw)) ? [] : false
end

function fft_rewrite_merge_twiddles(tw1::Twiddle, tw2::Twiddle)
  # TODO work out a version that preserves standard transforms
  if !(tw1 isa ArbitraryTwiddle)
    tw1 = ArbitraryTwiddle(tw1)
  end
  if !(tw2 isa ArbitraryTwiddle)
    tw2 = ArbitraryTwiddle(tw2)
  end

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

# generate all ordered lists of smaller positive integers such that prod(_) == n
function ordered_decompositions(n::Int)
  decomps = Vector{Int}[Int[n]]
  for p in reverse(divisors(n))
    if p == 1
      continue
    end

    q = n ÷ p
    if q > 1
      for decomp in ordered_decompositions(q)
        pushfirst!(decomp, p)
        push!(decomps, decomp)
      end
    end
  end

  decomps
end

# like ordered_decompositions but all terms are prime
function ordered_prime_decompositions(n::Int)
  decomps = Vector{Int}[Int[]]
  for (p, n) in eachfactor(n)
    for rep in 1:n
      new_decomps = Vector{Int}[]

      for decomp in decomps
        last_p = something(findlast(decomp .== p), 0)
        for i in reverse(1+last_p : 1+length(decomp))
          new_decomp = copy(decomp)
          insert!(new_decomp, i, p)
          push!(new_decomps, new_decomp)
        end
      end

      decomps = new_decomps
    end
  end

  decomps
end

function powerset(xs::AbstractVector{T}) where {T}
  result = Vector{T}[T[]]
  for x in xs, j in eachindex(result)
    push!(result, [result[j]; x])
  end
  result
end

function try_simplify_twiddle(tw::Twiddle)
  # TODO: support inverted twiddle stages... is there an easy way to diagnose them?

  order = twiddle_order(tw)
  indexes = twiddle_indexes(tw)
  if iszero(indexes)
    return StandardTwiddle(Int[], 0, 0)
  end

  o = inner_repeat_length(indexes)
  indexes = indexes[1:o:end]
  tw = ArbitraryTwiddle(order, indexes)

  # relatively simple nearly-brute-force
  # TODO this doesn't scale well to very large transforms, try to bound the search better
  # maybe first iterate over factorizations of the order, then over the rest of the radixes?
  # using only prime decompositions greatly prunes the search space, but using non-prime ones and ordering the
  # search by length (as done here) potentially finds actually-likely cases faster.
  for radixes in sort(ordered_decompositions(length(indexes)), by=length)
    n = length(radixes)

    for p_set in powerset(1:n)
      if isempty(p_set)
        # already excluded this case by triviality check
        continue
      end

      P = prod(radixes[p_set])
      if P >= order
        # infeasible, no Q can make it equal the order except possibly Q=1,
        # which would be a trivial transform (already excluded)
        continue
      end

      for q_set in powerset(setdiff(1:n, p_set))
        if isempty(q_set)
          # already excluded this case by triviality check
          continue
        end

        if !(1 in p_set) && !(1 in q_set)
          # outer repetitions are excluded by definition of twiddle generator types
          continue
        end

        if !(n in p_set) && !(n in q_set)
          # inner repetitions are excluded by factoring out 'o' above
          continue
        end

        Q = prod(radixes[q_set])
        if P * Q != order
          continue
        end

        for p_perm in PermGen(length(p_set))
          for q_perm in PermGen(length(q_set))
            p_digits = p_set[p_perm.data]
            q_digits = q_set[q_perm.data]

            match_tw = ExtendedTwiddle(radixes, p_digits, q_digits)
            if indexes == twiddle_indexes(match_tw)
              return repeat(match_tw, inner=o)
            end
          end
        end
      end
    end
  end

  return nothing
end

function simplify_twiddle(tw::Twiddle)
  maybe_subst = try_simplify_twiddle(tw)
  if isnothing(maybe_subst)
    tw
  else
    maybe_subst
  end
end
