
# TODO: document this from the perspective of bit/digit swaps, which give good intuition for overall MDC transform design.
# Generalizing to "swap digits N,M of radix R" is likely to be useful too. in the MDC case, it's that but with N or M fixed to 1.
# In the general case, it can be done with mixed radix transforms as well, which could potentially be useful for
# MDC transforms with composite lane counts such as 6.
struct MdcReorder
  depth :: Int
  radix :: Int
end

Base.length(r::MdcReorder) = r.depth * r.radix ^ 2
Base.inv(r::MdcReorder) = r # all mdc reorders are involutions

function permutation(r::MdcReorder)
  Reorder(mdc_permutation_general(r.radix, r.depth)).perm
end

function Base.repeat(r::MdcReorder; inner::Int=1, outer::Int=1)
  if inner == 1
    return r
  end

  # TODO can this be representad as another MdcReorder?
  # probably not without expanding the representation in some way
  repeat(Reorder(permutation(r)); inner, outer)
end

Base.show(io::IO, r::MdcReorder) = print(io, "MdcReorder($(r.depth), $(r.radix))")

struct ArbitraryReorder
  perm :: Permutation

  function ArbitraryReorder(perm::Permutation)
    data = perm.data
    reduced = false

    # normalize length by removing outer cycles.
    # That is, remove duplicate parallel copies of the same permutation. For example:
    #   reduce (1,2)(3,4)(5,6) to just (1,2)
    #   reduce (1)(2,3)(4)(5)(6,7)(8) to just (1)(2,3)(4)
    # and so on.
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

Base.length(r::ArbitraryReorder) = length(r.perm)
Base.inv(r::ArbitraryReorder) = ArbitraryReorder(inv(r.perm))

function permutation(r::ArbitraryReorder)
  r.perm
end

function Base.repeat(r::ArbitraryReorder; inner::Int=1, outer::Int=1)
  p = r.perm.data
  q = reshape(1:inner*length(p), inner, :)
  ArbitraryReorder(Permutation(vec(q[:, p])))
end

Base.show(io::IO, r::ArbitraryReorder) = print(io, "Reorder($(r.perm.data .- 1))")

const Reorder = Union{MdcReorder, ArbitraryReorder}

function Reorder(perm::Union{Permutation, Vector{Int}})
  if perm isa Vector
    perm = Permutation(perm .+ (1 - minimum(perm)))
  end
  ArbitraryReorder(perm)
end

Base.:(==)(x::Reorder, y::Reorder) = permutation(x) == permutation(y)

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

prefer_inplace(::Reorder) = false
cost(r::Reorder, n::Int=1) = Cost() # not currently modeling costs associated with reordering
radix(::Reorder) = Int[]
transform_length(::Reorder) = 1 # No inherent length; length comes from butterflies
dual(r::Reorder) = inv(r)

output_order(r::Reorder, in_order::AbstractVector=1:length(r)) = permute_cyclic(permutation(r), in_order)
input_order(r::Reorder, out_order::AbstractVector=1:length(r)) = permute_cyclic(inv(permutation(r)), out_order)

function permutation_vec(r::Reorder, n::Int=length(r.perm); inv::Bool=false)
  perm = permutation(r)
  inv && (perm = Base.inv(perm))
  perm = perm.data
  length(perm) == n ? perm : extend_perm(perm, n)
end

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

function LinearAlgebra.mul!(dst::AbstractVector, r::Reorder, src::AbstractVector; mode::FFTMode=forward)
  permute_cyclic!(dst, permutation(r), src)
end

Base.:(*)(r::Reorder, x::AbstractVector) = mul!(similar(x), r, x)
Base.:(*)(r1::Reorder, r2::Reorder) = Reorder(permute_cyclic(permutation(r1), permutation(r2)))
Base.one(::Type{Reorder}) = Reorder(Permutation(1))
Base.inv(r::Reorder) = invperm(r)
Base.invperm(r::Reorder) = Reorder(inv(r.perm))

# WIP: ways of modeling MDC commutator stages
# this one is very raw. Using the standard structure as shown in e.g. "A Survey on Pipelined FFT Hardware Architectures", models a single radix-2 reordering stage with potentially-different delay lengths (d1 is the top-right, d2 is the bottom-left), with explicit mux timing (mux1 is top mux, mux2 is bottom mux) and '0' is top path in a mux, '1' is bottom path in a mux.
#
# I probably won't do much with this any time soon, it's mostly a novelty for me. For my needs, I'm more interested in structures that can support natural input or output orders (MDF, mainly)
function mdc_permutation(d1::Int, d2::Int, mux1::AbstractVector{Bool}, mux2::AbstractVector{Bool} = .! mux1; compact::Bool=true)
  ram1 = Array{Union{Missing, Int}}(missing, d1); ram1_addr = 0
  ram2 = Array{Union{Missing, Int}}(missing, d2); ram2_addr = 0
  n = minimum(length, [mux1, mux2])
  output = Union{Missing,Int}[]
  for i in 0:n-1
    stage_in1   = 2i;                                     stage_in2   = 2i+1
    ram1_out    = ram1[1+ram1_addr];                      ram2_out    = ram2[1+ram2_addr]
    mux1_out    = mux1[1+i] == 0 ? stage_in1 : ram2_out;  mux2_out    = mux2[1+i] == 0 ? stage_in1 : ram2_out
    ram1_in     = mux1_out;                               ram2_in     = stage_in2
    stage_out1  = ram1_out;                               stage_out2  = mux2_out

    ram1[1+ram1_addr] = ram1_in;                          ram2[1+ram2_addr] = ram2_in
    ram1_addr   = mod(ram1_addr+1, d1);                   ram2_addr   = mod(ram2_addr+1, d2)

    append!(output, [stage_out1, stage_out2])
  end

  if compact
    output = collect(skipmissing(output))
  end
  output
end

# experiment: is there a natural 3-path generalization? (and from there, N path)
function mdc_r3_permutation(d::Int, mux1::AbstractVector{Int}, mux2::AbstractVector{Int}, mux3::AbstractVector{Int}; repeat::Int=1, compact::Bool=true)
  # TODO: is there a better generalization, perhaps using only length-d delays? or tapped length-2d delays?
  # I feel like the latter would work and generalize - in fact, i think it ends up just being a pretty natural reimagining of a generic SDF structure, with muxes in place of butterflies and butterflies applied in parallel
  ram = [
    Array{Union{Missing, Int}}(missing, 2d),
    Array{Union{Missing, Int}}(missing,  d),
    Array{Union{Missing, Int}}(missing,  d),
    Array{Union{Missing, Int}}(missing, 2d),
  ]
  addr = zeros(Int, length(ram))

  mux1 = mod.(mux1, 3)
  mux2 = mod.(mux2, 3)
  mux3 = mod.(mux3, 3)
  if repeat != 1
    mux1 = Base.repeat(mux1, outer=repeat)
    mux2 = Base.repeat(mux2, outer=repeat)
    mux3 = Base.repeat(mux3, outer=repeat)
  end

  n = minimum(length, [mux1, mux2, mux3])
  output = Union{Missing, Int}[]
  for i in 0:n-1
    stage_in = 3i .+ (0:2)

    ram_out = [ram[j][1+addr[j]] for j in 1:length(ram)]
    mux_out = [stage_in[1], ram_out[3], ram_out[4]][1 .+ [mux1[1+i], mux2[1+i], mux3[1+i]]]
    ram_in = [mux_out[1], mux_out[2], stage_in[2], stage_in[3]]
    stage_out = [ram_out[1], ram_out[2], mux_out[3]]

    for j in 1:length(ram)
      ram[j][1+addr[j]] = ram_in[j]
      addr[j] = mod(addr[j] + 1, length(ram[j]))
    end

    append!(output, stage_out)
  end

  if compact
    output = collect(skipmissing(output))
  end

  output
end

# In general, these generate the same permutation:
#     Permutation(1 .+ mdc_permutation_general(radix, depth))
#     permute_mixed_digits([radix, depth, radix], [3,2,1])
# 
# Another way to look at it: the MDC shuffle stage swaps two digits of the sample address.
# It always swaps the lowest with another, and which other one you want to swap determines the depth.
# The depth is the product of the radixes of the digits between the ones being swapped.
#
# The overall length of the permutation is depth*radix^2.
# Assuming a parallel implementation with `radix` lanes:
#   The total RAM used is `depth*radix*(radix-1)`
#   The total latency is `depth*(radix-1)`
#   A hardware implementation would use a N:N mux, where N = radix
function mdc_permutation_general(radix::Int, depth::Int, mux::AbstractVector{Int}=repeat(0:radix-1, inner=depth, outer=2); repeat::Int=1, compact::Bool=true, limit::Int=depth*radix^2)
  ram_in  = [Array{Union{Missing, Int}}(missing, depth*i) for i in 0:radix-1]
  ram_out = [Array{Union{Missing, Int}}(missing, depth*i) for i in radix-1:-1:1]
  addr_in  = zeros(Int, length(ram_in))
  addr_out = zeros(Int, length(ram_out))

  if repeat != 1
    mux = Base.repeat(mux, outer=repeat)
  end

  n = length(mux)
  output = Union{Missing, Int}[]
  for i in 0:n-1
    stage_in = radix*i .+ (0:radix-1)
    mux_in = [stage_in[1], [ram_in[j][1+addr_in[j]] for j in 2:radix]...]

    mux_sel = @. 1 + mod(mux[i+1] + (radix:-1:1), radix)
    mux_out = mux_in[mux_sel]

    stage_out = [[ram_out[j][1+addr_out[j]] for j in 1:radix-1]..., mux_out[radix]]

    for j in 2:radix
      ram_in[j][1+addr_in[j]] = stage_in[j]
      addr_in[j] = mod(addr_in[j] + 1, length(ram_in[j]))
    end
    for j in 1:radix-1
      ram_out[j][1+addr_out[j]] = mux_out[j]
      addr_out[j] = mod(addr_out[j] + 1, length(ram_out[j]))
    end

    # @info "$i" stage_in=repr(stage_in) mux_in=repr(mux_in) mux_sel=repr(mux_sel) mux_out=repr(mux_out) stage_out=repr(stage_out) ram_in ram_out
    append!(output, stage_out)
  end

  if compact
    output = collect(skipmissing(output))
    if length(output) > limit
      output = output[1:limit]
    end
  else
    if count(x -> !ismissing(x), output) > limit
      @warn "TODO: apply limit"
    end
  end

  output
end

# the standard way of driving the radix 2 MDC commutator, for depth `d`:
mdc_permutation(d::Int) = mdc_permutation(d, d, repeat(BitVector([0,1]), inner=d, outer=2))

# the same, but as a Reorder stage
function mdc_reorder(depth::Int; radix::Int=2)
  MdcReorder(depth, radix)
end

# for multi-lane MDC, the permutation ends up being equivalent to a deeper single-lane MDC
mdc_permutation(d::Int, lanes::Int) = mdc_permutation(d * Int(lanes / 2))
mdc_reorder(d::Int, lanes::Int) = mdc_reorder(d * Int(lanes / 2))

mdc_reorder(radix::Int, depth::Int, lanes::Int) = mdc_reorder(depth * Int(lanes / radix); radix)


function permute_cyclic!(dst::AbstractVector, perm::Permutation, src::AbstractVector)
  perm = perm.data

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

function permute_cyclic(perm::Permutation, x::AbstractVector)
  permute_cyclic!(similar(x), perm, x)
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
  Reorder(permute_cyclic(permutation(r1), permutation(r2)))
end

function fft_rewrite_delete_identities(r::Reorder)
  isone(permutation(r)) ? [] : false
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
