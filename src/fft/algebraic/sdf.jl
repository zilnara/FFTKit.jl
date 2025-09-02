
struct Butterfly
  radix     :: Int
  apply     :: Function
  apply!    :: Function
  cost      :: Cost

  Butterfly(radix::Int, apply::Function, apply!::Function) = new(radix, apply, apply!, cost(apply))
  Butterfly(radix::Int, apply::Function, apply!::Function, cost::Cost) = new(radix, apply, apply!, cost)
end

Base.:(==)(x::Butterfly, y::Butterfly) = (x.radix == y.radix) && (x.cost == y.cost)

const R2 = Butterfly(2, r2, r2!)
const R3 = Butterfly(3, r3, r3!)
const R4 = Butterfly(4, r4, r4!)
const R5 = Butterfly(5, r5, r5!)

const FIXED_BUTTERFLIES = [R2, R3, R4, R5]

function fftw_butterfly(radix::Int)
  # to play with radixes we haven't implemented, define a butterfly that just uses fftw

  function apply!(dst::AbstractArray, src::AbstractArray{<:Number}; mode::FFTMode=forward)
    fft_dst = reshape(dst, radix, :)
    fft_src = reshape(src, radix, :)

    if !(eltype(fft_src) <: Complex{<:AbstractFloat})
      T = promote_type(float(eltype(fft_src)), Complex{Int})
      fft_src = T.(fft_src)
    end

    if mode == forward
      p = plan_fft(fft_src, 1)
    elseif mode == backward
      p = plan_bfft(fft_src, 1)
    else
      p = plan_ifft(fft_src, 1)
    end

    mul!(fft_dst, p, fft_src)

    dst
  end

  function apply!(x::AbstractArray; mode::FFTMode=forward)
    fft_x = reshape(x, radix, :)
    if mode == forward
      fft!(fft_x, 1)
    elseif mode == backward
      bfft!(fft_x, 1)
    elseif mode == inverse
      ifft!(fft_x, 1)
    end
    x
  end

  function apply(x::AbstractArray; mode::FFTMode=forward)
    apply!(similar(x, promote_type(float(eltype(x)), Complex{Int})), x; mode)
  end

  c = Cost(UNKNOWN_COST=>1)

  Butterfly(radix, apply, apply!, c)
end

function Butterfly(radix::Int)
  for b in FIXED_BUTTERFLIES
    if b.radix == radix
      return b
    end
  end

  fftw_butterfly(radix)
end

cost(b::Butterfly, n::Int=1) = n * b.cost
radix(b::Butterfly) = [b.radix]
transform_length(b::Butterfly) = b.radix
dual(b::Butterfly) = b
Base.repeat(b::Butterfly; inner::Int=1, outer::Int=1) = SDF(b, inner)
prefer_inplace(::Butterfly) = true

function LinearAlgebra.mul!(dst::AbstractArray, b::Butterfly, src::AbstractArray; mode::FFTMode=forward)
  b.apply!(dst, src; mode)
end

function Base.:(*)(b::Butterfly, x::AbstractArray; mode::FFTMode=forward)
  b.apply(x; mode)
end

struct SDF
  butterfly :: Butterfly
  depth     :: Int
  inverse   :: Bool
end

SDF(radix::Int, depth::Int, inverse::Bool=false) = SDF(Butterfly(radix), depth, inverse)

Base.:(==)(x::SDF, y::SDF) = (x.butterfly == y.butterfly) && (x.depth == y.depth)

function Base.show(io::IO, s::SDF)
  if s.inverse
    print(io, "inv(")
  end
  if in(s.butterfly, FIXED_BUTTERFLIES)
    print(io, "SDF($(s.butterfly.radix), $(s.depth))")
  else
    print(io, "SDF($(s.butterfly), $(s.depth))")
  end
  if s.inverse
    print(io, ")")
  end
end

cost(s::SDF, n::Int=1) = cost(s.butterfly, n) + Cost() # TODO: add latency and memory elements related to SDF stage itself
radix(s::SDF) = radix(s.butterfly)
transform_length(s::SDF) = transform_length(s.butterfly)
dual(s::SDF) = SDF(dual(s.butterfly), s.depth, s.inverse)
Base.repeat(s::SDF; inner::Int=1, outer::Int=1) = SDF(s.butterfly, inner * s.depth, s.inverse)
prefer_inplace(::SDF) = true  # TODO: determine whether this is really the best (fastest?) way to process
function LinearAlgebra.mul!(dst::AbstractVector, s::SDF, src::AbstractVector; mode::FFTMode=forward)
  if s.depth == 0
    return copy!(dst, src)
  end

  n = length(src)
  @assert mod(n, s.depth * s.butterfly.radix) == 0

  if s.inverse
    if mode == forward
      mode = inverse
      scale = nothing
    elseif mode == backward
      mode = forward
      scale = 1/n
    elseif mode == inverse
      mode = forward
      scale = nothing
    end
  else
    scale = nothing
  end

  @simd for i in 1:s.depth
    s.butterfly.apply!((@view dst[i:s.depth:n]), (@view src[i:s.depth:n]); mode)
  end

  if !isnothing(scale)
    dst .*= scale
  end

  dst
end

function Base.:(*)(s::SDF, x::AbstractVector; mode::FFTMode=forward)
  T = promote_type(float(eltype(x)), Complex{Int})
  y = similar(x, T)
  mul!(y, s, x; mode)
  y
end

# TODO probably better to have the butterfly itself implement only forward/backward modes, and leave "inverse" scaling to the overal plan
Base.inv(s::SDF) = SDF(s.butterfly, s.depth, !s.inverse)

function valid_stage_structure(sdf_stages::AbstractArray{SDF})
  # Note that this only validates the SDF stage depths. It does not care at all if the twiddles
  # or internal reorderings are sane.

  n = 1
  for s in sort(sdf_stages, by=s->s.depth)
    if s.depth != n
      return false
    end

    n *= transform_length(s)
  end

  true
end

function sdf_groups(radix::Int, depth::Int, n::Int = radix * depth)
  sdf_len = radix * depth
  groups = StepRangeLen{Int}[]
  for i in 1:sdf_len:n
    for j in 1:depth
      push!(groups, StepRangeLen(i+j-1, depth, radix))
    end
  end
  groups
end

sdf_groups(s::SDF) = sdf_groups(s.butterfly.radix, s.depth)
sdf_groups(s::SDF, n::Int) = sdf_groups(s.butterfly.radix, s.depth, n)

function sdf_group_perms(in_groups::AbstractVector{<:AbstractVector{Int}}, out_groups::AbstractVector{<:AbstractVector{Int}})
  n_groups = length(in_groups)
  @assert length(out_groups) == n_groups
  if n_groups < 1
    p = Permutation(0)
    return p, p, p
  end

  radix = length(in_groups[1])
  @assert radix > 1 # TODO: handle this semi-trivial case?
  @assert all(length.(in_groups) .== radix)
  @assert all(length.(out_groups) .== radix)

  in_depth  = in_groups[1][2]  - in_groups[1][1]
  out_depth = out_groups[1][2] - out_groups[1][1]
  # TODO check consistency

  in_group_ids = sortperm([minimum(g) for g in in_groups])
  out_group_ids = sortperm([minimum(g) for g in out_groups])

  intergroup_perm = Permutation(indexin(out_group_ids, in_group_ids))

  # intra-group permutations can be described in either the in_groups order or the out_groups order.
  # we calculate just one but return both.

  in_groups_nodepth = [sortperm(g) for g in in_groups]
  out_groups_nodepth = [sortperm(g) for g in out_groups]
  out_groups_nodepth_pre = out_groups_nodepth[inv(intergroup_perm).data]

  intragroup_perms_pre = [Permutation(indexin(in_groups_nodepth[i], out_groups_nodepth_pre[i])) for i in 1:n_groups]
  intragroup_perms_post = intragroup_perms_pre[intergroup_perm.data]

  intragroup_perms_pre, intergroup_perm, intragroup_perms_post
end

function sdf_depth(groups::AbstractVector{<:AbstractVector{Int}}, radix::Union{Missing,Int}=missing; dbg::Bool=false)
  # find a depth that generates the given groups, ignoring ordering within the groups
  # first, convert each group to a StepRangeLen (if possible). If it's not possible, then it's not a valid grouping.
  function f(group)
    if group isa StepRangeLen
      group
    else
      group = sort(group)
      steps = diff(group)
      if all(steps .== steps[1])
        StepRangeLen(group[1], steps[1], length(group))
      else
        missing
      end
    end
  end

  n = length(groups)
  starts = Array{Int}(undef, n)
  return_radix = ismissing(radix)
  radix = radix
  depth = missing

  for (i, g) in enumerate(groups)
    g = f(g)
    if ismissing(g)
      if dbg
        @info "group not regularly spaced" i g
      end
      return nothing
    end

    if ismissing(radix)
      radix = length(g)
    elseif radix != length(g)
      if dbg
        @info "radix mismatch" i g radix length(g)
      end
      return nothing
    end

    if ismissing(depth)
      depth = g.step
    elseif depth != g.step
      if dbg
        @info "depth mismatch" i g depth g.step
      end
      return nothing
    end

    push!(starts, g[1])
  end

  sort!(starts)
  if all(diff(starts) .== 1)
    if dbg
      @info "start indexes wrong" starts
    end
    return nothing
  end

  return_radix ? (radix, depth) : depth
end

function sdf_regroup_permutation(radix::Int, old_depth::Int, new_depth::Int)
  if old_depth == new_depth
    return Permutation(1)
  end

  n = radix * lcm(old_depth, new_depth)

  old_order = sdf_groups(radix, old_depth, n)
  new_order = sdf_groups(radix, new_depth, n)

  perm = Array{Int}(undef, n)
  for (old_group, new_group) in zip(old_order, new_order)
    perm[new_group] .= old_group
  end

  Permutation(perm)
end

function sdf_change_depth(s::SDF, new_depth::Int)
  if s.depth == new_depth
    return [s]
  end

  p = sdf_regroup_permutation(s.butterfly.radix, s.depth, new_depth)
  [Reorder(p), SDF(s.butterfly, new_depth), Reorder(inv(p))]
end

function permute_depth(p::Permutation)
  maximum(io -> abs(io[1]-io[2]), enumerate(p.data))
end

function fft_rewrite_change_sdf_depth(stages::AbstractArray, segment_range::UnitRange)
  segment = @view stages[segment_range]
  sdf_pos = findall(s -> s isa SDF, segment)
  if length(sdf_pos) != 1
    return false
  end

  if any(s -> !(s isa SDF || s isa Reorder), segment)
    return false
  end

  n = prod(transform_length, stages)
  sdf_pos = sdf_pos[1]
  reorder_pre = prod(segment[1:sdf_pos-1], init=one(Reorder)).perm
  old_sdf = segment[sdf_pos]
  reorder_post = prod(segment[sdf_pos+1:end]; init=one(Reorder)).perm

  radix = old_sdf.butterfly.radix

  best = missing
  best_score = missing

  for new_depth in divisors(n ÷ radix)
    p = sdf_regroup_permutation(radix, old_sdf.depth, new_depth)
    new_sdf = SDF(old_sdf.butterfly, new_depth, old_sdf.inverse)

    candidate = (new_depth, permute_cyclic(reorder_pre, p), permute_cyclic(inv(p), reorder_post))
    score = (new_depth, permute_depth(candidate[2]), permute_depth(candidate[3]))

    #TODO: work out what scoring I actually want and implement it, also make it controllable by kwargs probably
    # this seems like a reasonable starting point... goals are, in priority order:
    #     * reduce reordering
    #     * in a tie, prefer to put greatest reordering on the right
    #     * all else being equal, prefer shorter stages
    # TODO: maybe even do a global optimization over all possible depths for all stages?
    #       or perhaps push parts of the permutation affecting input order to the left, and the rest to the right?
    score = (score[2] + score[3], score[2], score[2], score[1])

    if ismissing(best) || score < best_score
      best = candidate
      best_score = score
    end
  end

  (best_depth, best_pre, best_post) = best
  if best_depth != old_sdf.depth
    [Reorder(best_pre), SDF(old_sdf.butterfly, best_depth, old_sdf.inverse), Reorder(best_post)]
  else
    false
  end
end

fft_rewrite_reorder_sdf(r::Reorder, s::SDF) = fft_rewrite_reorder_sdf(r, s, Reorder(Permutation(1)))
fft_rewrite_reorder_sdf(s::SDF, r::Reorder) = fft_rewrite_reorder_sdf(Reorder(Permutation(1)), s, r)

function fft_rewrite_push_reorders_right(r::Reorder, s::SDF)
  # reorders can only move past a SDF stage if they don't change which subset of terms feed into each butterfly.
  # the way in which they are grouped might change, however (i.e., the depth of the SDF stage, and potentially
  # the permutation of each group (independently). Permutations of a group of inputs become a combination of
  # rotations and permutation of the outputs. Not all input permutations within a group are representable on the
  # output. For example, the permutation (1,2)(3)(4) on the input of a radix 4 butterfly cannot be moved past it
  # without a more complex representation than we support.

  # movements that change depth are, to an extent, handled by `fft_rewrite_order_sdf`.
  # here we try to pick up some more complex cases.

  depth = s.depth; radix = s.butterfly.radix

  perm_len = length(r.perm); sdf_len = radix * depth
  n = lcm(perm_len, sdf_len)

  in_groups = sdf_groups(radix, depth, n)

  perm = r.perm.data
  if perm_len != n
    perm = extend_perm(perm, n)
  end
  perm_groups = [perm[g] for g in in_groups]
  perm_depth = sdf_depth(perm_groups, radix)

  if isnothing(perm_depth)
    # permutation does not respect any SDF structure
    # TODO: see if there's a way to split the SDF so that a part of it does, leaving behind a permutation that is in some concrete sense "simpler". To an extent, this is done by `fft_rewrite_order_sdf`
    return false
  end

  intragroup_perms_pre, intergroup_perms, intragroup_perms_post = sdf_group_perms(in_groups, perm_groups)

  if all(isone, intragroup_perms_post)
    # Happy day! no intra-group perms to handle
    return [SDF(s.butterfly, perm_depth, s.inverse), r]
  end

  # TODO: To handle this permutation, we need to permute the inputs to at least one butterfly group.
  #       This may or may not be possible, depending on the permutation. We can potentially leave some
  #       of the permutations behind, while others may become some combination of rotation and permutation
  #       of the output.
  #
  #       Either way, we need to use the identified decomposition of the permutation, which may or may
  #       not actually be correct (I don't have high confidence, in particular for the inter-group ordering
  #       when depth differs), to identify the permutation for each group and decide how to handle it.
  #
  #       For now, just print out some stuff about what we would like to do
  perm = perm .- 1
  in_groups = [collect(g .- 1) for g in in_groups]
  perm_groups = [collect(g .- 1) for g in perm_groups]
  in_depth = depth
  @info "TODO: rewrite" r.perm (radix, depth)
  @show perm in_groups perm_groups radix in_depth perm_depth

  false
end
