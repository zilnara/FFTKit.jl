
# TODO: add scaling stage
const FFTStage = Union{SDF, Twiddle, Reorder, Conj, Scale, AdjointR2}

struct FFTPlan
  stages :: Vector{FFTStage}
end

function plan(radix_tree::BTree; dual::Bool=false, test::Bool=true)
  if dual
    radix_tree = reverse(radix_tree)
  end

  radixes = collect(radix_tree)
  stages = FFTStage[]
  schedules = twiddle_schedules(radix_tree)

  for (i, radix) in enumerate(radixes)
    depth = prod(radixes[i+1:end])
    push!(stages, SDF(radix, depth))
    if haskey(schedules, depth)
      push!(stages, Twiddle(schedules[depth]...))
    end
  end

  if dual
    reverse!(stages)
  end

  plan = FFTPlan(stages)

  if test
    test_fft_plan(plan, "plan($(fmt_compact(radix_tree)))")
  end

  plan
end

function plan(radixes::Vector{Int}; order::FFTTwiddleOrder=dif, dual::Bool=false, test::Bool=true)
  radix_tree = (order == dif) ? foldr(BTree, radixes) : foldl(BTree, radixes)
  plan(radix_tree; dual, test)
end

Base.:(==)(x::FFTPlan, y::FFTPlan) = (x.stages == y.stages)
Base.copy(x::FFTPlan) = FFTPlan(copy(x.stages))

cost(plan::FFTPlan, n::Int=1) = sum(s -> cost(s,n), plan.stages)
radix(plan::FFTPlan) = [r for s in plan.stages for r in radix(s)]
prefer_inplace(plan::FFTPlan) = all(prefer_inplace, plan.stages) # TODO: should be true whenever an even number of stages want out-of-place?
transform_length(plan::FFTPlan) = prod(transform_length, plan.stages, init=1)
dual(x::FFTPlan) = FFTPlan(reverse(dual.(x.stages)))
Base.repeat(x::FFTPlan; inner::Int=1, outer::Int=1) = FFTPlan([repeat(s; inner, outer) for s in x.stages])

function output_order(plan::FFTPlan)
  n = transform_length(plan)

  radixes = radix(plan)
  m = length(radixes)

  addrs = Array{Union{Missing, Int}}(missing, n, m)
  addr_col = 1

  for s in plan.stages
    if s isa SDF
      r = s.butterfly.radix
      q = s.depth; p = n ÷ q
      for i in 0:p-1
        addrs[i*q .+ (1:q), addr_col] .= mod(i, r)
      end

      addr_col += 1
    else
      perm = output_order(s, 1:n)
      addrs = addrs[perm, :]
    end
  end

  [from_mixed_digits(Int.(addrs[i, :]), radixes) for i in 1:n]
end

function input_order(plan::FFTPlan)
  output_order(dual(plan))
end

Base.:(*)(p::FFTPlan, s::FFTStage) = FFTPlan([p.stages..., s])
Base.:(*)(s::FFTStage, p::FFTPlan) = FFTPlan([s, p.stages...])
Base.:(*)(x::FFTPlan, y::FFTPlan) = FFTPlan(vcat(x.stages, y.stages))

Base.:(*)(p::FFTPlan, x::Number) = p * Scale(x)
Base.:(*)(x::Number, p::FFTPlan) = Scale(x) * p
Base.:(/)(p::FFTPlan, x::Number) = p * inv(x)

function LinearAlgebra.mul!(dst::AbstractVector{T}, plan::FFTPlan, src::AbstractVector{T}; mode::FFTMode=forward, reorder_input::Bool=false, reorder_output::Bool=false) where T
  stages = plan.stages
  if isempty(stages)
    copyto!(dst, src)
    return dst
  end

  if reorder_input
    in_perm = Permutation(input_order(plan) .+ 1)
    if !isone(in_perm)
      stages = [Reorder(in_perm), stages...]
    end
  end

  if reorder_output
    out_perm = Permutation(output_order(plan) .+ 1)
    if !isone(out_perm)
      stages = [stages..., Reorder(inv(out_perm))]
    end
  end

  # TODO: more flexibility on types
  tmp0 = dst
  tmp1 = missing

  # first stage is out-of-place no matter what; calculate how many additional out-of-place stages
  # there will be after that, to determine whether `dst` should be tmp0 or tmp1.
  swaps = count(x -> !prefer_inplace(x), @view stages[2:end])
  if swaps > 0
    tmp1 = Array{eltype(dst)}(undef, size(dst))
  end
  if isodd(swaps)
    tmp0, tmp1 = tmp1, tmp0
  end

  for (i, stage) in enumerate(stages)
    if i == 1
      stage_src = src
      stage_dst = tmp0
    elseif prefer_inplace(stage)
      stage_src = stage_dst = tmp0
    else
      stage_src = tmp0
      stage_dst = tmp1
      tmp0, tmp1 = tmp1, tmp0
    end

    mul!(stage_dst, stage, stage_src; mode)
  end

  @assert tmp0 === dst
  tmp0
end

Base.:(*)(p::FFTPlan, x::AbstractVector) = mul!(similar(x), p, x)
Base.one(::Type{FFTPlan}) = FFTPlan([])

function Base.inv(p::FFTPlan)
  # note that inv computes the _backward_ form, not inverse. TODO: determine if we can sensibly do the scaling,
  # in such a way that inv(inv(p)) is at least approximately equal to p. Seems like it should be possible.
  FFTPlan([inv(s) for s in reverse(p.stages)])
end

function permute_input(p::FFTPlan, in_perm::Union{Permutation, AbstractVector{Int}}; rewrite::Bool=true, preserve_order::Bool=true)
  r = Reorder(in_perm)
  if !isone(r.perm)
    p = FFTPlan([r, p.stages...])
  end
  if rewrite
    exclude = preserve_order ? [fft_rewrite_delete_initial_reorder, fft_rewrite_delete_final_reorder] : []
    fft_rewrite(p; exclude)
  else
    p
  end
end

function set_input_order(p::FFTPlan, in_order::AbstractVector{Int}=1:transform_length(p); rewrite::Bool=true, preserve_order::Bool=true)
  current_in_order = input_order(p)
  in_perm = indexin(current_in_order, in_order .- minimum(in_order))
  if any(ismissing, in_perm)
    error("Invalid input order")
  end
  permute_input(p, Int.(in_perm); rewrite, preserve_order)
end

function permute_output(p::FFTPlan, out_perm::Union{Permutation, AbstractVector{Int}}; rewrite::Bool=true, preserve_order::Bool=true)
  dual(permute_input(dual(p), inv(out_perm); rewrite, preserve_order))
end

function set_output_order(p::FFTPlan, out_order::AbstractVector{Int}=1:transform_length(p); rewrite::Bool=true, preserve_order::Bool=true)
  dual(set_input_order(dual(p), out_order; rewrite, preserve_order))
end

function set_stage_order(p::FFTPlan, stage_order::AbstractVector{Int}; rewrite::Bool=true)
  # attempt to restructure an FFT so that its stages, listed in ascending order of depth, will be in the order given
  radixes = radix(p)
  @assert length(stage_order) == length(radixes)
  out_order = inv(permute_mixed_digits(radixes, invperm(stage_order))).data
  set_output_order(p, out_order; rewrite, preserve_order=false)
end

function natural_order(p::FFTPlan; rewrite::Bool=true)
  if rewrite
    p = fft_rewrite(p)
  end
  p = set_input_order(p; rewrite=false)
  set_output_order(p; rewrite=false)
end

# Using our more exotic primitives, we can't do as much manipulation of them (yet) but we can create plans for various convolutions and cross-correlations:
# TODO: why does this fail for plans like the 30pt ones above? Initial fiddling suggests it fails when last stage is not R2...
# TODO: revise the way we process plans so that things like transform_length, input_order, output_order, etc all work for plans like this
# TODO: add stages that do things like discard unused values, insert zeros in the data, etc - just in general, support signal length changing over the duration of the transform
function conv_plan(p::FFTPlan)
  p = set_input_order(p, rewrite=false)
  p2 = repeat(p, inner=2)
  p2 * AdjointR2(1) * inv(p2)
end

# To run such a plan, we craft our inputs and interpret the output in a particular way.
# (Note that x1 and x2 should have length equal to half the size of the base transform, this is a non-circular convolution equivalent to DSP.conv)
function run_conv_plan(p, x1, x2)
  n = length(x1)
  x = vec([[transpose(x1); transpose(x2)] zeros(2, n)])
  y = p * x
  y[1:2:4n-2]
end

# similarly, we can construct a cross-correlation plan, which when run using run_xcorr_plan will act like DSP.xcorr
function xcorr_plan(p::FFTPlan)
  p = set_input_order(p, rewrite=false)
  n = transform_length(p)
  p2 = repeat(p, inner=2)

  # this twiddle stage isn't strictly required. Without it, run_xcorr_plan just needs to put either x1 or x2 in the other half of its row.
  # i prefer to do this for conceptual consistency.
  tw = zeros(Int, 2n)
  reshape(tw, 2, :)[1, findall(isodd, output_order(p))] .= 1

  # the conjugation stage at the fft output is equivalent to reversing and conjugating x2 in run_xcorr_plan, but avoids having to actually reverse any data.

  p2 * Conj([0,1]) * Twiddle(2, tw) * AdjointR2(1) * inv(p2)
end

# run a cross-correlation plan; it's basically the same as running a convolution, except that it slices a slightly different set of indexes due to the reversal of x1
# (which is implemented by some trivial rotations and a conjugation, the main difference between xcorr_plan and conv_plan.
function run_xcorr_plan(p, x1, x2)
  n = length(x1)
  x = vec([[transpose(x1); transpose(x2)] zeros(2, n)])
  y = p * x
  y[3:2:end]
end

