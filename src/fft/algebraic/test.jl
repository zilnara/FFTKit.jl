
function test_fft_plan(plan::FFTPlan, label::String=""; repetitions::Int=10)
  # maybe just trust linearity and use something like:
  #   sqrt(mean(abs2, fft_matrix(plan) .- dft_matrix(n)))
  x = test_fft_plan(plan, forward,  label; repetitions)
  y = test_fft_plan(plan, backward, label; repetitions)
  z = test_fft_plan(plan, inverse , label; repetitions)
  all([x, y, z])
end

# TODO this doesn't actually use the "expected_function" param. The real goal here, which is not
#      yet satisfied, is to be able to do things like construct the backward/inverse plan using Base.inv etc.,
#      and test whether that plan run in the "forward" mode will yield the correct output
function test_fft_plan(plan::FFTPlan, mode::FFTMode, label::String=""; repetitions::Int=10, expected_function=missing)
  n = transform_length(plan)

  plan_or_label = isempty(label) ? plan : label

  if !isperm(1 .+ input_order(plan)) || !isperm(1 .+ output_order(plan))
    @info "FFT plan is not valid" plan=plan_or_label
    return false
  end

  # TODO: not sure how reasonable this limit actually is.
  c = cost(plan)
  k = sqrt(4 * get(c, COMPLEX_MULT, 0) + get(c, COMPLEX_SUM, 0) + 2 * get(c, COMPLEX_SCALE, 0))
  q = quantile(Chi(n), 1 - 1e-9)
  err_limit = k * q * eps() * log2(n)

  rms(x) = sqrt(mean(abs2, x))
  x = Array{ComplexF64}(undef, n)
  y = Array{ComplexF64}(undef, n)
  z = Array{ComplexF64}(undef, n)

  if mode == forward
    fftw = plan_fft(x)
  elseif mode == backward
    fftw = plan_bfft(x)
  else
    fftw = plan_ifft(x)
  end

  scale = prod([s.scale for s in plan.stages if s isa Scale])

  for i in 1:repetitions
    seed = rand(Int)

    randn!(Xoshiro(seed), x)
    mul!(y, plan, x; mode, reorder_input=true, reorder_output=true); if !isone(scale) y .*= inv(scale); end
    if ismissing(expected_function)
      mul!(z, fftw, x)
    else
      z = expected_function(x)
    end

    err = rms(@. (y - z) / (abs(z) + eps()))
    if err >= err_limit
      @info "FFT plan fails after $i tests" mode plan=plan_or_label seed err err_limit c k
      return false
    end
  end

  true
end

macro test_fft_plan(name)
  return :( test_fft_plan($name, $(repr(name))) )
end

# build and test some plans to sanity-check the code here
const PLAN_16PT_R2_DIF      = plan([2,2,2,2], order=dif, dual=false)
const PLAN_16PT_R2_DIF_DUAL = plan([2,2,2,2], order=dif, dual=true)
const PLAN_16PT_R2_DIT      = plan([2,2,2,2], order=dit, dual=false)
const PLAN_16PT_R2_DIT_DUAL = plan([2,2,2,2], order=dit, dual=true)

# we don't currently have a planning routine that supports compound butterflies.
# we can build equivalent transforms using BTree scheduling and primitive butterflies, though.
const PLAN_16PT_R22         = plan(BTree((2,2),(2,2)), dual=false)
const PLAN_16PT_R22_DUAL    = plan(BTree((2,2),(2,2)), dual=true)

const PLAN_16PT_R4      = plan([4,4], dual=false)
const PLAN_16PT_R4_DUAL = plan([4,4], dual=true)

const PLAN_32PT_R2_DIF      = plan([2,2,2,2,2], order=dif, dual=false)
const PLAN_32PT_R2_DIF_DUAL = plan([2,2,2,2,2], order=dif, dual=true)
const PLAN_32PT_R2_DIT      = plan([2,2,2,2,2], order=dit, dual=false)
const PLAN_32PT_R2_DIT_DUAL = plan([2,2,2,2,2], order=dit, dual=true)

const PLAN_6PT_MIXED      = plan([3,2], dual=false)
const PLAN_6PT_MIXED_DUAL = plan([3,2], dual=true)

const PLAN_30PT_MIXED_DIF       = plan([5,3,2], order=dif, dual=false)
const PLAN_30PT_MIXED_DIF_DUAL  = plan([5,3,2], order=dif, dual=true)
const PLAN_30PT_MIXED_DIT       = plan([5,3,2], order=dit, dual=false)
const PLAN_30PT_MIXED_DIT_DUAL  = plan([5,3,2], order=dit, dual=true)

# It's also possible to create plans with non-standard stage ordering, but we don't yet have functions to do so.
# We can even insert weird reorderings internally and the order-detection logic above figures it out.
# None of the "Reorder" blocks here are necessary, they're just here to make the `output_order` function cry.
# Depending where they are put, it may be necessary to adjust twiddles - for example, in the first twiddle stage here.
const PLAN_12PT_MIXED = FFTPlan([
  Reorder([1,3,4,2,5,6,7,8,9,10,12,11]),
  SDF(2, 6),
  Reorder([7,8,9,10,11,12, 1,2,3,4,5,6]),
  Twiddle(permute_cyclic(Permutation([2,1]), permute_cyclic(Permutation([7,8,9,10,11,12, 1,2,3,4,5,6]), [0,0,0,0,0,0, 0,1,2,3,4,5]))),
  Reorder([2,1]),
  SDF(3, 2),
  Twiddle([0,0,0,0]), # some no-op stages to test rewriting (they should be deleted)
  Reorder([1,2,3,4,5,6]),
  Twiddle([0,0, 0,1, 0,2]),
  SDF(2, 1),
  Reorder([1,3,2,4,5,6,7,8,9,10,12,11]),
])
@test_fft_plan(PLAN_12PT_MIXED)

const PLAN_12PT_MIXED_STRUCTURE = FFTPlan([
  SDF(2, 3),
  Reorder([3,2,1]),
  Twiddle([0,0,0,0,2,4,0,0,0,1,3,5]),
  SDF(3, 1),
  Twiddle([0,0,0,0,0,0,0,2,4,0,2,4]),
  Reorder([5,6, 3,4, 1,2]),
  SDF(2, 6),
])
@test_fft_plan(PLAN_12PT_MIXED_STRUCTURE)

const PLAN_12PT_PERMUTED = FFTPlan([
  Reorder([1,4,2,5,3,6]),
  SDF(2, 1),
  Reorder([1,3,5,2,4,6]),
  #Reorder([3,2,1]),
  Twiddle([0,0,0,0,2,4,0,0,0,1,3,5]),
  SDF(3, 1),
  Twiddle([0,0,0,0,0,0,0,2,4,0,2,4]),
  #Reorder([5,6, 3,4, 1,2]),
  SDF(2, 6),
])
@test_fft_plan(PLAN_12PT_PERMUTED)

# 4-lane radix-2 DIF MDC. Figure 36 from "A Survey on Pipelined FFT Hardware Architectures"
const PLAN_16PT_R2_DIF_MDC = FFTPlan([
  SDF(2, 1)
  Twiddle([0, 0, 0, 4, 0, 1, 0, 5, 0, 2, 0, 6, 0, 3, 0, 7])
  Reorder([0, 2, 1, 3]) # parallel reordering; stages 1 and 2 form something like a R4 stage, but with different twiddles
  SDF(2, 1)
  Twiddle(8, [0, 0, 0, 0, 0, 1, 0, 1, 0, 2, 0, 2, 0, 3, 0, 3])
  mdc_reorder(2, 4)
  SDF(2, 1)
  Twiddle(4, [0, 0, 0, 0, 0, 1, 0, 1])
  mdc_reorder(1, 4)
  SDF(2, 1)
])
@test_fft_plan(PLAN_16PT_R2_DIF_MDC)

# Figure 40 from the same paper (minor variation; stage 3 twiddle is before reorder, instead of after)
const PLAN_16PT_R22_MDC = recompute_twiddles(FFTPlan([
  SDF(2, 1)
  Reorder([0, 2, 1, 3])
  SDF(2, 1)
  Reorder([0, 2, 1, 3])
  mdc_reorder(2,4)
  SDF(2, 1)
  Reorder([0, 2, 1, 3])
  mdc_reorder(1,4)
  SDF(2,1)
]), BTree((2,2),(2,2)))

# And a MDC of my own contrivance, to demonstrate flexibility of automatic twiddle computation
# This structure defines a 64-point transform which should be realizable as a 4-parallel transform
# with 7 non-trivial rotators
const PLAN_64PT_MDC = recompute_twiddles(FFTPlan([
  SDF(2,1)
  mdc_reorder(1,4)
  SDF(2,1)
  mdc_reorder(4,4)
  SDF(4,1)
  mdc_reorder(2,4)
  SDF(2,1)
  mdc_reorder(8,4)
  SDF(2,1)
]), order=dif)

# Give a name to a reordering that is used in many 4-lane MDC designs that follow.
# It's technically also equivalent to mdc_reorder(1, 2) but I think this is a more useful way to think of it
const MDC_4LANE_SWAP = Reorder([0,2,1,3])

# Another custom MDC, this one with an extra initial reordering stage that allows it to consume
# natural input order in what I believe is the most efficient way possible.
# Additionally, another slightly redundant internal delay allows the radix-2^2 form to have as
# few as 7 nontrivial rotators.
#
# I think, now that I think of it, some of the custom FFTs I want to build will have structures that
# let me eliminate the initial reorder here (zero-padded FFT for example will have all the zeros
# in places that don't require reordering at all - and the truncated form will end up being dual,
# so maybe MDC isn't so bad for a correlator after all.
const PLAN_64PT_MDC_NATURAL = recompute_twiddles(FFTPlan([
  mdc_reorder(8,4)
  SDF(2,1)
  mdc_reorder(4,4)
  SDF(2,1)
  mdc_reorder(2,4)
  SDF(2,1)
  MDC_4LANE_SWAP
  mdc_reorder(1,4)
  SDF(2,1)
  MDC_4LANE_SWAP
  mdc_reorder(1,4)
  SDF(2,1)
  mdc_reorder(8,4)
  SDF(2,1)
]), BTree((2, 2), ((2, 2), (2, 2))))

# most "obvious" structure that has no input reordering and accepts input in the most natural
# order for a 0-padded FFT (i.e, a 2N point FFT where the high N inputs are all zero)
const PLAN_64PT_MDC_PAD_NATURAL = recompute_twiddles(FFTPlan([
  SDF(2,1)
  mdc_reorder(8,4)
  SDF(2,1)
  mdc_reorder(4,4)
  SDF(2,1)
  mdc_reorder(2,4)
  SDF(2,1)
  mdc_reorder(1,4)
  SDF(2,1)
  MDC_4LANE_SWAP
  SDF(2,1)
]), BTree(2, (2, (2, (2, (2, 2))))))
# the following twiddle structures achieve 7 rotators (the first is the default DIF order):
#     BTree(2, (2, (2, (2, (2, 2)))))
#     BTree(2, ((2, 2), (2, (2, 2))))
#     BTree((2, 2), (2, (2, (2, 2))))
#     BTree((2, (2, 2)), (2, (2, 2)))

# the most obvious structure above unfortunately splits some natural radix-4 groups. At the cost
# of a bit of reordering, we can put them back together and use a radix-4 (or 2^2) structure to
# eliminate an extra rotator:
#
# TODO: this might also be a place where the "triangular matrix" representation provides useful
#       degrees of freedom?
const PLAN_64PT_MDC_PAD_NATURAL_V2 = recompute_twiddles(FFTPlan([
  # stage orders given as digit order, LSB to MSB.
  # If the given bit order is used as an array, the coefficient order at each labeled stage
  # can be computed as: permute_mixed_digits([2,2,2,2,2,2], invperm(bit_order)).
  # The vertical bar divides the first two bits from the rest; these bits are the ones accessible
  # by parallel FFT stages (SDF(2,1) and SDF(2,2))
  #
  # after the order, a mask is shown indicating which digits have been transformed along

  # TODO document my pen-and-paper method of tabulating these things.
  # the gist is that the actual stage orders are implicitly anchored by the sequence of SDF stages,
  # which set the output stages in forward order (or the input stages in reverse order). The rest is
  # just tracking digit order shuffles and which digits have been transformed over.

  # the third column shows this perspective, starting off with a list of unknown digit places and
  # solving for them along the way (using output indexes)
  # Note carefully in particular the difference in digit-reversal between radix 2 and 4 stages.

  # TODO also document how this tabulation makes it clear that you can't do much better than this
  # if you want to have a "padded natural order" input, unless there's an efficient operation to
  # shift the digits in the coefficient order (i.e., to go from 12|3456 to 61|2345 or something similar.
  # it's not an especially HARD transform, but it seems fairly clear it can't be done in a way that
  # doesn't add at least as much latency as the transforms derived by this method.

  # input             61|2345   __|____   ab|cdef
  
  #transform 6
  SDF(2,1)          # 61|2345   6_|____   1b|cdef [a = 6 in, 1 out]
  
  # stash 6, pull up 5 (grouping 56)
  MDC_4LANE_SWAP    # 16|2345   _6|____   b1|cdef
  mdc_reorder(8,4)  # 56|2341   _6|____   f1|cdeb

  # transferm 5
  SDF(2,1)          # 56|2341   56|____   21|cdeb [f = 5 in, 2 out]

  # swap 56 with 34
  # this can't be done with a simple radix-4 digit swap because 34 is not aligned.
  mdc_reorder(4,4)  # 46|2351   _6|__5_   e1|cd2b
  MDC_4LANE_SWAP    # 64|2351   6_|__5_   1e|cd2b
  mdc_reorder(2,4)  # 34|2651   __|_65_   de|c12b

  # transform 34
  SDF(4,1)          # 34|2651   34|_65_   34|c12b [de = 34 in, 34 out]

  # stash 3 and 4, swapping them with 1 and 2
  mdc_reorder(1,4)  # 24|3651   _4|365_   c4|312b
  MDC_4LANE_SWAP    # 42|3651   4_|365_   4c|312b
  mdc_reorder(8,4)  # 12|3654   __|3654   ac|3124

  # transform 12
  SDF(4,1)          # 12|3654   12|3654   56|3124 [ac = 12 in, 56 out]
]), BTree((2,2),(4,4)))

# it's also possible to create radix-4 MDCs, optionally with natural order input.
# This one has 6 nontrivial rotations on 4 lanes.
# TODO: is it possible to define a 4-lane commutator structure that's any cleaner or more efficient than stacking the 2-lane ones as done here?
const PLAN_64PT_MDC_R4 = recompute_twiddles(FFTPlan([
  mdc_reorder(4,4)
  MDC_4LANE_SWAP
  mdc_reorder(8,4)
  SDF(4,1)
  mdc_reorder(2,4)
  MDC_4LANE_SWAP
  mdc_reorder(1,4)
  SDF(4,1)
  MDC_4LANE_SWAP
  mdc_reorder(8,4)
  MDC_4LANE_SWAP
  mdc_reorder(4,4)
  SDF(4,1)
]))

# another 4-lane MDC, this one with 4 stages for 256 points, and with 9 nontrivial rotators.
# Again, the input is in "padded natural" order for use in a correlator
const PLAN_256PT_MDC_R4 = FFTPlan([
  MDC_4LANE_SWAP; mdc_reorder(32,4)
  SDF(4,1)
  twiddle_gen_v2([2,2,2,2,2,2,2,2], 7:8, 1:6, [1,6,5,4,3,2,8,7])
  mdc_reorder(16,4); MDC_4LANE_SWAP; mdc_reorder(8,4)
  SDF(4,1)
  twiddle_gen_v2([2,2,2,2,2,2,2,2], 5:6, 1:4, [1,7,8,4,3,2,6,5])
  mdc_reorder(4,4); MDC_4LANE_SWAP; mdc_reorder(2,4)
  # instead of the immediately previous twiddle, we can commute it past the MDC reordering
  # by just changing the digit order as follows (but we wouldn't in this case because it
  # would increase the number of rotators needed):
  # twiddle_gen_v2([2,2,2,2,2,2,2,2], 5:6, 1:4, [1,7,8,5,6,2,4,3])
  # TODO: it may be possible to use this trick to reach 6 rotators in a naive padded-natural-
  #       order transform which does less reordering
  SDF(4,1)
  twiddle_gen_v2([2,2,2,2,2,2,2,2], 3:4, 1:2, [1,7,8,5,6,2,4,3])
  MDC_4LANE_SWAP; mdc_reorder(1,4); MDC_4LANE_SWAP; mdc_reorder(32,4)
  SDF(4,1)
])
@test_fft_plan(PLAN_256PT_MDC_R4)

# and now, a big one to experiment with: 16384 points, natural-zero-pad input order, 4 lanes
# (actually define it generically for power-of-4 length, then instantiate one)
function plan_mdc_r4_4lane_pad_natural(size::Int; retwiddle::Bool=false, test::Bool=true)
  # TODO support optional extra radix-2 stage?

  # TODO work out for other radix/lane count?
  # in the places where these are used, i've tried to describe things in terms of the ways
  # they seem like they should depend on the parameters, but they are not fully correct and
  # so these values can't just freely change.
  radix = lanes = 4
  sequential_size = size ÷ lanes
  num_sequential_stages = round(Int, log(radix, sequential_size))
  @assert size == lanes * radix ^ num_sequential_stages

  lane_exchange = MDC_4LANE_SWAP

  stages = FFTStage[]
  function mdc(stage, depth)
    push!(stages, lane_exchange)
    push!(stages, mdc_reorder(depth, radix))
  end
  function sdf(stage)
    push!(stages, SDF(radix,1))
  end
  function twiddle(stage)
    tw = Twiddle([2, 4^(stage-1), 2sequential_size ÷ 4^stage, 4], [1,3], [4])
    push!(stages, tw)
  end

  for stage in 1:num_sequential_stages
    mdc(stage, 2 * sequential_size ÷ radix ^ stage)
    sdf(stage)
    twiddle(stage)
    mdc(stage, sequential_size ÷ radix ^ stage)
  end

  stage = 1 + num_sequential_stages
  mdc(stage, size ÷ 2radix)
  sdf(stage)

  plan = FFTPlan(stages)
  if retwiddle
    plan = recompute_twiddles(plan; test=false)
  end
  if test
    test_fft_plan(plan, "plan_mdc_r4_4lane_pad_natural($(size))")
  end

  plan
end

const PLAN_16KPT_MDC_R4 = plan_mdc_r4_4lane_pad_natural(16384; test=false)

PLAN_16KPT_MDC_R22(; test::Bool=true) = recompute_twiddles(FFTPlan([
  MDC_4LANE_SWAP; mdc_reorder(2048,4)
  SDF(2,2); SDF(2,1)
  
  MDC_4LANE_SWAP; mdc_reorder(1024,4); MDC_4LANE_SWAP; mdc_reorder(512,4)
  SDF(2,2); SDF(2,1)
  
  MDC_4LANE_SWAP; mdc_reorder(256,4); MDC_4LANE_SWAP; mdc_reorder(128,4)
  SDF(2,2); SDF(2,1)
  
  MDC_4LANE_SWAP; mdc_reorder(64,4); MDC_4LANE_SWAP; mdc_reorder(32,4)
  SDF(2,2); SDF(2,1)
  
  MDC_4LANE_SWAP; mdc_reorder(16,4); MDC_4LANE_SWAP; mdc_reorder(8,4)
  SDF(2,2); SDF(2,1)
  
  MDC_4LANE_SWAP; mdc_reorder(4,4); MDC_4LANE_SWAP; mdc_reorder(2,4)
  SDF(2,2); SDF(2,1)
  
  MDC_4LANE_SWAP; mdc_reorder(1,4); MDC_4LANE_SWAP; mdc_reorder(2048,4)
  SDF(2,2); SDF(2,1)
]); test)

# and for comparison, a R2 version. I haven't exhaustively figured out the best twiddle structure,
# but so far the best I've achieved is 23 nontrivial rotators, vs 18 for the R4 version.
# TODO: build some smaller ones and try to understand what twiddle structures work well
PLAN_16KPT_MDC_R2(; test::Bool=true) = recompute_twiddles(FFTPlan([
  SDF(2,1)
  mdc_reorder(2048,4)
  SDF(2,1)
  mdc_reorder(1024,4)
  SDF(2,1)
  mdc_reorder(512,4)
  SDF(2,1)
  mdc_reorder(256,4)
  SDF(2,1)
  mdc_reorder(128,4)
  SDF(2,1)
  mdc_reorder(64,4)
  SDF(2,1)
  mdc_reorder(32,4)
  SDF(2,1)
  mdc_reorder(16,4)
  SDF(2,1)
  mdc_reorder(8,4)
  SDF(2,1)
  mdc_reorder(4,4)
  SDF(2,1)
  mdc_reorder(2,4)
  SDF(2,1)
  mdc_reorder(1,4)
  SDF(2,1)
  MDC_4LANE_SWAP
  SDF(2,1)
]); test)

# And we can use MDC with other radixes as well (though it's not easy to mix radixes):
const PLAN_81PT_MDC_R3 = recompute_twiddles(FFTPlan([
  SDF(3,1)
  mdc_reorder(3,1,3)
  SDF(3,1)
  mdc_reorder(3,3,3)
  SDF(3,1)
  mdc_reorder(3,9,3)
  SDF(3,1)
]))

const PLAN_125PT_MDC_R5 = recompute_twiddles(FFTPlan([
  SDF(5,1)
  mdc_reorder(5,5,5)
  SDF(5,1)
  mdc_reorder(5,1,5)
  SDF(5,1)
]))

const CONV_PLAN_16PT_R22 = conv_plan(PLAN_16PT_R22)
const CONV_PLAN_30PT_MIXED_DIF = conv_plan(PLAN_30PT_MIXED_DIF)
const CONV_PLAN_256PT_MDC_R4 = conv_plan(PLAN_256PT_MDC_R4)

const XCORR_PLAN_16PT_R22 = xcorr_plan(PLAN_16PT_R22)
