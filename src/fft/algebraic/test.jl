
function test_fft_plan(plan::FFTPlan, label::String=""; repetitions::Int=10)
  # maybe just trust linearity and use something like:
  #   sqrt(mean(abs2, fft_matrix(plan) .- dft_matrix(n)))
  x = test_fft_plan(plan, forward,  label; repetitions)
  y = test_fft_plan(plan, backward, label; repetitions)
  z = test_fft_plan(plan, inverse , label; repetitions)
  all([x, y, z])
end

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
    mul!(z, fftw, x)

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

const CONV_PLAN_16PT_R22 = conv_plan(PLAN_16PT_R22)
const CONV_PLAN_30PT_MIXED_DIF = conv_plan(PLAN_30PT_MIXED_DIF)

const XCORR_PLAN_16PT_R22 = xcorr_plan(PLAN_16PT_R22)
