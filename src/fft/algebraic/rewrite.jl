
const DEFAULT_REWRITE_RULES = Vector{Function}[
  Function[
    fft_rewrite_delete_identities,
    fft_rewrite_merge_twiddles,
    fft_rewrite_change_sdf_depth,
    fft_rewrite_merge_reorders,
    fft_rewrite_push_reorders_right,
    fft_rewrite_push_reorders_right_past_sdf,
    fft_rewrite_push_scales_right,
  ], Function[
    fft_rewrite_delete_initial_reorder,
    fft_rewrite_delete_final_reorder,
  ],
]

function fft_rewrite(plan::FFTPlan, rules = DEFAULT_REWRITE_RULES; exclude=[], test::Bool=true, test_incrementally::Bool=false, dbg::Bool=false)
  if test_incrementally
    if !test_fft_plan(plan)
      @error "refusing to rewrite broken plan"
      return plan
    end
  end

  stages = copy(plan.stages)

  for rule_batch in rules
    if !isempty(exclude)
      rule_batch = setdiff(rule_batch, exclude)
      if isempty(rule_batch)
        continue
      end
    end

    while true
      rule = fft_rewrite_once!(stages, rule_batch; dbg)
      if isnothing(rule)
        break
      end

      if test_incrementally
        plan = FFTPlan(stages)
        if !test_fft_plan(plan)
          @error "rewrite failed after rule $rule"
          return plan
        end
      end
    end
  end

  plan = FFTPlan(stages)

  if test
    test_fft_plan(plan)
  end

  plan
end

function fft_rewrite_once!(stages::Vector{FFTStage}, rules; dbg::Bool=false)
  for i in 1:length(stages)
    for j in length(stages):-1:i
      rule = fft_rewrite_once!(stages, i:j, rules)
      if !isnothing(rule)
        if dbg
          @info "applied rewrite: $rule at $(i:j)"
        end
        return rule
      end
    end
  end

  return nothing
end

function fft_rewrite_once!(stages::Vector{FFTStage}, segment::UnitRange, rules)
  for rule in rules
    replacement = missing
    if applicable(rule, stages, segment)
      replacement = rule(stages, segment)
    end

    if ismissing(replacement)
      v = @view stages[segment]
      if applicable(rule, v)
        replacement = rule(v)
      end

      if ismissing(replacement)
        if applicable(rule, v...)
          replacement = rule(v...)
        end
      end
    end

    if replacement === true
      return rule
    elseif replacement isa AbstractArray
      splice!(stages, segment, replacement)
      return rule
    elseif replacement isa FFTStage
      splice!(stages, segment, [replacement])
      return rule
    end
  end
  nothing
end
