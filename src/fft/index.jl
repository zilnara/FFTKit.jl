
function to_mixed_digits(n::Integer, bases::AbstractVector{<:Integer})
  digits = Int[]
  for b in bases
    (q, r) = divrem(n, b, RoundDown)
    n = q
    push!(digits, r)
  end
  digits
end

function from_mixed_digits(digits::AbstractVector{<:Integer}, bases::AbstractVector{<:Integer})
  m = 0; p = 1
  for (d, b) in zip(digits, bases)
    m += d * p
    p *= b
  end
  m
end

# permute the digits in a number. Bases are listed LSB first, and the permutation gives the digit order of the
# output, from LSB to MSB (where 1 is the LSB of the input, n is the MSB of the input)
function permute_mixed_digits(n::Integer, bases::AbstractVector{<:Integer}, perm::Union{Permutation, AbstractVector{<:Integer}})
  digits = to_mixed_digits(n, bases)
  if perm isa Permutation
    perm = perm.data
  else
    @assert isperm(perm)
  end
  digits = digits[perm]; bases = bases[perm]
  from_mixed_digits(digits, bases)
end

function permute_mixed_digits(ns::AbstractVector{<:Integer}, bases::AbstractVector{<:Integer}, perm::Union{Permutation, AbstractVector{<:Integer}})
  [permute_mixed_digits(n, bases, perm) for n in ns]
end

# TODO document the correct intuition for what this does (it's easy to get certain dualities flipped)
function permute_mixed_digits(bases::AbstractVector{<:Integer}, perm::Union{Permutation, AbstractVector{<:Integer}})
  Permutation([1 + permute_mixed_digits(n-1, bases, perm) for n in 1:prod(bases)])
end

