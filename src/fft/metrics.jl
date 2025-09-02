
struct Cost <: AbstractDict{Symbol, Number}
  costs :: Dict{Symbol, Number}

  function Cost(pairs::Pair{Symbol, <:Number}...)
    new(Dict{Symbol, Number}(pairs...))
  end

  function Cost(kv)
    new(Dict{Symbol, Number}(kv))
  end
end

# Keys for "Cost" dictionary:
const COMPLEX_SUM     = :csum             # Complex + Complex
const COMPLEX_MULT    = :cmult            # Complex * Complex
const COMPLEX_DIV     = :cdiv             # Complex / Complex
const COMPLEX_SQRT    = :csqrt            # sqrt(Complex)
const COMPLEX_SCALE   = :cscale           # Real * Complex
const TRIVIAL_CMULT   = :trivial_cmult    # Complex * {im, -im}
const TRIVIAL_CSCALE  = :trivial_cscale   # Complex * 2^N
const UNKNOWN_COST    = :unknown_cost

function cost end

Base.keys(c::Cost) = keys(c.costs)
Base.values(c::Cost) = values(c.costs)
Base.iterate(c::Cost) = iterate(c.costs)
Base.iterate(c::Cost, s) = iterate(c.costs, s)
Base.length(c::Cost) = length(c.costs)
Base.empty(::Cost, ::Type{Symbol}, ::Type{<:Number}) = Cost()
Base.empty!(c::Cost) = empty!(c.costs)
Base.get(c::Cost, args...) = get(c.costs, args...)
Base.get!(c::Cost, args...) = get!(c.costs, args...)
Base.get!(f, c::Cost, args...) = get!(f, c.costs, args...)
Base.getindex(c::Cost, args...) = getindex(c.costs, args...)
Base.setindex!(c::Cost, args...) = setindex!(c.costs, args...)
Base.delete!(c::Cost, args...) = delete!(c.costs, args...)

Base.zero(::Type{Cost}) = Cost()

Base.:+(c1::Cost, c2::Cost) = Cost(mergewith(+, c1.costs, c2.costs))

Base.:*(x::Cost, y::Number) = Cost(k => v * y for (k,v) in x.costs)
Base.:*(x::Number, y::Cost) = Cost(k => x * v for (k,v) in y.costs)

Base.:/(x::Cost, y::Number) = Cost(k => v/y for (k,v) in x.costs)
Base.div(x::Cost, y::Integer, r::RoundingMode=RoundToZero) = Cost(k => div(v, y, r) for (k,v) in x.costs)

Base.round(c::Cost, r::RoundingMode=RoundNearest) = Cost(k => round(v, r) for (k,v) in c.costs)
Base.round(T::Type{<:Number}, c::Cost, r::RoundingMode=RoundNearest) = Cost(k => round(T, v, r) for (k,v) in c.costs)
