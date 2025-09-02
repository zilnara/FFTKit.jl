
struct Rotation <: Number
  index::Rational

  Rotation(index::Rational) = new(mod(index + 1//2, 1) - 1//2)
  Rotation(::Integer) = new(0//1)
end

Base.convert(::Type{Rotation}, x::Real) = isone(x) ? Rotation(0) : isone(abs(x)) ? Rotation(1//2) : error("cannot convert $x to a Rotation")

Base.convert(::Type{Complex}, r::Rotation) = convert(ComplexF64, r)
Base.convert(T::Type{<:Complex}, r::Rotation) = T(r)
Base.convert(Tq::Type{<:Quantity{T}}, r::Rotation) where {T <: Complex} = Tq(convert(T, r))

OMEGA_VARS_CACHE = Dict{Int, Num}()
omega_var(n::Int) = get!(() -> Symbolics.variable("ω_$n"), OMEGA_VARS_CACHE, n)
Base.convert(::Type{Num}, r::Rotation) = Num(r)
Symbolics.Num(r::Rotation) = omega_var(denominator(r.index)) ^ mod(-numerator(r.index), denominator(r.index))

Complex(r::Rotation) = ComplexF64(r)
Complex{T}(r::Rotation) where {T<:AbstractFloat} = refine_root_of_unity(denominator(r.index), cispi(2r.index), T_out=T)
Complex{BigFloat}(r::Rotation; precision::Int=precision(BigFloat)) = refine_root_of_unity(denominator(r.index), cispi(2r.index), T_out=BigFloat, out_prec=precision)

Base.promote_rule(::Type{Rotation}, T::Type{<:Number}) = promote_rule(ComplexF64, T)
Base.promote_rule(T::Type{<:Number}, ::Type{Rotation}) = promote_rule(T, ComplexF64)

Base.zero(::Type{Rotation}) = error("The type $(Rotation) cannot represent the number zero")
Base.one(::Type{Rotation}) = Rotation(0//1)

Base.:+(x::Rotation, y::Rotation) = convert(Complex, x) + convert(Complex, y)
Base.:-(x::Rotation, y::Rotation) = convert(Complex, x) - convert(Complex, y)

Base.:*(x::Rotation, y::Rotation) = Rotation(x.index + y.index)
Base.:*(x::Rotation, y::Number) = convert(Complex, x) * y
Base.:*(x::Number, y::Rotation) = x * convert(Complex, y)
Base.:*(x::Num, y::Rotation) = x * Num(y)

Base.:/(x::Rotation, y::Rotation) = Rotation(x.index - y.index)
Base.:/(x::Rotation, y::Number) = convert(Complex, x) / y
Base.:/(x::Number, y::Rotation) = x / convert(Complex, y)

Base.:^(x::Rotation, y::Union{Int,Rational}) = Rotation(y * x.index)

Base.inv(x::Rotation) = Rotation(-x.index)
Base.:-(x::Rotation) = Rotation(x.index + 1//2)

Base.abs(x::Rotation) = 1
Base.conj(x::Rotation) = Rotation(-x.index)
Base.sqrt(x::Rotation) = Rotation(x.index / 2)
Base.cbrt(x::Rotation) = Rotation(x.index / 3)

order(x::Rotation) = denominator(x.index)

istrivial(x::Rotation) = order(x) in [1,2,4]

Base.isone(x::Rotation) = order(x) == 1
Base.iszero(::Rotation) = false
Base.isinteger(x::Rotation) = order(x) <= 2
Base.isreal(x::Rotation) = order(x) <= 2
Base.isfinite(::Rotation) = true
Base.iseven(::Rotation) = false
Base.isodd(r::Rotation) = isinteger(r)
Base.isnan(::Rotation) = false
Base.ispow2(r::Rotation) = isone(r)
Base.isinf(::Rotation) = false
Base.issubnormal(::Rotation) = false

function cost(x::Rotation, n::Int=1)
  d = order(x)
  if d <= 2
    Cost()
  elseif d == 4
    Cost(TRIVIAL_CMULT => n)
  else
    Cost(COMPLEX_MULT => n)
  end
end

function refine_root_of_unity(
    n::Int,
    x::Complex{T};

    T_out::Type = T,
    out_prec::Int = precision(T_out),
    work_prec::Int = out_prec + 8,
    eps_scale::Real = 0.25,
    max_steps::Int = out_prec < 1024 ? 15 : 4 + floor(Int, log2(out_prec))
) where T <: AbstractFloat
  # TODO: cache results?

  # refine a Nth root of unity using Newton-Raphson on the equation x^n - 1 = 0,
  # keeping the individual terms as a list to sum every time in magnitude order for numerical accuracy
  #   TODO: this probably isn't really necessary, this converges EXTREMELY quickly either way
  cutoff = setprecision(out_prec) do
    eps_scale * eps(BigFloat)
  end

  converged = false
  setprecision(work_prec) do
    x_re = BigFloat[x.re]
    x_im = BigFloat[x.im]
    if T != BigFloat
      x = Complex(x_re[1], x_im[1])
    end

    f(x) = x^n-1; df(x) = n*x^(n-1)
    step(x) = - f(x) / df(x)
    for i in 1:max_steps
      prev = x
      dx = step(x)
      push!(x_re, dx.re); sort!(x_re, by=abs)
      push!(x_im, dx.im); sort!(x_im, by=abs)

      x = Complex{BigFloat}(sum(x_re), sum(x_im))
      if abs(dx) < cutoff
       #  @info "converged after $i iterations"  # x=Complex.(x_re, x_im) dx cutoff out_prec work_prec
        converged = true
        break
      end
    end

    if !converged
      @warn "refine_root_of_unity did not converge in $max_steps steps" x_re x_im out_prec work_prec
    end

    if T_out == BigFloat && out_prec != work_prec
      x_re = BigFloat(x.re, precision=out_prec)
      x_im = BigFloat(x.im, precision=out_prec)
      Complex{BigFloat}(x_re, x_im)
    elseif T_out != BigFloat
      Complex{T_out}(x)
    else
      x
    end
  end
end

function rewrite_rotations(n::Int)
  rules = Dict{Num, Num}()

  omega_n = omega_var(n)
  for p in divisors(n)
    omega_p = omega_var(p)

    if p < n
      q = mod(n ÷ p, n)
      rules[omega_p] = omega_n ^ q
    end
  end
  rules
end

rewrite_rotations(n::Int, expr::Num) = substitute(expr, rewrite_rotations(n))
rewrite_rotations(n::Int, exprs::AbstractArray{Num}) = [substitute(expr, rewrite_rotations(n)) for expr in exprs]


