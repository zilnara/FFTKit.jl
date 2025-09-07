
include("algebraic/reorder.jl")
include("algebraic/sdf.jl")
include("algebraic/twiddle.jl")
include("algebraic/conj.jl")
include("algebraic/scale.jl")
include("algebraic/adjoint_r2.jl")
include("algebraic/plan.jl")

include("algebraic/rewrite.jl")
include("algebraic/test.jl")

# TODO: figure out better way to package this that avoids bringing in the stale dependencies
# include("algebraic/tikz.jl")

# TODO: support modeling fixed-point with scaling schedules
# TODO: support extracting quantized twiddle tables
# TODO: functions to convert any of these elements into matrix representation

