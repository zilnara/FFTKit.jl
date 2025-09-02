
@enum FFTMode forward backward inverse
export FFTMode, forward, backward, inverse

# A common misconception is that DIF and DIT specify the structure of the stage delays.
# It actually refers to the structure of the rotation stages, and is independent of the
# structure of the delays.
@enum FFTTwiddleOrder dif dit
export FFTTwiddleOrder, dif, dit

include("butterflies/r2.jl");  export r2, r2!;
include("butterflies/r3.jl");  export r3, r3!;
include("butterflies/r4.jl");  export r4, r4!;
include("butterflies/r5.jl");  export r5, r5!;

