
using TikzPictures

#TODO: put headings over each stage describing it concisely? at least for the SDF stages...
#TODO: support more stage types

function tikz_input_stage(io::IO, stage::Int, indexes::AbstractVector{Int})
  for (i, j) in enumerate(indexes)
    if i == 1
      out_opts=""
    else
      out_opts="below=of stage-out-$stage-$(i-1)"
    end

    println(io, "\\coordinate [$out_opts] (stage-out-$stage-$i)")
    println(io, "  node[left=of stage-out-$stage-$i] (stage-in-$stage-$i) {\$\\chi_{$j}\$};")
    println(io, "\\path (stage-in-$stage-$i) edge (stage-out-$stage-$i);")
  end
end

function tikz_output_stage(io::IO, prev_stage::Int, indexes::AbstractVector{Int})
  prev = prev_stage; cur = prev - 1

  for (i, j) in enumerate(indexes)
    println(io, "\\node also[alias=stage-in-$cur-$i] (stage-out-$prev-$i)")
    println(io, "  node [right=of stage-in-$cur-$i] (stage-out-$cur-$i) {\$X_{$j}\$};")
    println(io, "\\path (stage-in-$cur-$i) edge (stage-out-$cur-$i);")
  end
end

function Base.show(io::IO, ::MIME"text/tikz", r::Reorder)
  n = get(io, :fft_transform_length, length(r.perm))
  w = max(1, 0.5log2(length(r.perm)) - 1)
  prev = get(io, :fft_prev_stage, 0); cur=prev+1

  # declare nodes
  for i in 1:n
    println(io, "\\node also[alias=stage-in-$cur-$i] (stage-out-$prev-$i)")
    println(io, "  coordinate [right=$(w)cm of stage-in-$cur-$i] (stage-out-$cur-$i);")
  end

  # draw edges
  for (i,j) in enumerate(permutation(r, n))
    print(io, "\\path (stage-in-$cur-$i) edge[color=black!30] (stage-out-$cur-$j);")
  end
end

function Base.show(io::IO, ::MIME"text/tikz", s::SDF)
  radix = s.butterfly.radix
  w = max(1, 0.2 * (log(radix) + radix * log(s.depth)))
  n = get(io, :fft_transform_length, radix * s.depth)
  prev = get(io, :fft_prev_stage, 0); cur=prev+1

  # declare nodes
  for i in 1:n
    println(io, "\\node also[alias=stage-in-$cur-$i] (stage-out-$prev-$i)")
    println(io, "  coordinate[right=$(w)cm of stage-in-$cur-$i] (stage-out-$cur-$i) {};")
  end

  # draw butterflies
  for g in sdf_groups(s, n)
    # TODO: is this how we really want to draw butterflies? It gets really nasty for higher radixes
    for i in g
      for j in g
        print(io, "\\path (stage-in-$cur-$i) edge (stage-out-$cur-$j);")
      end
    end
  end
end

function Base.show(io::IO, ::MIME"text/tikz", tw::Twiddle)
  n = get(io, :fft_transform_length, length(tw.indexes))
  prev = get(io, :fft_prev_stage, 0); cur=prev+1

  # declare nodes
  for i in 1:n
    println(io, "\\node also[alias=stage-in-$cur-$i] (stage-out-$prev-$i)")
    println(io, "  coordinate[right=0.5cm of stage-in-$cur-$i] (stage-out-$cur-$i) {};")
  end

  println(io, "\\begin{scope}[style={inner sep=0pt,outer sep=0pt,font=\\tiny}]")
  order = tw.order
  for (i, index) in enumerate(twiddle_indexes(tw, n))
    #println(io, "\\path (stage-in-$cur-$i) edge[\"\$\\omega^{$index}_{$order}\$\"] (stage-out-$cur-$i);")
    node_shape  = (index == 0 ? "" : "contact")
    # TODO: indicate trivial rotations differently?
    tw_text     = (index == 0 ? "" : "\$\\omega^{$index}_{$order}\$")
    println(io, "\\draw (stage-in-$cur-$i) -- (stage-out-$cur-$i) node[$node_shape, pos=0.35, label=60:$tw_text] {};")
  end
  println(io, "\\end{scope}")
end

function Base.show(io::IO, mime::MIME"text/tikz", p::FFTPlan)
  n = get(io, :fft_transform_length, missing)
  if ismissing(n)
    n = transform_length(p)
    io = IOContext(io, :fft_transform_length => n)
  end

  prev_stage = get(io, :fft_prev_stage, missing)
  if ismissing(prev_stage)
    in_order = get(io, :fft_input_order, missing)
    tikz_input_stage(io, 0, ismissing(in_order) ? input_order(p) : in_order)
    prev_stage = 0
  end

  for (i, s) in enumerate(p.stages)
    show(IOContext(io, :fft_prev_stage => prev_stage), mime, s)
    prev_stage += 1
  end

  out_order = get(io, :fft_output_order, missing)
  tikz_output_stage(io, prev_stage, ismissing(out_order) ? output_order(p) : out_order)
end

function Base.convert(
    ::Type{TikzPicture},
    s::Union{FFTStage, FFTPlan},
    n::Union{Missing,Int}=missing;

    in_order::Union{Missing,AbstractVector{Int}}=missing,
    out_order::Union{Missing,AbstractVector{Int}}=missing)
  io = IOBuffer()
  ctx = Pair[]
  if !ismissing(n)
    push!(ctx, :fft_transform_length => n)
  end

  if s isa FFTPlan
    push!(ctx, :fft_input_order => ismissing(in_order) ? input_order(s) : in_order)
    push!(ctx, :fft_output_order => ismissing(out_order) ? output_order(s) : out_order)
  end

  show(IOContext(io, ctx...), MIME"text/tikz"(), s)
  TikzPicture(String(take!(io));
    preamble="\\usetikzlibrary{positioning,quotes,circuits.ee.IEC}",
    options="thick, node distance=0.5cm, circuit ee IEC,\n"
           *"box/.style={\n"
           *"  draw, align=center, shape=rectangle, minimum width=1.5cm, minimum height=4cm,\n"
           *"  append after command={\n"
           *"    \\foreach \\side in {east,west} {\n"
           *"      \\foreach \\i in {1,...,#1} {\n"
           *"        (\\tikzlastnode.north \\side) edge[draw=none, line to]\n"
           *"          coordinate[pos=(\\i-.5)/(#1)] (\\tikzlastnode-\\i-\\side) (\\tikzlastnode.south \\side)\n"
           *"  }}}}")
end

