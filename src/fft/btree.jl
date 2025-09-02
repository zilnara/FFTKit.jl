
struct Leaf{V}
  val::V
end

struct BNode{V}
  l::Union{Leaf{V}, BNode{V}}
  r::Union{Leaf{V}, BNode{V}}
end

const BTree{V} = Union{Leaf{V}, BNode{V}}

function BNode(l::BTree{L}, r::BTree{R}) where {L, R}
  T = promote_type(L,R)
  BNode{T}(l, r)
end

Base.convert(::Type{Leaf{T}}, t::Leaf) where {T} = Leaf{T}(convert(T, t.val))
Base.convert(::Type{BNode{T}}, t::BNode) where {T} = BNode{T}(convert(BTree{T}, t.l), convert(BTree{T}, t.r))

Base.convert(::Type{Union{L, N}}, t::Leaf) where {L <: Leaf, N <: BNode} = convert(L, t)
Base.convert(::Type{Union{L, N}}, t::BNode) where {L <: Leaf, N <: BNode} = convert(N, t)

Base.eltype(t::BTree{T}) where T = T
Base.length(t::BTree) = (t isa Leaf) ? 1 : length(t.l) + length(t.r)

Base.iterate(t::Leaf) = (t.val, nothing)
function Base.iterate(t::BNode)
  (x, l_state) = iterate(t.l)
  (x, (l_state, t.r))
end

Base.iterate(::BTree, nothing) = nothing
function Base.iterate(root::BTree, t::Tuple)
  l_state, r = t
  if isnothing(l_state)
    iterate(r)
  else
    x, l_state = iterate(root, l_state)
    (x, (l_state, r))
  end
end

Base.reverse(t::Leaf) = t
Base.reverse(t::BNode{T}) where T = BNode{T}(reverse(t.r), reverse(t.l))

# a slightly nicer constructor
BTree(x) = Leaf(x)
BTree(x,y) = BNode(BTree(x), BTree(y))
BTree(t::Tuple) = BTree(t...)
BTree(t::BTree) = t

fmt_compact(x::Leaf; top::Bool=true) = top ? "BTree($(x.val))" : string(x.val)
fmt_compact(x::BNode; top::Bool=true) = "$(top ? "BTree" : "")($(fmt_compact(x.l; top=false)), $(fmt_compact(x.r; top=false)))"

# call function 'f' once for each node in the tree, passing four fragments:
# The node's left and right subtrees, and the "exterior" left and right context trees (pivoted to be seen relative to the node)
function foreach_node_with_context(f, t::BTree{T}) where T
  stack = Any[(nothing, t, nothing)]
  while !isempty(stack)
    (l, t, r) = pop!(stack)
    if t isa BNode
      push!(stack, (isnothing(l) ? t.l : BNode(l, t.l), t.r, r))
      push!(stack, (l, t.l, isnothing(r) ? t.r : BNode(t.r, r)))

      f(t.l, t.r, context_right=r, context_left=l)
    end
  end
  ;
end

function enumerate_btrees(leaves::AbstractVector{T}) where T
  n = length(leaves)
  if n <= 2
    [BTree(leaves...)]
  else
    [BNode(l, r) for i in 1:n-1 for l in enumerate_btrees(leaves[1:i]) for r in enumerate_btrees(leaves[i+1:end])]
  end
end

