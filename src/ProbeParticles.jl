export ProbeParticleSystem, AbstractProbeParticleSystem

# Abstract supertype (mirrors AbstractVirtualParticleSystem/
# AbstractGhostParticleSystem) so DeviceViews.jl's isbits device-view
# counterpart (DeviceProbeSystem) can share every pfn_contribution dispatch
# written against ProbeParticleSystem without touching those signatures.
abstract type AbstractProbeParticleSystem{T<:AbstractFloat, ND} <: AbstractParticleSystem{T, ND} end

"""
    ProbeParticleSystem{T,ND,EX,UPD,MT,VA,SA,IA}

Passive observer particle system that takes measurements only at save cadence.

Owns probe positions `x`, stable original indices `id`, a kernel-weight
accumulator `w_sum`, and an `extras` NamedTuple of per-probe accumulator arrays.

The per-particle array fields (`x`, `id`, `w_sum`) are generic over container
type (`VA`/`IA`/`SA`) — this is what lets `Adapt.adapt(CuArray, probe)`
(defined below) produce a device-resident probe. `extras` arrays are adapted
element-for-element too. Like every other system in this codebase, the
keyword constructors below *always* build plain `Vector`s regardless of any
argument's own array type (e.g. passing an already GPU-resident
`mirror_target` does **not** make the new probe GPU-resident — `id`/`w_sum`
would still come out `Vector`, giving a broken mixed-backend object): GPU
residency is only ever reached via `Adapt.adapt` on a fully-constructed,
CPU-resident probe, never by feeding device arrays into a constructor.

- `prescribed_v`: constant velocity applied every timestep to advance probe positions.
- `mirror_target`: when set, probe positions are overwritten from `mirror_target.x`
  at each measurement (just before the probe sweep).

  `mirror_target` is adapted independently by `Adapt.adapt_structure` (like
  every other field) — if a driver script needs the probe's mirror target to
  *alias* the same GPU-resident system used elsewhere (so the mirror step
  tracks the live simulation, not a frozen snapshot), build the probe
  CPU-side as usual, referencing the CPU mirror target, then adapt the probe
  as one unit and pull the canonical GPU-resident copy back out via
  `getfield(probe, :mirror_target)` — the same self-referencing idiom
  `GhostParticleSystem` uses (see its docstring). Adapting the probe and its
  mirror target separately gives two independent, non-aliased copies.

Construct with explicit positions:

    probe = ProbeParticleSystem("probes", positions; extras=(nbr=zeros(n),))

Or mirroring all particles of an existing system:

    probe = ProbeParticleSystem("probes", source_ps; extras=(nbr=zeros(source_ps.n),))
"""
struct ProbeParticleSystem{T<:AbstractFloat,ND,EX<:NamedTuple,UPD<:Tuple,MT,
                            VA<:AbstractVector{SVector{ND,T}}, SA<:AbstractVector{T}, IA<:AbstractVector{Int}} <: AbstractProbeParticleSystem{T,ND}
    name::String
    n::Int
    x::VA
    id::IA
    w_sum::SA
    extras::EX
    state_updater::UPD
    prescribed_v::SVector{ND,T}
    mirror_target::MT
    _print_fields::Vector{Symbol}
    function ProbeParticleSystem{T,ND,EX,UPD,MT,VA,SA,IA}(args...) where {T,ND,EX,UPD,MT,VA,SA,IA}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T,ND,EX,UPD,MT,VA,SA,IA}(args...)
    end
end

# Generic positional constructor: infers T, ND, EX, UPD, MT, VA, SA, IA from
# the arguments' own concrete types. Used directly by `Adapt.adapt_structure`
# (below) to rebuild the struct from adapted (e.g. device) arrays; the keyword
# constructors further below are the normal user-facing entry point.
function ProbeParticleSystem(
    name::AbstractString, n::Integer, x::VA, id::IA, w_sum::SA,
    extras::EX, state_updater::UPD, prescribed_v::SVector{ND,T}, mirror_target::MT,
    print_fields::Vector{Symbol},
) where {T<:AbstractFloat, ND, EX<:NamedTuple, UPD<:Tuple, MT,
         VA<:AbstractVector{SVector{ND,T}}, SA<:AbstractVector{T}, IA<:AbstractVector{Int}}
    ProbeParticleSystem{T,ND,EX,UPD,MT,VA,SA,IA}(
        String(name), Int(n), x, id, w_sum, extras, state_updater, prescribed_v,
        mirror_target, print_fields)
end

# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

function ProbeParticleSystem(
    name::AbstractString,
    positions::AbstractVector;
    extras::NamedTuple        = NamedTuple(),
    state_updater             = (),
    prescribed_v              = nothing,
    dtype::Type{<:AbstractFloat} = Float64,
)
    isempty(positions) && throw(ArgumentError("positions must not be empty"))
    T  = dtype
    ND = length(first(positions))
    n  = length(positions)
    x  = [SVector{ND,T}(p) for p in positions]
    _probe_inner(name, n, ND, T, x, extras, state_updater, prescribed_v, nothing)
end

function ProbeParticleSystem(
    name::AbstractString,
    mirror_target::AbstractParticleSystem{T,ND};
    extras::NamedTuple = NamedTuple(),
    state_updater      = (),
    prescribed_v       = nothing,
) where {T,ND}
    n = mirror_target.n
    x = copy(getfield(mirror_target, :x))
    _probe_inner(name, n, ND, T, x, extras, state_updater, prescribed_v, mirror_target)
end

function _probe_inner(name, n, ND, T, x, extras, state_updater, prescribed_v, mirror_target)
    EX = typeof(extras)
    for fname in fieldnames(EX)
        arr = getfield(extras, fname)
        length(arr) == n || throw(ArgumentError(
            "extras.$fname has length $(length(arr)) but n=$n"))
    end
    updaters = state_updater isa Tuple ? state_updater : (state_updater,)
    pv = prescribed_v === nothing ? zero(SVector{ND,T}) : SVector{ND,T}(prescribed_v)
    ProbeParticleSystem(
        String(name), n, x, collect(1:n), zeros(T, n),
        extras, updaters, pv, mirror_target, Symbol[],
    )
end

# ---------------------------------------------------------------------------
# Adapt.jl support — see docstring above for the mirror_target aliasing caveat.
# ---------------------------------------------------------------------------

function Adapt.adapt_structure(to, probe::ProbeParticleSystem)
    mt = getfield(probe, :mirror_target)
    ProbeParticleSystem(
        getfield(probe, :name), probe.n,
        Adapt.adapt(to, getfield(probe, :x)),
        Adapt.adapt(to, getfield(probe, :id)),
        Adapt.adapt(to, getfield(probe, :w_sum)),
        map(a -> Adapt.adapt(to, a), getfield(probe, :extras)),
        getfield(probe, :state_updater), getfield(probe, :prescribed_v),
        mt === nothing ? nothing : Adapt.adapt(to, mt),
        getfield(probe, :_print_fields),
    )
end

# ---------------------------------------------------------------------------
# getproperty
# ---------------------------------------------------------------------------

@inline function Base.getproperty(
    probe::ProbeParticleSystem{T,ND,EX,UPD,MT}, s::Symbol
) where {T,ND,EX,UPD,MT}
    s === :ndims && return ND
    s in (:name, :n, :x, :id, :w_sum, :extras, :state_updater, :prescribed_v,
          :mirror_target, :_print_fields) && return getfield(probe, s)
    s in fieldnames(EX) && return getproperty(getfield(probe, :extras), s)
    MT !== Nothing && return getproperty(getfield(probe, :mirror_target), s)
    error("ProbeParticleSystem has no field $s")
end

# ---------------------------------------------------------------------------
# add_print_field! override — also accepts extras field names
# ---------------------------------------------------------------------------

function add_print_field!(probe::ProbeParticleSystem{T,ND,EX}, field::Symbol) where {T,ND,EX}
    field in fieldnames(EX) || hasfield(typeof(probe), field) ||
        throw(ArgumentError("unknown field :$field"))
    field ∉ probe._print_fields && push!(probe._print_fields, field)
    return probe
end

# ---------------------------------------------------------------------------
# Accumulator helpers
# ---------------------------------------------------------------------------

# x first (sort key), then id (survives cell sorts), w_sum, then all extras.
function _particle_arrays(probe::ProbeParticleSystem)
    (getfield(probe, :x),
     getfield(probe, :id),
     getfield(probe, :w_sum),
     values(getfield(probe, :extras))...)
end

function auto_zero_probe!(probe::ProbeParticleSystem{T,ND,EX}) where {T,ND,EX}
    fill!(getfield(probe, :w_sum), zero(T))
    for fname in fieldnames(EX)
        arr = getproperty(getfield(probe, :extras), fname)
        fill!(arr, zero(eltype(arr)))
    end
end

function _sort_probe_by_id!(probe::ProbeParticleSystem, perm_buf, scratch)
    n = probe.n
    n <= 1 && return
    length(perm_buf) < n && resize!(perm_buf, n)
    _resize_scratches!(scratch, n)
    perm_view = view(perm_buf, 1:n)
    sortperm!(perm_view, getfield(probe, :id))
    _apply_perms!(_particle_arrays(probe), scratch, perm_view, n)
end

# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

write_h5(probe::ProbeParticleSystem, target::AbstractString) =
    h5open(target, "w") do f; write_h5(probe, f) end

read_h5!(probe::ProbeParticleSystem, source::AbstractString) =
    (h5open(source, "r") do f; read_h5!(probe, f) end; probe)

function write_h5(
    probe::ProbeParticleSystem{T,ND,EX,UPD,MT},
    group::Union{HDF5.File, HDF5.Group},
) where {T,ND,EX,UPD,MT}
    n = probe.n
    HDF5.attrs(group)["n"]     = n
    HDF5.attrs(group)["ndims"] = ND
    if n > 0
        group["x"]  = reinterpret(reshape, T, getfield(probe, :x))
        group["id"] = getfield(probe, :id)
        for fname in fieldnames(EX)
            arr = getproperty(getfield(probe, :extras), fname)
            if eltype(arr) <: SVector
                group[string(fname)] = reinterpret(reshape, eltype(eltype(arr)), arr)
            else
                group[string(fname)] = arr
            end
        end
    end
end

function read_h5!(
    probe::ProbeParticleSystem{T,ND,EX,UPD,MT},
    group::Union{HDF5.File, HDF5.Group},
) where {T,ND,EX,UPD,MT}
    saved_n = Int(HDF5.attrs(group)["n"][])
    saved_n == probe.n || throw(ArgumentError(
        "Saved probe count ($saved_n) does not match probe.n=$(probe.n)"))

    if haskey(group, "x")
        x_data = group["x"][]
        getfield(probe, :x) .= reinterpret(reshape, eltype(probe.x), x_data)
    end

    # File was written in id-order; reset id to 1:n
    getfield(probe, :id) .= 1:probe.n

    for fname in fieldnames(EX)
        s = string(fname)
        haskey(group, s) || continue
        arr = getproperty(getfield(probe, :extras), fname)
        data = group[s][]
        if arr isa AbstractVector{<:SVector}
            arr .= reinterpret(reshape, eltype(arr), data)
        else
            arr .= data
        end
    end
    return probe
end
