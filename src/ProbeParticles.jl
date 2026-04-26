export ProbeParticleSystem

"""
    ProbeParticleSystem{T,ND,EX,UPD,MT}

Passive observer particle system that takes measurements only at save cadence.

Owns probe positions `x`, stable original indices `id`, a kernel-weight
accumulator `w_sum`, and an `extras` NamedTuple of per-probe accumulator arrays.

- `prescribed_v`: constant velocity applied every timestep to advance probe positions.
- `mirror_target`: when set, probe positions are overwritten from `mirror_target.x`
  at each measurement (just before the probe sweep).

Construct with explicit positions:

    probe = ProbeParticleSystem("probes", positions; extras=(nbr=zeros(n),))

Or mirroring all particles of an existing system:

    probe = ProbeParticleSystem("probes", source_ps; extras=(nbr=zeros(source_ps.n),))
"""
struct ProbeParticleSystem{T<:AbstractFloat,ND,EX<:NamedTuple,UPD<:Tuple,MT} <: AbstractParticleSystem{T,ND}
    name::String
    n::Int
    x::Vector{SVector{ND,T}}
    id::Vector{Int}
    w_sum::Vector{T}
    extras::EX
    state_updater::UPD
    prescribed_v::SVector{ND,T}
    mirror_target::MT
    _print_fields::Vector{Symbol}
    function ProbeParticleSystem{T,ND,EX,UPD,MT}(args...) where {T,ND,EX,UPD,MT}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T,ND,EX,UPD,MT}(args...)
    end
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
    MT  = typeof(mirror_target)
    UPD = typeof(updaters)
    ProbeParticleSystem{T,ND,EX,UPD,MT}(
        String(name), n, x, collect(1:n), zeros(T, n),
        extras, updaters, pv, mirror_target, Symbol[],
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
