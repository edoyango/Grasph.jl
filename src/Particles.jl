export AbstractParticleSystem,
       BasicParticleSystem, FluidParticleSystem, StressParticleSystem,
       ElastoPlasticParticleSystem, VirtualParticleSystem,
       update_state!, write_h5, read_h5!, print_summary,
       add_print_field!, remove_print_field!

# ---------------------------------------------------------------------------
# Abstract type
# ---------------------------------------------------------------------------

abstract type AbstractParticleSystem{T<:AbstractFloat, ND} end

# ---------------------------------------------------------------------------
# getproperty override — shared by all concrete types
# ---------------------------------------------------------------------------

@inline function Base.getproperty(ps::AbstractParticleSystem{T,ND}, s::Symbol) where {T,ND}
    s === :ndims && return ND
    return getfield(ps, s)
end

# ---------------------------------------------------------------------------
# Default conjugate pairs for time integration
# ---------------------------------------------------------------------------

const _DEFAULT_PAIRS = ((Val(:v), Val(:dvdt)), (Val(:rho), Val(:drhodt)))

# ---------------------------------------------------------------------------
# BasicParticleSystem
# ---------------------------------------------------------------------------

"""
    BasicParticleSystem{T, ND, PAIRS<:Tuple} <: AbstractParticleSystem{T, ND}

A particle system with position, velocity, density, and their derivatives.
No pressure field.
"""
struct BasicParticleSystem{T<:AbstractFloat, ND, PAIRS<:Tuple, UPD<:Tuple} <: AbstractParticleSystem{T, ND}
    name::String
    n::Int
    x::Vector{SVector{ND,T}}
    v::Vector{SVector{ND,T}}
    v_adjustment::Vector{SVector{ND,T}}
    rho::Vector{T}
    dvdt::Vector{SVector{ND,T}}
    drhodt::Vector{T}
    mass::T
    c::T
    source_v::SVector{ND,T}
    source_rho::T
    _print_fields::Vector{Symbol}
    state_updater::UPD
    pairs::PAIRS
    function BasicParticleSystem{T, ND, PAIRS, UPD}(args...) where {T, ND, PAIRS, UPD}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T, ND, PAIRS, UPD}(args...)
    end
end

function BasicParticleSystem(
    name::AbstractString,
    n::Integer,
    ndims::Integer,
    mass::Real,
    c::Real;
    dtype::Type{<:AbstractFloat} = Float64,
    source_v::AbstractVector{<:Real} = zeros(dtype, Int(ndims)),
    source_rho::Real = 0.0,
    state_updater = (),
)
    n = Int(n); ndims = Int(ndims)
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    ndims > 0 || throw(ArgumentError("ndims must be positive, got $ndims"))
    length(source_v) == ndims || throw(ArgumentError(
        "length(source_v) = $(length(source_v)) must equal ndims = $ndims"))

    T = dtype

    x      = Vector{SVector{ndims,T}}(undef, n)
    v      = Vector{SVector{ndims,T}}(undef, n)
    v_adjustment = fill(zero(SVector{ndims,T}), n)
    rho    = Vector{T}(undef, n)
    dvdt   = fill(zero(SVector{ndims,T}), n)
    drhodt = zeros(T, n)

    state_updaters = state_updater isa Tuple ? state_updater : (state_updater,)
    _check_functors_eltype(state_updaters, T, "state updater")

    BasicParticleSystem{T, ndims, typeof(_DEFAULT_PAIRS), typeof(state_updaters)}(
        String(name), n,
        x, v, v_adjustment, rho, dvdt, drhodt,
        T(mass), T(c),
        SVector{ndims,T}(source_v),
        T(source_rho),
        Symbol[],
        state_updaters,
        _DEFAULT_PAIRS,
    )
end

function _read_h5_metadata(path::AbstractString, name::AbstractString, dtype::Type{T}) where {T<:AbstractFloat}
    local n, ndims, mass, c, ns
    h5open(path, "r") do f
        group = haskey(f, name) ? f[name] : f
        n = Int(HDF5.attrs(group)["n"][])
        ndims = Int(HDF5.attrs(group)["ndims"][])
        mass = T(HDF5.attrs(group)["mass"][])
        c = T(HDF5.attrs(group)["c"][])
        
        ns = nothing
        if haskey(group, "stress")
            stress_dset = group["stress"]
            ns = size(stress_dset)[1]
        end
    end
    return n, ndims, mass, c, ns
end

function BasicParticleSystem(
    path::AbstractString,
    name::AbstractString;
    source_v::Union{Nothing,AbstractVector{<:Real}} = nothing,
    source_rho::Real = 0.0,
    state_updater = (),
    dtype::Type{T} = Float64,
) where {T<:AbstractFloat}
    n, ndims, mass, c, _ = _read_h5_metadata(path, name, T)

    sv = source_v === nothing ? zeros(T, ndims) : source_v

    ps = BasicParticleSystem(name, n, ndims, mass, c;
                             source_v=sv, source_rho=source_rho,
                             state_updater=state_updater)
    read_h5!(ps, path)
    return ps
end

# ---------------------------------------------------------------------------
# FluidParticleSystem
# ---------------------------------------------------------------------------

"""
    FluidParticleSystem{T, ND, PAIRS<:Tuple} <: AbstractParticleSystem{T, ND}

A particle system with all BasicParticleSystem fields plus pressure `p`.
"""
struct FluidParticleSystem{T<:AbstractFloat, ND, PAIRS<:Tuple, UPD<:Tuple} <: AbstractParticleSystem{T, ND}
    name::String
    n::Int
    x::Vector{SVector{ND,T}}
    v::Vector{SVector{ND,T}}
    v_adjustment::Vector{SVector{ND,T}}
    rho::Vector{T}
    dvdt::Vector{SVector{ND,T}}
    drhodt::Vector{T}
    p::Vector{T}
    mass::T
    c::T
    source_v::SVector{ND,T}
    source_rho::T
    _print_fields::Vector{Symbol}
    state_updater::UPD
    pairs::PAIRS
    function FluidParticleSystem{T, ND, PAIRS, UPD}(args...) where {T, ND, PAIRS, UPD}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T, ND, PAIRS, UPD}(args...)
    end
end

function FluidParticleSystem(
    name::AbstractString,
    n::Integer,
    ndims::Integer,
    mass::Real,
    c::Real;
    dtype::Type{<:AbstractFloat} = Float64,
    source_v::AbstractVector{<:Real} = zeros(dtype, Int(ndims)),
    source_rho::Real = 0.0,
    state_updater = (),
)
    n = Int(n); ndims = Int(ndims)
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    ndims > 0 || throw(ArgumentError("ndims must be positive, got $ndims"))
    length(source_v) == ndims || throw(ArgumentError(
        "length(source_v) = $(length(source_v)) must equal ndims = $ndims"))

    T = dtype

    x      = Vector{SVector{ndims,T}}(undef, n)
    v      = Vector{SVector{ndims,T}}(undef, n)
    v_adjustment = fill(zero(SVector{ndims,T}), n)
    rho    = Vector{T}(undef, n)
    dvdt   = fill(zero(SVector{ndims,T}), n)
    drhodt = zeros(T, n)
    p      = zeros(T, n)

    state_updaters = state_updater isa Tuple ? state_updater : (state_updater,)
    _check_functors_eltype(state_updaters, T, "state updater")

    FluidParticleSystem{T, ndims, typeof(_DEFAULT_PAIRS), typeof(state_updaters)}(
        String(name), n,
        x, v, v_adjustment, rho, dvdt, drhodt, p,
        T(mass), T(c),
        SVector{ndims,T}(source_v),
        T(source_rho),
        Symbol[],
        state_updaters,
        _DEFAULT_PAIRS,
    )
end

function FluidParticleSystem(
    path::AbstractString,
    name::AbstractString;
    source_v::Union{Nothing,AbstractVector{<:Real}} = nothing,
    source_rho::Real = 0.0,
    state_updater = (),
    dtype::Type{T} = Float64,
) where {T<:AbstractFloat}
    n, ndims, mass, c, _ = _read_h5_metadata(path, name, T)

    sv = source_v === nothing ? zeros(T, ndims) : source_v

    ps = FluidParticleSystem(name, n, ndims, mass, c;
                             source_v=sv, source_rho=source_rho,
                             state_updater=state_updater)
    read_h5!(ps, path)
    return ps
end

# ---------------------------------------------------------------------------
# StressParticleSystem
# ---------------------------------------------------------------------------

"""
    StressParticleSystem{T, ND, NS, PAIRS<:Tuple} <: AbstractParticleSystem{T, ND}

A particle system with all FluidParticleSystem fields plus `stress` and
`strain_rate` in Voigt notation (NS components, typically 3, 4, or 6).
"""
struct StressParticleSystem{T<:AbstractFloat, ND, NS, PAIRS<:Tuple, UPD<:Tuple} <: AbstractParticleSystem{T, ND}
    name::String
    n::Int
    x::Vector{SVector{ND,T}}
    v::Vector{SVector{ND,T}}
    v_adjustment::Vector{SVector{ND,T}}
    rho::Vector{T}
    dvdt::Vector{SVector{ND,T}}
    drhodt::Vector{T}
    p::Vector{T}
    stress::Vector{SVector{NS,T}}
    strain_rate::Vector{SVector{NS,T}}
    mass::T
    c::T
    source_v::SVector{ND,T}
    source_rho::T
    _print_fields::Vector{Symbol}
    state_updater::UPD
    pairs::PAIRS
    function StressParticleSystem{T, ND, NS, PAIRS, UPD}(args...) where {T, ND, NS, PAIRS, UPD}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        NS isa Int || throw(ArgumentError("NS must be an Int, got $(typeof(NS))"))
        new{T, ND, NS, PAIRS, UPD}(args...)
    end
end

function StressParticleSystem(
    name::AbstractString,
    n::Integer,
    ndims::Integer,
    ns::Integer,
    mass::Real,
    c::Real;
    dtype::Type{<:AbstractFloat} = Float64,
    source_v::AbstractVector{<:Real} = zeros(dtype, Int(ndims)),
    source_rho::Real = 0.0,
    state_updater = (),
)
    n = Int(n); ndims = Int(ndims); ns = Int(ns)
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    ndims > 0 || throw(ArgumentError("ndims must be positive, got $ndims"))
    ns ∈ (3, 4, 6) || throw(ArgumentError("ns must be 3, 4, or 6, got $ns"))
    length(source_v) == ndims || throw(ArgumentError(
        "length(source_v) = $(length(source_v)) must equal ndims = $ndims"))

    T = dtype

    x            = Vector{SVector{ndims,T}}(undef, n)
    v            = Vector{SVector{ndims,T}}(undef, n)
    v_adjustment = fill(zero(SVector{ndims,T}), n)
    rho          = Vector{T}(undef, n)
    dvdt         = fill(zero(SVector{ndims,T}), n)
    drhodt       = zeros(T, n)
    p            = zeros(T, n)
    stress       = fill(zero(SVector{ns,T}), n)
    strain_rate  = fill(zero(SVector{ns,T}), n)

    state_updaters = state_updater isa Tuple ? state_updater : (state_updater,)
    _check_functors_eltype(state_updaters, T, "state updater")

    StressParticleSystem{T, ndims, ns, typeof(_DEFAULT_PAIRS), typeof(state_updaters)}(
        String(name), n,
        x, v, v_adjustment, rho, dvdt, drhodt, p, stress, strain_rate,
        T(mass), T(c),
        SVector{ndims,T}(source_v),
        T(source_rho),
        Symbol[],
        state_updaters,
        _DEFAULT_PAIRS,
    )
end

function StressParticleSystem(
    path::AbstractString,
    name::AbstractString;
    source_v::Union{Nothing,AbstractVector{<:Real}} = nothing,
    source_rho::Real = 0.0,
    state_updater = (),
    dtype::Type{T} = Float64,
) where {T<:AbstractFloat}
    n, ndims, mass, c, ns = _read_h5_metadata(path, name, T)
    if ns === nothing
        throw(ArgumentError("StressParticleSystem requires 'stress' dataset in the HDF5 file to determine NS"))
    end

    sv = source_v === nothing ? zeros(T, ndims) : source_v

    ps = StressParticleSystem(name, n, ndims, ns, mass, c;
                              source_v=sv, source_rho=source_rho,
                              state_updater=state_updater)
    read_h5!(ps, path)
    return ps
end

# ---------------------------------------------------------------------------
# ElastoPlasticParticleSystem
# ---------------------------------------------------------------------------

"""
    ElastoPlasticParticleSystem{T, ND, NS, VT, PAIRS<:Tuple} <: AbstractParticleSystem{T, ND}

A particle system for elasto-plastic materials. Includes all StressParticleSystem
fields plus `vorticity`, `strain`, and `strain_p`.
"""
struct ElastoPlasticParticleSystem{T<:AbstractFloat, ND, NS, VT, PAIRS<:Tuple, UPD<:Tuple} <: AbstractParticleSystem{T, ND}
    name::String
    n::Int
    x::Vector{SVector{ND,T}}
    v::Vector{SVector{ND,T}}
    v_adjustment::Vector{SVector{ND,T}}
    rho::Vector{T}
    dvdt::Vector{SVector{ND,T}}
    drhodt::Vector{T}
    p::Vector{T}
    stress::Vector{SVector{NS,T}}
    strain_rate::Vector{SVector{NS,T}}
    vorticity::Vector{VT}
    strain::Vector{SVector{NS,T}}
    strain_p::Vector{SVector{NS,T}}
    mass::T
    c::T
    source_v::SVector{ND,T}
    source_rho::T
    _print_fields::Vector{Symbol}
    state_updater::UPD
    pairs::PAIRS
    function ElastoPlasticParticleSystem{T, ND, NS, VT, PAIRS, UPD}(args...) where {T, ND, NS, VT, PAIRS, UPD}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        NS isa Int || throw(ArgumentError("NS must be an Int, got $(typeof(NS))"))
        new{T, ND, NS, VT, PAIRS, UPD}(args...)
    end
end

function ElastoPlasticParticleSystem(
    name::AbstractString,
    n::Integer,
    ndims::Integer,
    ns::Integer,
    mass::Real,
    c::Real;
    dtype::Type{<:AbstractFloat} = Float64,
    source_v::AbstractVector{<:Real} = zeros(dtype, Int(ndims)),
    source_rho::Real = 0.0,
    state_updater = (),
)
    n = Int(n); ndims = Int(ndims); ns = Int(ns)
    n > 0 || throw(ArgumentError("n must be positive, got $n"))
    ndims > 0 || throw(ArgumentError("ndims must be positive, got $ndims"))
    ns ∈ (3, 4, 6) || throw(ArgumentError("ns must be 3, 4, or 6, got $ns"))
    length(source_v) == ndims || throw(ArgumentError(
        "length(source_v) = $(length(source_v)) must equal ndims = $ndims"))

    T = dtype
    VT = ndims == 2 ? T : SVector{3, T}

    x            = Vector{SVector{ndims,T}}(undef, n)
    v            = Vector{SVector{ndims,T}}(undef, n)
    v_adjustment = fill(zero(SVector{ndims,T}), n)
    rho          = Vector{T}(undef, n)
    dvdt         = fill(zero(SVector{ndims,T}), n)
    drhodt       = zeros(T, n)
    p            = zeros(T, n)
    stress       = fill(zero(SVector{ns,T}), n)
    strain_rate  = fill(zero(SVector{ns,T}), n)
    vorticity    = fill(zero(VT), n)
    strain       = fill(zero(SVector{ns,T}), n)
    strain_p     = fill(zero(SVector{ns,T}), n)

    state_updaters = state_updater isa Tuple ? state_updater : (state_updater,)
    _check_functors_eltype(state_updaters, T, "state updater")

    ElastoPlasticParticleSystem{T, ndims, ns, VT, typeof(_DEFAULT_PAIRS), typeof(state_updaters)}(
        String(name), n,
        x, v, v_adjustment, rho, dvdt, drhodt, p, stress, strain_rate, vorticity, strain, strain_p,
        T(mass), T(c),
        SVector{ndims,T}(source_v),
        T(source_rho),
        Symbol[],
        state_updaters,
        _DEFAULT_PAIRS,
    )
end

function ElastoPlasticParticleSystem(
    path::AbstractString,
    name::AbstractString;
    source_v::Union{Nothing,AbstractVector{<:Real}} = nothing,
    source_rho::Real = 0.0,
    state_updater = (),
    dtype::Type{T} = Float64,
) where {T<:AbstractFloat}
    n, ndims, mass, c, ns = _read_h5_metadata(path, name, T)
    if ns === nothing
        throw(ArgumentError("ElastoPlasticParticleSystem requires 'stress' dataset in the HDF5 file to determine NS"))
    end

    sv = source_v === nothing ? zeros(T, ndims) : source_v

    ps = ElastoPlasticParticleSystem(name, n, ndims, ns, mass, c;
                                     source_v=sv, source_rho=source_rho,
                                     state_updater=state_updater)
    read_h5!(ps, path)
    return ps
end

# ---------------------------------------------------------------------------
# VirtualParticleSystem
# ---------------------------------------------------------------------------

"""
    VirtualParticleSystem{T, ND, PS, UPD} <: AbstractParticleSystem{T, ND}

A lightweight wrapper around a source particle system that exposes all of its
fields and additionally owns `w_sum::Vector{T}` for kernel-weight accumulation
(SPH normalisation denominator).

All per-particle and scalar properties are forwarded to the wrapped `source`
system. `w_sum` and `state_updater` are owned by the virtual system.

State updaters are called by `update_state!` in the usual way — useful for
zeroing accumulated arrays before a sweep or normalising by `w_sum` after one.

    vps = VirtualParticleSystem(source_ps, name, n, ndims, mass, c;
                                state_updater=(ZeroFn(), NormaliseFn()))

`n`, `ndims`, `mass`, and `c` are validated against the source system.
"""
struct VirtualParticleSystem{T<:AbstractFloat, ND, PS<:AbstractParticleSystem{T,ND}, UPD<:Tuple, ZF} <: AbstractParticleSystem{T, ND}
    name::String
    source::PS
    w_sum::Vector{T}
    state_updater::UPD
    prescribed_v::SVector{ND,T}
    function VirtualParticleSystem{T,ND,PS,UPD,ZF}(args...) where {T,ND,PS,UPD,ZF}
        ND isa Int || throw(ArgumentError("ND must be an Int, got $(typeof(ND))"))
        new{T,ND,PS,UPD,ZF}(args...)
    end
end

function VirtualParticleSystem(
    ps::AbstractParticleSystem{T,ND},
    name::AbstractString,
    n::Integer,
    ndims::Integer,
    mass::Real,
    c::Real;
    dtype::Type{<:AbstractFloat} = T,
    state_updater = (),
    zero_fields::Tuple = (),
    prescribed_v = nothing,
) where {T,ND}
    n = Int(n); ndims = Int(ndims)
    ndims == ND  || throw(ArgumentError("ndims=$ndims does not match source ndims=$ND"))
    n    == ps.n || throw(ArgumentError("n=$n does not match source n=$(ps.n)"))
    state_updaters = state_updater isa Tuple ? state_updater : (state_updater,)
    _check_functors_eltype(state_updaters, T, "state updater")
    pv = prescribed_v === nothing ? zero(SVector{ND,T}) : SVector{ND,T}(prescribed_v)
    VirtualParticleSystem{T, ND, typeof(ps), typeof(state_updaters), zero_fields}(
        String(name), ps, zeros(T, n), state_updaters, pv,
    )
end

@inline function Base.getproperty(vps::VirtualParticleSystem{T,ND,PS,UPD,ZF}, s::Symbol) where {T,ND,PS,UPD,ZF}
    s === :ndims && return ND
    s === :n     && return length(getfield(getfield(vps, :source), :x))
    s in (:name, :source, :w_sum, :state_updater, :prescribed_v) && return getfield(vps, s)
    return getproperty(getfield(vps, :source), s)
end

# Auto-zero: clears w_sum and all ZF fields before each sweep loop.
_auto_zero_virtual!(::Tuple{}, vps) = nothing
@inline @Base.propagate_inbounds function _auto_zero_virtual!(zf::Tuple, vps)
    arr = getproperty(vps, first(zf))
    fill!(arr, zero(eltype(arr)))
    _auto_zero_virtual!(Base.tail(zf), vps)
end
@inline function auto_zero_virtual!(vps::VirtualParticleSystem{T,ND,PS,UPD,ZF}) where {T,ND,PS,UPD,ZF}
    fill!(getfield(vps, :w_sum), zero(T))
    _auto_zero_virtual!(ZF, vps)
end

# ---------------------------------------------------------------------------
# State update
# ---------------------------------------------------------------------------

"""
    update_state!(ps::AbstractParticleSystem, stage)
    update_state!(ps::AbstractParticleSystem)

Call each state updater function on all particle indices for the given stage.
"""
function update_state!(ps::AbstractParticleSystem, stage::Int)
    _update_state_pfns!(ps, ps.state_updater, stage)
    return nothing
end
update_state!(ps::AbstractParticleSystem) = update_state!(ps, 1)

# Type-stable tuple walk: peel off one updater at a time so `first(updaters)` is
# always a concrete type, avoiding the Union{A,B} that ps.state_updater[stage::Int]
# would produce for multi-element heterogeneous tuples.
_update_state_pfns!(ps, ::Tuple{}, stage) = nothing
@inline function _update_state_pfns!(ps, updaters::Tuple, stage)
    if stage == 1
        _update_state!(ps, first(updaters))
    else
        _update_state_pfns!(ps, Base.tail(updaters), stage - 1)
    end
end

_update_state!(ps, ::Nothing) = nothing
function _update_state!(ps, fn::SFN) where {SFN}
    @inbounds @batch for i in 1:ps.n
        fn(ps, i)
    end
end

# ---------------------------------------------------------------------------
# Print-field management
# ---------------------------------------------------------------------------

"""
    add_print_field!(ps, field::Symbol)

Mark `field` to be included in `print_summary` output.
"""
function add_print_field!(ps::AbstractParticleSystem, field::Symbol)
    hasfield(typeof(ps), field) || throw(ArgumentError(
        "unknown field :$field; valid fields are $(fieldnames(typeof(ps)))"))
    field ∉ ps._print_fields && push!(ps._print_fields, field)
    return ps
end

"""
    remove_print_field!(ps, field::Symbol)

Stop including `field` in `print_summary` output.
"""
function remove_print_field!(ps::AbstractParticleSystem, field::Symbol)
    idx = findfirst(==(field), ps._print_fields)
    idx === nothing && throw(ArgumentError("':$field' is not in the print list"))
    deleteat!(ps._print_fields, idx)
    return ps
end

# ---------------------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------------------

function _scalar_stats(a::AbstractVector)
    n    = length(a)
    imin = imax = 1
    vmin = vmax = vsum = a[1]
    @inbounds for i in 2:n
        v = a[i]
        if v < vmin; vmin = v; imin = i; end
        if v > vmax; vmax = v; imax = i; end
        vsum += v
    end
    return imin, vmin, vsum / n, imax, vmax
end

function _field_rows(name::AbstractString, arr::AbstractVector)
    imin, vmin, vmean, imax, vmax = _scalar_stats(arr)
    return [(name, imin, vmin, vmean, imax, vmax)]
end

function _field_rows(name::AbstractString, arr::AbstractVector{SVector{ND,T}}) where {ND,T}
    n = length(arr)

    mag0     = norm(arr[1])
    mag_min  = mag_max  = mag_sum = mag0
    mag_imin = mag_imax = 1

    comp_min  = comp_max  = arr[1]
    comp_sum  = arr[1]
    comp_imin = comp_imax = SVector{ND,Int}(ntuple(_ -> 1, Val(ND)))

    @inbounds for i in 2:n
        vi  = arr[i]
        mag = norm(vi)

        if mag < mag_min; mag_min = mag; mag_imin = i; end
        if mag > mag_max; mag_max = mag; mag_imax = i; end
        mag_sum += mag

        comp_sum += vi
        for d in 1:ND
            if vi[d] < comp_min[d]
                comp_min  = Base.setindex(comp_min,  vi[d], d)
                comp_imin = Base.setindex(comp_imin, i, d)
            end
            if vi[d] > comp_max[d]
                comp_max  = Base.setindex(comp_max,  vi[d], d)
                comp_imax = Base.setindex(comp_imax, i, d)
            end
        end
    end

    rows = Vector{Tuple{String,Int,T,T,Int,T}}(undef, ND + 1)
    rows[1] = ("$name |mag|", mag_imin, T(mag_min), T(mag_sum / n), mag_imax, T(mag_max))
    for d in 1:ND
        rows[d+1] = ("$name [$d]", comp_imin[d], comp_min[d], comp_sum[d] / n, comp_imax[d], comp_max[d])
    end
    return rows
end

"""
    print_summary(ps::AbstractParticleSystem, io::IO=stdout)

Print per-field statistics for all fields registered via `add_print_field!`.
"""
function print_summary(ps::AbstractParticleSystem, io::IO=stdout)
    isempty(ps._print_fields) && return

    all_rows = Tuple[]
    for field in ps._print_fields
        append!(all_rows, _field_rows(string(field), getproperty(ps, field)))
    end
    isempty(all_rows) && return

    nrows = length(all_rows)
    data = Matrix{Any}(undef, nrows, 6)
    for (i, (name, imin, vmin, vmean, imax, vmax)) in enumerate(all_rows)
        data[i, 1] = name
        data[i, 2] = imin
        data[i, 3] = @sprintf("%.6g", vmin)
        data[i, 4] = @sprintf("%.6g", vmean)
        data[i, 5] = imax
        data[i, 6] = @sprintf("%.6g", vmax)
    end

    pretty_table(io, data;
        header = ["field", "i_min", "min", "mean", "i_max", "max"],
        tf = tf_markdown,
        alignment = [:l, :r, :r, :r, :r, :r],
    )
    flush(io)
end

# ---------------------------------------------------------------------------
# HDF5 I/O
# ---------------------------------------------------------------------------

_sim_field_names(::BasicParticleSystem)  = (:v, :rho, :dvdt, :drhodt)
_sim_field_names(::FluidParticleSystem)  = (:v, :rho, :dvdt, :drhodt, :p)
_sim_field_names(::StressParticleSystem) = (:v, :rho, :dvdt, :drhodt, :p, :stress, :strain_rate)
_sim_field_names(::ElastoPlasticParticleSystem) = (:v, :rho, :dvdt, :drhodt, :p, :stress, :strain_rate, :vorticity, :strain, :strain_p)
_sim_field_names(vps::VirtualParticleSystem) = _sim_field_names(getfield(vps, :source))

"""
    write_h5(ps::AbstractParticleSystem, target)

Write particle data and metadata to HDF5. `target` is either a file path
(written with mode `"w"`) or an already-open `HDF5.File`/`HDF5.Group`.
"""
function write_h5(ps::AbstractParticleSystem, target::AbstractString)
    h5open(target, "w") do f
        _write_to_group!(ps, f)
    end
end

function write_h5(ps::AbstractParticleSystem, group::Union{HDF5.File,HDF5.Group})
    _write_to_group!(ps, group)
end

function _write_to_group!(ps::AbstractParticleSystem, group)
    HDF5.attrs(group)["n"]     = ps.n
    HDF5.attrs(group)["ndims"] = ps.ndims
    HDF5.attrs(group)["mass"]  = ps.mass
    HDF5.attrs(group)["c"]     = ps.c
    group["x"] = reinterpret(reshape, eltype(eltype(ps.x)), ps.x)
    for k in _sim_field_names(ps)
        arr = getproperty(ps, k)
        if arr isa AbstractVector{<:SVector}
            group[String(k)] = reinterpret(reshape, eltype(eltype(arr)), arr)
        else
            group[String(k)] = arr
        end
    end
end

"""
    read_h5!(ps::AbstractParticleSystem, source)

Read particle data from an HDF5 file or group into an existing `AbstractParticleSystem`.
The target system `ps` must have the same number of particles (`n`) and dimensions
(`ndims`) as the saved data. Only arrays present in both the file and the system
are updated.

Returns the modified `ps`.
"""
function read_h5!(ps::AbstractParticleSystem, source::AbstractString)
    h5open(source, "r") do f
        _read_from_group!(ps, f)
    end
    return ps
end

function read_h5!(ps::AbstractParticleSystem, group::Union{HDF5.File,HDF5.Group})
    _read_from_group!(ps, group)
    return ps
end

function _read_from_group!(ps::AbstractParticleSystem, group)
    n = Int(HDF5.attrs(group)["n"][])
    ndims = Int(HDF5.attrs(group)["ndims"][])
    
    if n != ps.n
        throw(ArgumentError("Saved particle count ($n) does not match system count ($(ps.n))"))
    end
    if ndims != ps.ndims
        throw(ArgumentError("Saved ndims ($ndims) does not match system ndims ($(ps.ndims))"))
    end
    
    if haskey(group, "x")
        x_data = group["x"][]
        ps.x .= reinterpret(reshape, eltype(ps.x), x_data)
    end
    
    for k in _sim_field_names(ps)
        s_k = String(k)
        if haskey(group, s_k)
            arr = getfield(ps, k)
            data = group[s_k][]
            if arr isa AbstractVector{<:SVector}
                arr .= reinterpret(reshape, eltype(arr), data)
            else
                arr .= data
            end
        end
    end
end