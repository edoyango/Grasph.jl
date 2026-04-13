# ---------------------------------------------------------------------------
# Type-stable particle field accessors
# ---------------------------------------------------------------------------

# Type-stable field accessor: S is a compile-time constant, so getfield is fully inferred.
@inline _getf(ps, ::Val{S}) where {S} = getfield(ps, S)

# Source (reset) value for each known derivative field; zero for user-added pairs.
# Called with a Val{S} dqdt key, so dispatch is always type-stable.
@inline _source_for(ps::AbstractParticleSystem, ::Val{:dvdt})          = ps.source_v
@inline _source_for(ps::AbstractParticleSystem, ::Val{:drhodt})        = ps.source_rho
@inline _source_for(ps::AbstractParticleSystem, ::Val{name}) where {name} =
    zero(eltype(getfield(ps, name)))

# ---------------------------------------------------------------------------
# Primitive axpy operations
# ---------------------------------------------------------------------------

@inline function _axpy_ip!(q, dqdt, a)
    @inbounds @fastmath @batch for i in eachindex(q)
        q[i] += a * dqdt[i]
    end
end

@inline function _axpy_oop!(q, q0, dqdt, a)
    @inbounds @fastmath @batch for i in eachindex(q)
        q[i] = q0[i] + a * dqdt[i]
    end
end

@inline function _zero_field(ps::AbstractParticleSystem, field::Symbol)
    f = _getf(ps, Val(field))
    fill!(f, zero(eltype(f)))
end

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

"""
    _check_fields(ps::AbstractParticleSystem, required::NTuple{N,Symbol}, context::String)

Verify that particle system `ps` has all fields listed in `required`.
Throws `ArgumentError` if any are missing.
"""
function _check_fields(ps::AbstractParticleSystem, required::NTuple{N,Symbol}, context::String) where {N}
    for f in required
        hasfield(typeof(ps), f) || throw(ArgumentError(
            "$context: $(nameof(typeof(ps))) \"$(ps.name)\" is missing required field :$f"))
    end
end

"""
    _check_functor_eltype(functor, ::Type{T}, label::String)

Verify that if `functor` has type parameters that are `AbstractFloat`,
they match the expected type `T`.
"""
function _check_functor_eltype(functor, ::Type{T}, label::String) where {T}
    for P in typeof(functor).parameters
        if P isa Type && P <: AbstractFloat && P !== T
            throw(ArgumentError(
                "$label $(nameof(typeof(functor))) is specialised on $P " *
                "but target expects $T; construct it with $T values"))
        end
    end
end

"""
    _check_functors_eltype(functors::Tuple, ::Type{T}, label::String)

Batch version of `_check_functor_eltype`.
"""
function _check_functors_eltype(functors::Tuple, ::Type{T}, label::String) where {T}
    for f in functors
        _check_functor_eltype(f, T, label)
    end
end
