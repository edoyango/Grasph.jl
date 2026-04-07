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
