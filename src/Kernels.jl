export CubicSplineKernel, kernel_w, kernel_dw_dq

# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------

abstract type AbstractKernel{T<:AbstractFloat, ND, H} end

"""
    CubicSplineKernel{T<:AbstractFloat, ND, H}

SPH cubic spline kernel with smoothing length `H` (encoded in the type).
Support radius `interaction_length = 2H`.

All properties (`h`, `norm_coeff`, `interaction_length`) are derived from the
type parameters and constant-folded by the compiler — the struct has no fields.
"""
struct CubicSplineKernel{T<:AbstractFloat, ND, H} <: AbstractKernel{T,ND,H}
end

# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

"""
    CubicSplineKernel(h; ndims=3, dtype=Float64) -> CubicSplineKernel

Construct a cubic spline kernel with smoothing length `h` in `ndims` dimensions.

# Keyword arguments
- `ndims`: spatial dimension (1, 2, or 3; default 3)
- `dtype`: floating-point element type (default `Float64`)
"""
function CubicSplineKernel(h::T; ndims::Int=3) where {T<:AbstractFloat}
    CubicSplineKernel{T, ndims, h}()
end

# ---------------------------------------------------------------------------
# Property access — all values derived from type parameters, constant-folded
# ---------------------------------------------------------------------------

@inline function Base.getproperty(::AbstractKernel{T, ND, H}, s::Symbol) where {T, ND, H}
    if s === :ndims
        return ND
    elseif s === :h
        return H
    elseif s === :interaction_length
        return T(2) * H
    elseif s === :norm_coeff
        if ND == 1
            return T(2 / (3*H))
        elseif ND == 2
            return T(10 / (7*π*H^2))
        else
            return T(1 / (π*H^3))
        end
    end
    error("CubicSplineKernel has no field: $s")
end

# ---------------------------------------------------------------------------
# Kernel functions
# ---------------------------------------------------------------------------

"""
    kernel_w(k::CubicSplineKernel, q) -> W(q)

Returns the normalised kernel value at `q = r/h`.
"""
@inline function kernel_w(k::CubicSplineKernel{T}, q::T) where {T<:AbstractFloat}
    k.norm_coeff * (T(0.25)*max(zero(T), T(2)-q)^3 - max(zero(T), T(1)-q)^3)
end

"""
    kernel_dw_dq(k::CubicSplineKernel, q) -> dW/dq

Returns the normalised kernel derivative at `q = r/h`.
"""
@inline function kernel_dw_dq(k::CubicSplineKernel{T}, q::T) where {T<:AbstractFloat}
    k.norm_coeff * (-T(3)*(T(0.25)*max(zero(T), T(2)-q)^2 - max(zero(T), T(1)-q)^2))
end
