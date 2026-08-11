export AbstractTimeIntegrator, LeapFrogTimeIntegrator, RK4TimeIntegrator, time_integrate!

# ---------------------------------------------------------------------------
# Integrator types
# ---------------------------------------------------------------------------

"""
    AbstractTimeIntegrator

Abstract supertype for all time integrators.  Concrete subtypes must implement
`time_integrate!(integrator::ConcreteType, ...)`.
"""
abstract type AbstractTimeIntegrator end

# ---------------------------------------------------------------------------
# verlet_skin validation — shared by LeapFrogTimeIntegrator/RK4TimeIntegrator.
#
# verlet_skin > 0 opts into time_integrate!'s rebuild-cadence gate (skip
# sort+grid rebuild on steps where no tracked particle has moved far enough to
# invalidate the current cell list — see docs/gpu-migration-plan.md, deferred
# item 1). Scope is deliberately narrow: only the onesided sweep modes
# (OnesidedCPU/OnesidedKA), plus the NeighbourListKA benchmarking spike
# (which reuses the same grid-pitch/physical-cutoff split and additionally
# needs verlet_skin>0 for its cached candidate list to ever be reused across
# more than one step), have the grid-pitch/physical-cutoff split needed
# for a widened, staleness-tolerant cell list (Interaction.jl, KAKernels.jl);
# ghosts are regenerated from live boundary positions every step regardless of
# skin, and virtual systems aren't tracked either, so both are rejected
# outright rather than silently going stale.
# ---------------------------------------------------------------------------
function _validate_verlet_skin(verlet_skin, ints, gsts, vsys, T)
    iszero(verlet_skin) && return nothing
    verlet_skin > 0 || throw(ArgumentError("verlet_skin must be >= 0, got $verlet_skin"))
    isempty(gsts) || throw(ArgumentError(
        "verlet_skin > 0 is not supported together with ghosts (generate_ghosts! regenerates them from live boundary positions every step, regardless of skin)"))
    isempty(vsys) || throw(ArgumentError(
        "verlet_skin > 0 is not supported together with virtual_systems (not tracked by the rebuild-cadence displacement check)"))
    all(inter -> _exec_mode(inter) isa Union{OnesidedCPU,OnesidedKA,NeighbourListKA}, ints) || throw(ArgumentError(
        "verlet_skin > 0 requires every interaction to use onesided=true (OnesidedCPU/OnesidedKA) — the coloured sweep's grid-pitch/cutoff split is not implemented"))
    min_cutoff = T(minimum(inter._cell_size for inter in ints))
    T(verlet_skin) < 2*min_cutoff || throw(ArgumentError(
        "verlet_skin ($verlet_skin) must be < 2 * the smallest interaction cutoff ($(2*min_cutoff)) — the grid's existing bounding-box padding can't absorb more drift than that without risking an out-of-bounds cell index"))
    return nothing
end

"""
    LeapFrogTimeIntegrator

Leapfrog time integrator for one or more `AbstractParticleSystem`s.

Conjugate pairs `(q, dqdt)` are driven by `ps.pairs`. At each step:

1. Save each `q`.
2. **Half-step**: `q += 0.5 dt * dqdt`, reset `dqdt ← source`, call
   `update_state!`.
3. Rebuild the cell list and sweep all interactions.
4. **Full-step**: `q = q₀ + dt * dqdt`.
5. **Update positions**: `x += dt * v`.
"""
struct LeapFrogTimeIntegrator{SYS<:Tuple, INTS<:Tuple, GHOSTS<:Tuple, VSYS<:Tuple, PRBS<:Tuple, PINTS<:Tuple, T<:AbstractFloat} <: AbstractTimeIntegrator
    systems::SYS
    interactions::INTS
    ghosts::GHOSTS
    virtual_systems::VSYS
    probes::PRBS
    probe_interactions::PINTS
    c::T
    h::T
    Γ::T
    verlet_skin::T
end

"""
    LeapFrogTimeIntegrator(systems, interactions; ghosts=(), virtual_systems=()) -> LeapFrogTimeIntegrator

Construct a `LeapFrogTimeIntegrator`.

- `systems`: an `AbstractParticleSystem` or iterable of `AbstractParticleSystem`s.
- `interactions`: a `SystemInteraction` or iterable of `SystemInteraction`s.
- `virtual_systems`: `VirtualParticleSystem`s that are sorted and state-updated each step
  but whose velocity is zeroed before position integration (fixed boundaries).

Raises `ArgumentError` if either collection is empty.

Pass `verlet_skin` (default `0`) to opt into a rebuild-cadence gate that skips
the per-step sort + grid rebuild while no tracked particle has moved far
enough to invalidate the current cell list (see
`docs/gpu-migration-plan.md`, deferred item 1). Requires every interaction to
use `onesided=true`, and no `ghosts`/`virtual_systems`; raises `ArgumentError`
otherwise. Default `0` reproduces today's rebuild-every-step behaviour
exactly.
"""
function LeapFrogTimeIntegrator(systems, interactions; ghosts=(), virtual_systems=(), probes=(), probe_interactions=(), Γ=0, verlet_skin=0)
    sys   = systems            isa AbstractParticleSystem  ? (systems,)            : Tuple(systems)
    ints  = interactions       isa SystemInteraction       ? (interactions,)       : Tuple(interactions)
    gsts  = ghosts             isa GhostEntry              ? (ghosts,)             : Tuple(ghosts)
    vsys  = virtual_systems    isa VirtualParticleSystem   ? (virtual_systems,)    : Tuple(virtual_systems)
    prbs  = probes             isa ProbeParticleSystem     ? (probes,)             : Tuple(probes)
    pints = probe_interactions isa SystemInteraction       ? (probe_interactions,) : Tuple(probe_interactions)
    isempty(sys)  && throw(ArgumentError("systems must not be empty"))
    isempty(ints) && throw(ArgumentError("interactions must not be empty"))
    T = eltype(eltype(first(sys).x))
    c = T(maximum(ps.c           for ps   in sys))
    h = T(minimum(inter.kernel.h for inter in ints))
    _validate_verlet_skin(verlet_skin, ints, gsts, vsys, T)
    LeapFrogTimeIntegrator{typeof(sys), typeof(ints), typeof(gsts), typeof(vsys), typeof(prbs), typeof(pints), T}(
        sys, ints, gsts, vsys, prbs, pints, c, h, T(Γ), T(verlet_skin))
end

"""
    RK4TimeIntegrator

Classical 4th-order Runge-Kutta time integrator for one or more
`AbstractParticleSystem`s.

The neighbour grid is built once per timestep (frozen Lagrangian approximation).
Each timestep evaluates the RHS four times with intermediate states:

    k1 = f(q0)
    k2 = f(q0 + dt/2 · k1)
    k3 = f(q0 + dt/2 · k2)
    k4 = f(q0 + dt   · k3)
    q  = q0 + dt · (k1/6 + k2/3 + k3/3 + k4/6)

Positions are updated once at the end: `x += dt · v`.
"""
struct RK4TimeIntegrator{SYS<:Tuple, INTS<:Tuple, GHOSTS<:Tuple, VSYS<:Tuple, PRBS<:Tuple, PINTS<:Tuple, T<:AbstractFloat} <: AbstractTimeIntegrator
    systems::SYS
    interactions::INTS
    ghosts::GHOSTS
    virtual_systems::VSYS
    probes::PRBS
    probe_interactions::PINTS
    c::T
    h::T
    Γ::T
    verlet_skin::T
end

# See LeapFrogTimeIntegrator's `verlet_skin` docstring — identical contract
# here; the grid is "frozen" across a timestep's 4 RK stages regardless, and
# `verlet_skin > 0` extends that freezing across multiple timesteps too.
function RK4TimeIntegrator(systems, interactions; ghosts=(), virtual_systems=(), probes=(), probe_interactions=(), Γ=0, verlet_skin=0)
    sys   = systems            isa AbstractParticleSystem  ? (systems,)            : Tuple(systems)
    ints  = interactions       isa SystemInteraction       ? (interactions,)       : Tuple(interactions)
    gsts  = ghosts             isa GhostEntry              ? (ghosts,)             : Tuple(ghosts)
    vsys  = virtual_systems    isa VirtualParticleSystem   ? (virtual_systems,)    : Tuple(virtual_systems)
    prbs  = probes             isa ProbeParticleSystem     ? (probes,)             : Tuple(probes)
    pints = probe_interactions isa SystemInteraction       ? (probe_interactions,) : Tuple(probe_interactions)
    isempty(sys)  && throw(ArgumentError("systems must not be empty"))
    isempty(ints) && throw(ArgumentError("interactions must not be empty"))
    T = eltype(eltype(first(sys).x))
    c = T(maximum(ps.c           for ps   in sys))
    h = T(minimum(inter.kernel.h for inter in ints))
    _validate_verlet_skin(verlet_skin, ints, gsts, vsys, T)
    RK4TimeIntegrator{typeof(sys), typeof(ints), typeof(gsts), typeof(vsys), typeof(prbs), typeof(pints), T}(
        sys, ints, gsts, vsys, prbs, pints, c, h, T(Γ), T(verlet_skin))
end

# ---------------------------------------------------------------------------
# Per-system step helpers — generic over ps.pairs
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Pair-level tuple walkers
#
# ps.pairs is a heterogeneous Tuple of (Val{:q}, Val{:dqdt}) pairs — one entry
# per conjugate (q, dqdt) field that the integrator should step.  Because the
# tuple is heterogeneous (each element has a distinct concrete type carrying the
# field name as a type parameter), we cannot iterate it with a plain `for` loop:
# Julia would infer the element type as a Union and box intermediate values on
# the heap, preventing constant-folding of the Val field names.
#
# The solution is recursive dispatch using `Base.tail`:
#   - The base case matches the empty tuple `::Tuple{}` and stops.
#   - The recursive case matches any non-empty Tuple, so `first(pairs)` is
#     always inferred as the *concrete* type of the first element (not a Union).
#     After processing it we call ourselves with `Base.tail(pairs)`, which
#     sheds the first element and gives a new concrete tuple type.
# Each call is a distinct specialisation, so the whole chain is inlined by the
# compiler into a flat sequence of array operations — zero runtime overhead.
# ---------------------------------------------------------------------------

# Advance each q field by a * dqdt (used for the leapfrog half-step).
_halfstep_pairs!(ps, ::Tuple{}, a) = nothing
@inline function _halfstep_pairs!(ps, pairs::Tuple, a)
    q_val, dqdt_val = first(pairs)           # compile-time field-name constants
    _axpy_ip!(_getf(ps, q_val), _getf(ps, dqdt_val), a)
    _halfstep_pairs!(ps, Base.tail(pairs), a) # recurse on the remaining pairs
end

# Reset each dqdt field back to its source value (gravity for dvdt, 0 for the
# rest) so the next sweep accumulates onto a clean slate.
_reset_dqdt_pairs!(ps, ::Tuple{}) = nothing
@inline function _reset_dqdt_pairs!(ps, pairs::Tuple)
    _, dqdt_val = first(pairs)
    fill!(_getf(ps, dqdt_val), _source_for(ps, dqdt_val))
    _reset_dqdt_pairs!(ps, Base.tail(pairs))
end

# Half-step a single particle system: advance q, then reset dqdt.
# Separated into two passes so the reset is a distinct, clearly-named step.
@inline function _halfstep_ps!(ps::AbstractParticleSystem, half_dt)
    pairs = getfield(ps, :pairs)
    _halfstep_pairs!(ps, pairs, half_dt)
    _reset_dqdt_pairs!(ps, pairs)
end

# ---------------------------------------------------------------------------
# q0 buffer helpers
#
# The leapfrog full-step computes q = q0 + dt * dqdt, where q0 is the value
# of q at the *start* of the timestep (before the half-step advance).  We
# pre-allocate a typed tuple of arrays — one per conjugate pair — and copy q
# into them before the half-step so the original values are preserved.
#
# Using a Tuple (rather than a Vector) to hold the buffers is important: a
# Vector would erase the element types of each array (e.g. Vector{SVector{2,F}}
# vs Vector{SVector{3,F}}), forcing _advance_q_pairs! to box values or fall back
# to dynamic dispatch.  A Tuple keeps each buffer's concrete type visible to
# the compiler throughout the full-step walk.
# ---------------------------------------------------------------------------

# Allocate one copy-buffer per pair by recursing through ps.pairs at startup.
_make_q0_bufs(ps, ::Tuple{}) = ()
@inline function _make_q0_bufs(ps, pairs::Tuple)
    q_val = first(first(pairs))
    (copy(_getf(ps, q_val)), _make_q0_bufs(ps, Base.tail(pairs))...)
end
@inline _make_q0_bufs(ps::AbstractParticleSystem) = _make_q0_bufs(ps, getfield(ps, :pairs))

# Snapshot current q values into the pre-allocated buffers.
_save_q0_pairs!(ps, ::Tuple{}, ::Tuple{}) = nothing
@inline function _save_q0_pairs!(ps, pairs::Tuple, bufs::Tuple)
    q_val = first(first(pairs))
    copyto!(first(bufs), _getf(ps, q_val))
    _save_q0_pairs!(ps, Base.tail(pairs), Base.tail(bufs))
end


# ---------------------------------------------------------------------------
# All-systems tuple walkers
#
# `sys` is a heterogeneous Tuple of particle systems (potentially different
# concrete types).  A plain `for (i, ps) in enumerate(sys)` loop infers `ps`
# as a Union of all system types.  Any function call inside the loop that
# dispatches on `ps`'s concrete type then returns a Union result, which Julia
# must heap-box before passing onward — preventing type inference from
# propagating through the call chain.
#
# The same Base.tail recursion used for pair walkers above resolves this:
# `first(sys)` is always a single concrete type, so every downstream call is
# fully specialised with no Union boxing.
# ---------------------------------------------------------------------------

# Sort every real particle system by cell index (required before grid build).
_sort_all_systems!(::Tuple{}, ::Tuple{}, cutoff, perm_buf, key_buf, to, labels, idx) = nothing
@inline function _sort_all_systems!(sys::Tuple, scratches::Tuple, cutoff, perm_buf, key_buf, to, labels, idx)
    @timeit to labels[idx].sort @timeit to labels[idx].name sort_particles!(first(sys), cutoff, perm_buf, key_buf, first(scratches))
    _sort_all_systems!(Base.tail(sys), Base.tail(scratches), cutoff, perm_buf, key_buf, to, labels, idx + 1)
end

# Snapshot q0 for every system before the half-step.
_save_q0_all!(::Tuple{}, ::Tuple{}) = nothing
@inline function _save_q0_all!(sys::Tuple, q0s::Tuple)
    ps = first(sys)
    _save_q0_pairs!(ps, getfield(ps, :pairs), first(q0s))
    _save_q0_all!(Base.tail(sys), Base.tail(q0s))
end

# Apply the full-step update (q = q0 + dt * dqdt) to every system.
_fullstep_q_all!(::Tuple{}, ::Tuple{}, dt, to, labels, idx) = nothing
@inline function _fullstep_q_all!(sys::Tuple, q0s::Tuple, dt, to, labels, idx)
    ps = first(sys)
    @timeit to labels[idx].full @timeit to labels[idx].name _advance_q_pairs!(ps, getfield(ps, :pairs), first(q0s), dt)
    _fullstep_q_all!(Base.tail(sys), Base.tail(q0s), dt, to, labels, idx + 1)
end

# Integrate positions forward: x += dt * v.
_update_positions_all!(::Tuple{}, dt, to, labels, idx) = nothing
@inline function _update_positions_all!(sys::Tuple, dt, to, labels, idx)
    ps = first(sys)
    @timeit to labels[idx].pos @timeit to labels[idx].name _axpy_ip!(ps.x, ps.v, dt)
    _update_positions_all!(Base.tail(sys), dt, to, labels, idx + 1)
end

# Reset every dqdt field to its source value across all systems.
_reset_dqdt_all!(::Tuple{}) = nothing
@inline function _reset_dqdt_all!(sys::Tuple)
    ps = first(sys)
    _reset_dqdt_pairs!(ps, getfield(ps, :pairs))
    _reset_dqdt_all!(Base.tail(sys))
end

# Velocity damping: dvdt[i] -= (Γ/dt) * v[i] for the (:v, :dvdt) conjugate pair.
@inline _apply_damping_pair!(ps, ::Val{:v}, dqdt_val::Val, Γ_dt) = begin
    v    = _getf(ps, Val{:v}())
    dvdt = _getf(ps, dqdt_val)
    @. dvdt -= Γ_dt * v
end
@inline _apply_damping_pair!(ps, ::Val, ::Val, ::Any) = nothing

_apply_damping_pairs!(ps, ::Tuple{}, ::Any) = nothing
@inline function _apply_damping_pairs!(ps, pairs::Tuple, Γ_dt)
    q_val, dqdt_val = first(pairs)
    _apply_damping_pair!(ps, q_val, dqdt_val, Γ_dt)
    _apply_damping_pairs!(ps, Base.tail(pairs), Γ_dt)
end

_apply_damping_all!(::Tuple{}, ::Any) = nothing
@inline function _apply_damping_all!(sys::Tuple, Γ_dt)
    _apply_damping_pairs!(first(sys), getfield(first(sys), :pairs), Γ_dt)
    _apply_damping_all!(Base.tail(sys), Γ_dt)
end

# ---------------------------------------------------------------------------
# RK4 helpers
#
# Three new pair-level walkers, each with an _all! wrapper:
#
#   _advance_q_pairs!    q = q0 + coeff*dqdt   (stage setup for stages 2-4)
#   _acc_dqdt_pairs!     acc += weight*dqdt     (accumulate ki after each sweep)
#   _apply_acc_pairs!    q = q0 + dt*acc        (final RK4 update)
#
# acc buffers are typed tuples parallel to q0 buffers, allocated via
# _make_acc_bufs and zeroed at the start of each timestep via _zero_acc_all!.
# ---------------------------------------------------------------------------

# Allocate zeroed accumulator buffers (one per dqdt field, same shape as dqdt).
_make_acc_bufs(ps, ::Tuple{}) = ()
@inline function _make_acc_bufs(ps, pairs::Tuple)
    _, dqdt_val = first(pairs)
    dqdt_arr = _getf(ps, dqdt_val)
    buf = fill!(similar(dqdt_arr), zero(eltype(dqdt_arr)))
    (buf, _make_acc_bufs(ps, Base.tail(pairs))...)
end
@inline _make_acc_bufs(ps::AbstractParticleSystem) = _make_acc_bufs(ps, getfield(ps, :pairs))

# Zero all accumulator buffers for one system.
_zero_acc_pairs!(::Tuple{}) = nothing
@inline function _zero_acc_pairs!(accs::Tuple)
    acc = first(accs)
    fill!(acc, zero(eltype(acc)))
    _zero_acc_pairs!(Base.tail(accs))
end

# Zero accumulators for every system.
_zero_acc_all!(::Tuple{}, ::Tuple{}) = nothing
@inline function _zero_acc_all!(sys::Tuple, accs::Tuple)
    _zero_acc_pairs!(first(accs))
    _zero_acc_all!(Base.tail(sys), Base.tail(accs))
end

# q = q0 + coeff * dqdt  (used for both the LeapFrog full-step and RK4 stage advances).
_advance_q_pairs!(ps, ::Tuple{}, ::Tuple{}, coeff) = nothing
@inline function _advance_q_pairs!(ps, pairs::Tuple, q0s::Tuple, coeff)
    q_val, dqdt_val = first(pairs)
    _axpy_oop!(_getf(ps, q_val), first(q0s), _getf(ps, dqdt_val), coeff)
    _advance_q_pairs!(ps, Base.tail(pairs), Base.tail(q0s), coeff)
end

_rk4_advance_all!(::Tuple{}, ::Tuple{}, coeff, to, labels, idx) = nothing
@inline function _rk4_advance_all!(sys::Tuple, q0s::Tuple, coeff, to, labels, idx)
    ps = first(sys)
    @timeit to labels[idx].mid @timeit to labels[idx].name begin
        _advance_q_pairs!(ps, getfield(ps, :pairs), first(q0s), coeff)
        _reset_dqdt_pairs!(ps, getfield(ps, :pairs))
    end
    _rk4_advance_all!(Base.tail(sys), Base.tail(q0s), coeff, to, labels, idx + 1)
end

# acc += weight * dqdt
_acc_dqdt_pairs!(ps, ::Tuple{}, ::Tuple{}, weight) = nothing
@inline function _acc_dqdt_pairs!(ps, pairs::Tuple, accs::Tuple, weight)
    _, dqdt_val = first(pairs)
    _axpy_ip!(first(accs), _getf(ps, dqdt_val), weight)
    _acc_dqdt_pairs!(ps, Base.tail(pairs), Base.tail(accs), weight)
end

_acc_dqdt_all!(::Tuple{}, ::Tuple{}, weight) = nothing
@inline function _acc_dqdt_all!(sys::Tuple, accs::Tuple, weight)
    ps = first(sys)
    _acc_dqdt_pairs!(ps, getfield(ps, :pairs), first(accs), weight)
    _acc_dqdt_all!(Base.tail(sys), Base.tail(accs), weight)
end

# q = q0 + dt * acc  (final RK4 update; structurally identical to _advance_q_pairs!)
_apply_acc_pairs!(ps, ::Tuple{}, ::Tuple{}, ::Tuple{}, dt) = nothing
@inline function _apply_acc_pairs!(ps, pairs::Tuple, q0s::Tuple, accs::Tuple, dt)
    q_val, _ = first(pairs)
    _axpy_oop!(_getf(ps, q_val), first(q0s), first(accs), dt)
    _apply_acc_pairs!(ps, Base.tail(pairs), Base.tail(q0s), Base.tail(accs), dt)
end

_apply_acc_all!(::Tuple{}, ::Tuple{}, ::Tuple{}, dt, to, labels, idx) = nothing
@inline function _apply_acc_all!(sys::Tuple, q0s::Tuple, accs::Tuple, dt, to, labels, idx)
    ps = first(sys)
    @timeit to labels[idx].full @timeit to labels[idx].name _apply_acc_pairs!(ps, getfield(ps, :pairs), first(q0s), first(accs), dt)
    _apply_acc_all!(Base.tail(sys), Base.tail(q0s), Base.tail(accs), dt, to, labels, idx + 1)
end

# ---------------------------------------------------------------------------
# Shared per-step helpers (used by both LeapFrog and RK4)
# ---------------------------------------------------------------------------

# Generate ghosts, sort them, sort virtual systems, then build all interaction grids.
#
# `skin` (default 0) is `integrator.verlet_skin`, forwarded to every
# `create_grid!` call so each interaction's cell list is padded consistently
# with the shared `sort_cutoff` used by `_sort_all_systems!` — see
# `time_integrate!`'s rebuild-cadence gate. Ghosts/virtual systems are always
# empty here when `skin > 0` (enforced at integrator construction), so the
# loops below are unaffected either way.
function _prepare_grids!(ghosts, virtual_sys, ints, sort_cutoff, sort_perm_buf, sort_key_buf,
                          ghost_scratches, virtual_scratches, to, ghost_labels, inter_labels, skin)
    for (i, ge) in enumerate(ghosts)
        @timeit to ghost_labels[i].gen @timeit to ghost_labels[i].name generate_ghosts!(ge)
    end
    for (i, ge) in enumerate(ghosts)
        @timeit to ghost_labels[i].sort @timeit to ghost_labels[i].name sort_particles!(
            ge.ghost, sort_cutoff, sort_perm_buf, sort_key_buf, ghost_scratches[i])
    end
    for (i, vps) in enumerate(virtual_sys)
        sort_particles!(vps, sort_cutoff, sort_perm_buf, sort_key_buf, virtual_scratches[i])
    end
    for (i, inter) in enumerate(ints)
        @timeit to inter_labels[i].grid @timeit to inter_labels[i].name begin
            create_grid!(inter, skin)
            _maybe_build_neighbour_list!(inter)
        end
    end
end

# ---------------------------------------------------------------------------
# Verlet-skin rebuild-cadence gate (verlet_skin > 0 only — see
# _validate_verlet_skin above and docs/gpu-migration-plan.md deferred item 1).
#
# Tracks every system in `sys` plus each interaction's `system_b` (e.g. a
# coupled boundary), snapshotting positions at the last rebuild and comparing
# against current positions every step. `2 * max_displacement <= verlet_skin`
# is the standard Verlet-list bound: two particles farther apart than the
# padded grid's cell pitch when it was built cannot have moved within the
# true cutoff of each other while every tracked particle's own displacement
# since that build stays under `skin / 2`.
#
# No attempt is made to deduplicate `system_b`s already present in `sys` by
# object identity (e.g. dambreak's boundary is both a `sys` member and, via a
# `StaticBoundarySystem` wrapper, an interaction's `system_b`) — tracking the
# same underlying positions twice is wasted work, not a correctness issue.
# ---------------------------------------------------------------------------

function _verlet_tracked_systems(sys::Tuple, ints::Tuple)
    extra = Any[]
    for inter in ints
        sb = inter.system_b
        sb === nothing && continue
        any(ps -> ps === sb, sys) && continue
        push!(extra, sb)
    end
    return (sys..., extra...)
end

_verlet_ref_bufs(tracked::Tuple)     = map(ps -> copy(ps.x), tracked)
_verlet_disp_scratch(tracked::Tuple) = map(ps -> similar(ps.x), tracked)

_reset_verlet_refs!(::Tuple{}, ::Tuple{}) = nothing
@inline function _reset_verlet_refs!(tracked::Tuple, refs::Tuple)
    copyto!(first(refs), first(tracked).x)
    _reset_verlet_refs!(Base.tail(tracked), Base.tail(refs))
end

_max_verlet_displacement(::Tuple{}, ::Tuple{}, ::Tuple{}, running::T) where {T} = running
@inline function _max_verlet_displacement(tracked::Tuple, refs::Tuple, scratches::Tuple, running::T) where {T}
    ps, refbuf, scratch = first(tracked), first(refs), first(scratches)
    n = ps.n
    d = n == 0 ? zero(T) : T(maximum(norm, (view(scratch, 1:n) .= view(ps.x, 1:n) .- view(refbuf, 1:n))))
    _max_verlet_displacement(Base.tail(tracked), Base.tail(refs), Base.tail(scratches), max(running, d))
end

# Auto-zero all virtual systems' w_sum and ZF fields before the stage loop.
_auto_zero_all_virtual!(::Tuple{}) = nothing
@inline function _auto_zero_all_virtual!(vsys::Tuple)
    auto_zero_virtual!(first(vsys))
    _auto_zero_all_virtual!(Base.tail(vsys))
end

# Advance virtual particle positions by prescribed_v·dt (zero for fixed boundaries).
_update_virtual_positions!(::Tuple{}, dt) = nothing
@inline function _update_virtual_positions!(vsys::Tuple, dt)
    vps = first(vsys)
    pv  = getfield(vps, :prescribed_v)
    iszero(pv) || _axpy_const_ip!(vps.x, pv, dt)
    _update_virtual_positions!(Base.tail(vsys), dt)
end

# Run one full sweep pass: auto-zero virtuals, then per-stage state updates,
# ghost updates, virtual state updates, and interaction sweeps.
function _sweep_all_stages!(sys, virtual_sys, ghosts, ints, num_stages, to, ps_labels, ghost_labels, inter_labels, dt)
    _auto_zero_all_virtual!(virtual_sys)
    for stage in 1:num_stages
        for (i, ps) in enumerate(sys)
            length(ps.state_updater) == num_stages || continue
            @timeit to ps_labels[i].upd @timeit to ps_labels[i].name update_state!(ps, stage, dt)
        end
        for (i, ge) in enumerate(ghosts)
            @timeit to ghost_labels[i].stage @timeit to ghost_labels[i].name update_ghost!(ge, stage)
        end
        for vps in virtual_sys
            length(vps.state_updater) == num_stages || continue
            update_state!(vps, stage, dt)
        end
        for (i, inter) in enumerate(ints)
            @timeit to inter_labels[i].sweep @timeit to inter_labels[i].name sweep!(inter, stage)
        end
    end
end

# XSPH velocity correction: subtract the accumulated v_adjustment, re-run the
# XSPH sweep via adjust_v!, then add the freshly computed adjustment back.
#
# Skipped entirely when no interaction has a velocity-adjust pfn: v_adjustment
# then stays permanently zero (nothing ever writes it), so the whole sequence
# is mathematically a no-op — subtract 0, zero an already-zero field, run a
# sweep that dispatches to a no-op stub for pfn=nothing, add 0 back. Guarding
# it skips those launches rather than actually performing a no-op.
function _xsph_correction!(sys, ints, to, ps_labels, inter_labels)
    any(inter -> inter.vadjust_pfn !== nothing, ints) || return nothing
    for (i, ps) in enumerate(sys)
        @timeit to ps_labels[i].v_adjust @timeit to ps_labels[i].name _axpy_ip!(ps.v, ps.v_adjustment, -1)
    end
    for (i, ps) in enumerate(sys)
        @timeit to ps_labels[i].v_adjust @timeit to ps_labels[i].name _zero_field(ps, :v_adjustment)
    end
    for (i, inter) in enumerate(ints)
        @timeit to inter_labels[i].v_adjust @timeit to inter_labels[i].name adjust_v!(inter)
    end
    for (i, ps) in enumerate(sys)
        @timeit to ps_labels[i].v_adjust @timeit to ps_labels[i].name _axpy_ip!(ps.v, ps.v_adjustment, 1)
    end
end

# Advance probe positions by prescribed_v·dt each timestep.
_advance_probe_positions!(::Tuple{}, dt) = nothing
@inline function _advance_probe_positions!(probes::Tuple, dt)
    probe = first(probes)
    pv = getfield(probe, :prescribed_v)
    iszero(pv) || _axpy_const_ip!(getfield(probe, :x), pv, dt)
    _advance_probe_positions!(Base.tail(probes), dt)
end

# Scatter src_x[i] into dst_x[src_id[i]] — src_id is always a permutation of
# 1:n, so every thread/iteration writes a distinct slot; no aliasing hazard.
# Backend-dispatched like every other array-level primitive in this file:
# a plain scalar loop on CPU, a dedicated KA kernel (KAKernels.jl) elsewhere —
# a Base fancy-indexing scatter (`dst_x[src_id] = src_x`) would work for CUDA
# specifically but isn't guaranteed across arbitrary KA backends.
@inline _mirror_positions!(dst_x, src_x, src_id) =
    _mirror_positions!(KA.get_backend(dst_x), dst_x, src_x, src_id)

@inline function _mirror_positions!(::KA.CPU, dst_x, src_x, src_id)
    @inbounds @batch for i in eachindex(src_id)
        dst_x[src_id[i]] = src_x[i]
    end
end

function _mirror_positions!(backend::KA.Backend, dst_x, src_x, src_id)
    n = length(src_id)
    n == 0 && return nothing
    _probe_mirror_kernel!(backend, _KA_WORKGROUP)(dst_x, src_x, src_id; ndrange = n)
    KA.synchronize(backend)
    return nothing
end

# Measure all probes: mirror → sort-by-cell → grids → zero → sweep → update → sort-by-id.
# Called only at save cadence, so the per-step cost is zero.
function _measure_probes!(probes, probe_ints, sort_cutoff, perm_buf, key_buf, probe_scratches)
    # Re-sort each unique source system that appears in probe interactions.
    # Source positions advanced since the step-start sort; create_grid! requires
    # pre-sorted inputs, so we re-sort here (save-cadence allocation is acceptable).
    sorted_sources = IdDict{Any,Bool}()
    for pint in probe_ints
        src = pint.system_a
        haskey(sorted_sources, src) && continue
        sorted_sources[src] = true
        sort_particles!(src, sort_cutoff, perm_buf, key_buf, _make_sort_scratch(src))
    end
    # Mirror probe positions from source.  probe.id == 1:n is invariant at entry
    # (maintained by _sort_probe_by_id! at end of previous measurement), so
    # probe.x[source.id[i]] = source.x[i] correctly maps each source particle to
    # the probe slot that tracks its original identity.
    for probe in probes
        mt = getfield(probe, :mirror_target)
        if mt !== nothing
            _mirror_positions!(getfield(probe, :x), getfield(mt, :x), getfield(mt, :id))
        end
    end
    # Sort probes by cell so create_grid! can build a CSR grid
    for (i, probe) in enumerate(probes)
        sort_particles!(probe, sort_cutoff, perm_buf, key_buf, probe_scratches[i])
    end
    # Build interaction grids for probe interactions
    for pint in probe_ints
        create_grid!(pint)
    end
    # Zero accumulators, sweep, then run state updaters
    for probe in probes
        auto_zero_probe!(probe)
    end
    for pint in probe_ints
        sweep!(pint, 1)
    end
    for probe in probes
        update_state!(probe)
    end
    # Sort by id so HDF5 row k always maps to original probe k
    for (i, probe) in enumerate(probes)
        _sort_probe_by_id!(probe, perm_buf, probe_scratches[i])
    end
end

# Print a per-system summary at the requested interval.
function _maybe_print!(sys, to, global_step, print_interval_step, dt)
    if global_step % print_interval_step == 0
        @timeit to "print summary" begin
            sim_time = global_step * dt
            println("\nStep $global_step (t = $(@sprintf("%.6g", sim_time)))")
            for ps in sys
                print_summary(_to_host(ps))
            end
        end
    end
end

# Write an HDF5 snapshot at the requested interval.
# Probe measurement (mirror → sort-by-cell → sweep → sort-by-id) happens here,
# inside the save guard, so there is zero per-step cost when no save occurs.
#
# Returns `true` iff probes were measured this step — `_measure_probes!`
# re-sorts each probe source system independently of `time_integrate!`'s own
# rebuild-cadence gate, which invalidates that system's Verlet-skin reference
# snapshot (the "same index = same particle" assumption it relies on no
# longer holds once an out-of-band re-sort happens). Callers with
# `verlet_skin > 0` must force a full rebuild + reference reset on the next
# step when this returns `true`.
function _maybe_save!(sys, ghosts, virtual_sys, probes, probe_ints, probe_scratches,
                      sort_cutoff, perm_buf, key_buf,
                      to, global_step, save_interval_step, output_prefix, width, dt)
    if output_prefix !== nothing && global_step % save_interval_step == 0
        @timeit to "save h5" begin
            isempty(probes) ||
                _measure_probes!(probes, probe_ints, sort_cutoff, perm_buf, key_buf, probe_scratches)

            path = "$(output_prefix)_$(lpad(global_step, width, '0')).h5"
            d    = dirname(path)
            !isempty(d) && mkpath(d)
            h5open(path, "w") do f
                HDF5.attrs(f)["step"]     = global_step
                HDF5.attrs(f)["sim_time"] = Float64(global_step * dt)
                for ps in sys
                    write_h5(_to_host(ps), create_group(f, ps.name))
                end
                for ge in ghosts
                    write_h5(_to_host(ge.ghost), create_group(f, ge.ghost.name))
                end
                for vps in virtual_sys
                    write_h5(_to_host(vps), create_group(f, vps.name))
                end
                for probe in probes
                    write_h5(_to_host(probe), create_group(f, probe.name))
                end
            end
        end
        return !isempty(probes)
    end
    return false
end

# ---------------------------------------------------------------------------
# Integration loop
# ---------------------------------------------------------------------------

"""
    time_integrate!(integrator, num_timesteps, print_interval_step,
                    save_interval_step, CFL, output_prefix;
                    step_offset=0, output_width=nothing, print_timer=true, to=TimerOutput())

Run the leapfrog loop for `num_timesteps` steps.

- `CFL`: Courant number; timestep is `dt = CFL * h / c`.
- `print_interval_step`: print a per-system summary every this many steps.
- `save_interval_step`: write HDF5 snapshots every this many steps.
- `output_prefix`: path prefix for HDF5 output, e.g. `"output/run"`.
  Files are named `"\$(prefix)_\$(step).h5"` with zero-padded step numbers.
  Pass `nothing` to disable saving.
- `step_offset`: global step number before this batch starts (for continuous
  file numbering and interval checks across multiple calls). Default `0`.
- `output_width`: minimum zero-padding width for step numbers in filenames.
  `nothing` (default) derives the width from `ndigits(step_offset + num_timesteps)`,
  preserving single-call behaviour. Pass an explicit value (e.g. from `run_driver!`)
  to keep padding consistent across stages.
- `print_timer`: print a timing breakdown to `stdout` on completion. Default `true`.
  Pass `false` when calling from `run_driver!`, which handles printing itself.
- `to`: a `TimerOutput` object to record timings into. If not provided, a new one is created.

Returns the `TimerOutput` for this batch.
"""
function time_integrate!(
    integrator::LeapFrogTimeIntegrator,
    num_timesteps::Int,
    print_interval_step::Int,
    save_interval_step::Int,
    CFL::Real,
    output_prefix;
    step_offset::Int             = 0,
    output_width::Union{Nothing,Int} = nothing,
    print_timer::Bool            = true,
    to::TimerOutput              = TimerOutput(),
)
    sys   = integrator.systems
    ints  = integrator.interactions
    vsys  = integrator.virtual_systems
    prbs  = integrator.probes
    pints = integrator.probe_interactions
    T     = typeof(integrator.c)
    dt    = T(CFL) * integrator.h / integrator.c
    Γ     = integrator.Γ

    num_stages = length(integrator.interactions[1].pfns)
    @assert all(length(inter.pfns) == num_stages for inter in integrator.interactions) "All interactions must have the same number of stages (pfns length), got: $(map(inter -> length(inter.pfns), integrator.interactions))"
    for ps in sys
        n_upd = length(ps.state_updater)
        if n_upd != num_stages
            @warn "ParticleSystem \"$(ps.name)\" has $n_upd state updater(s) but num_stages=$num_stages; stages $(n_upd + 1) and later will skip the state update"
        end
    end

    q0_bufs = map(_make_q0_bufs, sys)

    verlet_skin       = integrator.verlet_skin
    sort_cutoff       = T(2) * integrator.h + verlet_skin
    sort_max_n        = maximum(ps.n for ps in sys)
    sort_perm_buf     = similar(first(sys).x, Int, sort_max_n)
    sort_key_buf      = similar(first(sys).x, UInt64, sort_max_n)
    sys_scratches     = map(_make_sort_scratch, sys)
    ghost_scratches   = [_make_empty_sort_scratch(ge.ghost) for ge in integrator.ghosts]
    virtual_scratches = [_make_sort_scratch(vps) for vps in vsys]
    probe_scratches   = [_make_sort_scratch(probe) for probe in prbs]

    # Verlet-skin rebuild-cadence gate state; unused (zero overhead) when
    # verlet_skin == 0 — see _validate_verlet_skin and the helpers above.
    verlet_tracked  = _verlet_tracked_systems(sys, ints)
    verlet_refs     = _verlet_ref_bufs(verlet_tracked)
    verlet_scratch  = _verlet_disp_scratch(verlet_tracked)
    force_rebuild   = true   # step 1 always rebuilds — nothing has been built yet

    ps_labels = [(name=ps.name,
                  sort="sort",
                  mid="half-step",
                  full="full-step",
                  pos="update pos",
                  upd="state update",
                  v_adjust="vel adjust") for ps in sys]

    inter_labels = []
    for inter in ints
        ps_a  = inter.system_a
        label = is_coupled(inter) ? "$(ps_a.name)×$(inter.system_b.name)" : ps_a.name
        push!(inter_labels, (name=label, grid="grid", sweep="sweep", v_adjust="vel adjust"))
    end

    ghost_labels = [(name=ge.ghost.name,
                     gen="ghost gen",
                     sort="ghost sort",
                     kin="ghost kinematics",
                     stage="ghost stage") for ge in integrator.ghosts]

    width = output_width === nothing ? ndigits(step_offset + num_timesteps) : output_width

    for itimestep in 1:num_timesteps
        global_step = step_offset + itimestep

        # ---- 1, 3-5. Sort + rebuild grids — gated by the Verlet-skin cadence
        # check when verlet_skin > 0 (always true when verlet_skin == 0, i.e.
        # today's exact behaviour). force_rebuild covers step 1 and any step
        # immediately after a probe measurement re-sorted a tracked system
        # out of band (see _maybe_save!'s docstring).
        need_rebuild = if iszero(verlet_skin)
            true
        elseif force_rebuild
            true
        else
            2 * _max_verlet_displacement(verlet_tracked, verlet_refs, verlet_scratch, zero(T)) > verlet_skin
        end
        if need_rebuild
            _sort_all_systems!(sys, sys_scratches, sort_cutoff, sort_perm_buf, sort_key_buf, to, ps_labels, 1)
            _prepare_grids!(integrator.ghosts, vsys, ints, sort_cutoff, sort_perm_buf, sort_key_buf,
                            ghost_scratches, virtual_scratches, to, ghost_labels, inter_labels, verlet_skin)
            iszero(verlet_skin) || _reset_verlet_refs!(verlet_tracked, verlet_refs)
            force_rebuild = false
        end

        # ---- 2. Save initial values ----------------------------------------
        @timeit to "save q0" _save_q0_all!(sys, q0_bufs)

        # ---- 6. Half-step --------------------------------------------------
        for (i, ps) in enumerate(sys)
            @timeit to ps_labels[i].mid @timeit to ps_labels[i].name _halfstep_ps!(ps, dt / 2)
        end

        # ---- 7. Update ghost kinematics (v, rho) ---------------------------
        for (i, ge) in enumerate(integrator.ghosts)
            @timeit to ghost_labels[i].kin @timeit to ghost_labels[i].name update_ghost_kinematics!(ge)
        end

        # ---- 8. Sweep (auto-zeros virtual fields before stage loop) --------
        _sweep_all_stages!(sys, vsys, integrator.ghosts, ints, num_stages, to,
                           ps_labels, ghost_labels, inter_labels, dt)

        # ---- 8b. Velocity damping: dvdt -= (Γ/dt) * v ----------------------
        iszero(Γ) || _apply_damping_all!(sys, Γ / dt)

        # ---- 9. Full-step: update q = q0 + dt·dqdt -------------------------
        _fullstep_q_all!(sys, q0_bufs, dt, to, ps_labels, 1)

        # ---- 10. XSPH velocity correction -----------------------------------
        _xsph_correction!(sys, ints, to, ps_labels, inter_labels)

        # ---- 11. Update positions -------------------------------------------
        _update_positions_all!(sys, dt, to, ps_labels, 1)
        _update_virtual_positions!(vsys, dt)
        _advance_probe_positions!(prbs, dt)

        # ---- 12. Print ------------------------------------------------------
        _maybe_print!(sys, to, global_step, print_interval_step, dt)

        # ---- 13. Save -------------------------------------------------------
        measured_probes = _maybe_save!(sys, integrator.ghosts, vsys, prbs, pints, probe_scratches,
                     sort_cutoff, sort_perm_buf, sort_key_buf,
                     to, global_step, save_interval_step, output_prefix, width, dt)
        measured_probes && (force_rebuild = true)
    end

    print_timer && show(to; allocations=true, compact=false)
    return to
end

# ---------------------------------------------------------------------------
# RK4 time_integrate!
# ---------------------------------------------------------------------------

"""
    time_integrate!(integrator::RK4TimeIntegrator, ...)

RK4 variant.  Signature identical to the LeapFrog version (including `output_width`).

The neighbour grid is built once per timestep (frozen Lagrangian).
The four RK stages share the same grid; intermediate states are formed by
advancing q from q0 using the previous stage's dqdt.
"""
function time_integrate!(
    integrator::RK4TimeIntegrator,
    num_timesteps::Int,
    print_interval_step::Int,
    save_interval_step::Int,
    CFL::Real,
    output_prefix;
    step_offset::Int             = 0,
    output_width::Union{Nothing,Int} = nothing,
    print_timer::Bool            = true,
    to::TimerOutput              = TimerOutput(),
)
    sys   = integrator.systems
    ints  = integrator.interactions
    vsys  = integrator.virtual_systems
    prbs  = integrator.probes
    pints = integrator.probe_interactions
    T     = typeof(integrator.c)
    dt    = T(CFL) * integrator.h / integrator.c
    Γ     = integrator.Γ

    num_stages = length(integrator.interactions[1].pfns)
    @assert all(length(inter.pfns) == num_stages for inter in integrator.interactions) "All interactions must have the same number of stages (pfns length), got: $(map(inter -> length(inter.pfns), integrator.interactions))"
    for ps in sys
        n_upd = length(ps.state_updater)
        if n_upd != num_stages
            @warn "ParticleSystem \"$(ps.name)\" has $n_upd state updater(s) but num_stages=$num_stages; stages $(n_upd + 1) and later will skip the state update"
        end
    end

    rk4_advance = (T(0),   T(0.5), T(0.5), T(1.0))
    rk4_weight  = (T(1/6), T(1/3), T(1/3), T(1/6))

    q0_bufs  = map(_make_q0_bufs,  sys)
    acc_bufs = map(_make_acc_bufs, sys)

    verlet_skin       = integrator.verlet_skin
    sort_cutoff       = T(2) * integrator.h + verlet_skin
    sort_max_n        = maximum(ps.n for ps in sys)
    sort_perm_buf     = similar(first(sys).x, Int, sort_max_n)
    sort_key_buf      = similar(first(sys).x, UInt64, sort_max_n)
    sys_scratches     = map(_make_sort_scratch, sys)
    ghost_scratches   = [_make_empty_sort_scratch(ge.ghost) for ge in integrator.ghosts]
    virtual_scratches = [_make_sort_scratch(vps) for vps in vsys]
    probe_scratches   = [_make_sort_scratch(probe) for probe in prbs]

    # Verlet-skin rebuild-cadence gate state; unused (zero overhead) when
    # verlet_skin == 0 — see _validate_verlet_skin and the helpers above.
    verlet_tracked  = _verlet_tracked_systems(sys, ints)
    verlet_refs     = _verlet_ref_bufs(verlet_tracked)
    verlet_scratch  = _verlet_disp_scratch(verlet_tracked)
    force_rebuild   = true   # step 1 always rebuilds — nothing has been built yet

    ps_labels = [(name=ps.name,
                  sort="sort",
                  mid="rk stage",
                  full="rk apply",
                  pos="update pos",
                  upd="state update",
                  v_adjust="vel adjust") for ps in sys]

    inter_labels = []
    for inter in ints
        ps_a  = inter.system_a
        label = is_coupled(inter) ? "$(ps_a.name)×$(inter.system_b.name)" : ps_a.name
        push!(inter_labels, (name=label, grid="grid", sweep="sweep", v_adjust="vel adjust"))
    end

    ghost_labels = [(name=ge.ghost.name,
                     gen="ghost gen",
                     sort="ghost sort",
                     kin="ghost kinematics",
                     stage="ghost stage") for ge in integrator.ghosts]

    width = output_width === nothing ? ndigits(step_offset + num_timesteps) : output_width

    for itimestep in 1:num_timesteps
        global_step = step_offset + itimestep

        # ---- 1, 3-5. Sort + rebuild grids — gated by the Verlet-skin cadence
        # check when verlet_skin > 0 (always true when verlet_skin == 0, i.e.
        # today's exact behaviour, including the "frozen across 4 RK stages"
        # guarantee already documented above). force_rebuild covers step 1 and
        # any step immediately after a probe measurement re-sorted a tracked
        # system out of band (see _maybe_save!'s docstring).
        need_rebuild = if iszero(verlet_skin)
            true
        elseif force_rebuild
            true
        else
            2 * _max_verlet_displacement(verlet_tracked, verlet_refs, verlet_scratch, zero(T)) > verlet_skin
        end
        if need_rebuild
            @timeit to "sort" _sort_all_systems!(sys, sys_scratches, sort_cutoff, sort_perm_buf, sort_key_buf, to, ps_labels, 1)
            _prepare_grids!(integrator.ghosts, vsys, ints, sort_cutoff, sort_perm_buf, sort_key_buf,
                            ghost_scratches, virtual_scratches, to, ghost_labels, inter_labels, verlet_skin)
            iszero(verlet_skin) || _reset_verlet_refs!(verlet_tracked, verlet_refs)
            force_rebuild = false
        end

        # ---- 2. Save q0 and zero accumulators -------------------------------
        @timeit to "save q0" _save_q0_all!(sys, q0_bufs)
        _zero_acc_all!(sys, acc_bufs)

        # ---- 6-8. Four RK stages --------------------------------------------
        for rk_iter in 1:4
            rk_label = "rk$rk_iter"

            if rk_iter == 1
                @timeit to "rk stage" @timeit to rk_label _reset_dqdt_all!(sys)
            else
                _rk4_advance_all!(sys, q0_bufs, rk4_advance[rk_iter] * dt, to, ps_labels, 1)
            end

            for (i, ge) in enumerate(integrator.ghosts)
                @timeit to ghost_labels[i].kin @timeit to ghost_labels[i].name update_ghost_kinematics!(ge)
            end

            @timeit to "sweep" @timeit to rk_label _sweep_all_stages!(
                sys, vsys, integrator.ghosts, ints, num_stages, to, ps_labels, ghost_labels, inter_labels, dt)

            iszero(Γ) || _apply_damping_all!(sys, Γ / dt)

            _acc_dqdt_all!(sys, acc_bufs, rk4_weight[rk_iter])
        end

        # ---- 9. Apply accumulated RK4 update --------------------------------
        _apply_acc_all!(sys, q0_bufs, acc_bufs, dt, to, ps_labels, 1)

        # ---- 10. XSPH velocity correction ------------------------------------
        _xsph_correction!(sys, ints, to, ps_labels, inter_labels)

        # ---- 11. Update positions --------------------------------------------
        _update_positions_all!(sys, dt, to, ps_labels, 1)
        _update_virtual_positions!(vsys, dt)
        _advance_probe_positions!(prbs, dt)

        # ---- 12. Print -------------------------------------------------------
        _maybe_print!(sys, to, global_step, print_interval_step, dt)

        # ---- 13. Save --------------------------------------------------------
        measured_probes = _maybe_save!(sys, integrator.ghosts, vsys, prbs, pints, probe_scratches,
                     sort_cutoff, sort_perm_buf, sort_key_buf,
                     to, global_step, save_interval_step, output_prefix, width, dt)
        measured_probes && (force_rebuild = true)
    end

    print_timer && show(to; allocations=true, compact=false)
    return to
end
