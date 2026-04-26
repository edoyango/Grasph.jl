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
end

"""
    LeapFrogTimeIntegrator(systems, interactions; ghosts=(), virtual_systems=()) -> LeapFrogTimeIntegrator

Construct a `LeapFrogTimeIntegrator`.

- `systems`: an `AbstractParticleSystem` or iterable of `AbstractParticleSystem`s.
- `interactions`: a `SystemInteraction` or iterable of `SystemInteraction`s.
- `virtual_systems`: `VirtualParticleSystem`s that are sorted and state-updated each step
  but whose velocity is zeroed before position integration (fixed boundaries).

Raises `ArgumentError` if either collection is empty.
"""
function LeapFrogTimeIntegrator(systems, interactions; ghosts=(), virtual_systems=(), probes=(), probe_interactions=(), Γ=0)
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
    LeapFrogTimeIntegrator{typeof(sys), typeof(ints), typeof(gsts), typeof(vsys), typeof(prbs), typeof(pints), T}(
        sys, ints, gsts, vsys, prbs, pints, c, h, T(Γ))
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
end

function RK4TimeIntegrator(systems, interactions; ghosts=(), virtual_systems=(), probes=(), probe_interactions=(), Γ=0)
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
    RK4TimeIntegrator{typeof(sys), typeof(ints), typeof(gsts), typeof(vsys), typeof(prbs), typeof(pints), T}(
        sys, ints, gsts, vsys, prbs, pints, c, h, T(Γ))
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

# Snapshot current q values into the pre-allocated buffers (buf = q + 0*q).
_save_q0_pairs!(ps, ::Tuple{}, ::Tuple{}) = nothing
@inline function _save_q0_pairs!(ps, pairs::Tuple, bufs::Tuple)
    q_val = first(first(pairs))
    _axpy_oop!(first(bufs), _getf(ps, q_val), _getf(ps, q_val), 0)
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
    @inbounds for i in eachindex(v)
        dvdt[i] -= Γ_dt * v[i]
    end
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
function _prepare_grids!(ghosts, virtual_sys, ints, sort_cutoff, sort_perm_buf, sort_key_buf,
                          ghost_scratches, virtual_scratches, to, ghost_labels, inter_labels)
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
        @timeit to inter_labels[i].grid @timeit to inter_labels[i].name create_grid!(inter)
    end
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
    @inbounds for i in 1:vps.n
        vps.x[i] += pv * dt
    end
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
function _xsph_correction!(sys, ints, to, ps_labels, inter_labels)
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
    if !iszero(pv)
        x = getfield(probe, :x)
        @inbounds for i in 1:probe.n
            x[i] += pv * dt
        end
    end
    _advance_probe_positions!(Base.tail(probes), dt)
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
            src_id = getfield(mt, :id)
            src_x  = getfield(mt, :x)
            x = getfield(probe, :x)
            @inbounds for i in 1:probe.n
                x[src_id[i]] = src_x[i]
            end
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
                print_summary(ps)
            end
        end
    end
end

# Write an HDF5 snapshot at the requested interval.
# Probe measurement (mirror → sort-by-cell → sweep → sort-by-id) happens here,
# inside the save guard, so there is zero per-step cost when no save occurs.
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
                    write_h5(ps, create_group(f, ps.name))
                end
                for ge in ghosts
                    write_h5(ge.ghost, create_group(f, ge.ghost.name))
                end
                for vps in virtual_sys
                    write_h5(vps, create_group(f, vps.name))
                end
                for probe in probes
                    write_h5(probe, create_group(f, probe.name))
                end
            end
        end
    end
end

# ---------------------------------------------------------------------------
# Integration loop
# ---------------------------------------------------------------------------

"""
    time_integrate!(integrator, num_timesteps, print_interval_step,
                    save_interval_step, CFL, output_prefix;
                    step_offset=0, print_timer=true, to=TimerOutput())

Run the leapfrog loop for `num_timesteps` steps.

- `CFL`: Courant number; timestep is `dt = CFL * h / c`.
- `print_interval_step`: print a per-system summary every this many steps.
- `save_interval_step`: write HDF5 snapshots every this many steps.
- `output_prefix`: path prefix for HDF5 output, e.g. `"output/run"`.
  Files are named `"\$(prefix)_\$(step).h5"` with zero-padded step numbers.
  Pass `nothing` to disable saving.
- `step_offset`: global step number before this batch starts (for continuous
  file numbering and interval checks across multiple calls). Default `0`.
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
    step_offset::Int  = 0,
    print_timer::Bool = true,
    to::TimerOutput   = TimerOutput(),
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

    sort_cutoff       = T(2) * integrator.h
    sort_nd           = first(sys).ndims
    sort_max_n        = maximum(ps.n for ps in sys)
    sort_perm_buf     = Vector{Int}(undef, sort_max_n)
    sort_key_buf      = Vector{SVector{sort_nd,Int}}(undef, sort_max_n)
    sys_scratches     = map(_make_sort_scratch, sys)
    ghost_scratches   = [_make_empty_sort_scratch(ge.ghost) for ge in integrator.ghosts]
    virtual_scratches = [_make_sort_scratch(vps) for vps in vsys]
    probe_scratches   = [_make_sort_scratch(probe) for probe in prbs]

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

    width = ndigits(step_offset + num_timesteps)

    for itimestep in 1:num_timesteps
        global_step = step_offset + itimestep

        # ---- 1. Sort real particle systems ---------------------------------
        _sort_all_systems!(sys, sys_scratches, sort_cutoff, sort_perm_buf, sort_key_buf, to, ps_labels, 1)

        # ---- 2. Save initial values ----------------------------------------
        @timeit to "save q0" _save_q0_all!(sys, q0_bufs)

        # ---- 3-5. Generate ghosts, sort ghosts + virtuals, build grids -----
        _prepare_grids!(integrator.ghosts, vsys, ints, sort_cutoff, sort_perm_buf, sort_key_buf,
                        ghost_scratches, virtual_scratches, to, ghost_labels, inter_labels)

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
        _maybe_save!(sys, integrator.ghosts, vsys, prbs, pints, probe_scratches,
                     sort_cutoff, sort_perm_buf, sort_key_buf,
                     to, global_step, save_interval_step, output_prefix, width, dt)
    end

    print_timer && show(to; allocations=true, compact=false)
    return to
end

# ---------------------------------------------------------------------------
# RK4 time_integrate!
# ---------------------------------------------------------------------------

"""
    time_integrate!(integrator::RK4TimeIntegrator, ...)

RK4 variant.  Signature identical to the LeapFrog version.

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
    step_offset::Int  = 0,
    print_timer::Bool = true,
    to::TimerOutput   = TimerOutput(),
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

    sort_cutoff       = T(2) * integrator.h
    sort_nd           = first(sys).ndims
    sort_perm_buf     = Vector{Int}(undef, maximum(ps.n for ps in sys))
    sort_key_buf      = Vector{SVector{sort_nd,Int}}(undef, maximum(ps.n for ps in sys))
    sys_scratches     = map(_make_sort_scratch, sys)
    ghost_scratches   = [_make_empty_sort_scratch(ge.ghost) for ge in integrator.ghosts]
    virtual_scratches = [_make_sort_scratch(vps) for vps in vsys]
    probe_scratches   = [_make_sort_scratch(probe) for probe in prbs]

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

    width = ndigits(step_offset + num_timesteps)

    for itimestep in 1:num_timesteps
        global_step = step_offset + itimestep

        # ---- 1. Sort real particle systems ----------------------------------
        @timeit to "sort" _sort_all_systems!(sys, sys_scratches, sort_cutoff, sort_perm_buf, sort_key_buf, to, ps_labels, 1)

        # ---- 2. Save q0 and zero accumulators -------------------------------
        @timeit to "save q0" _save_q0_all!(sys, q0_bufs)
        _zero_acc_all!(sys, acc_bufs)

        # ---- 3-5. Generate ghosts, sort ghosts + virtuals, build grids ------
        _prepare_grids!(integrator.ghosts, vsys, ints, sort_cutoff, sort_perm_buf, sort_key_buf,
                        ghost_scratches, virtual_scratches, to, ghost_labels, inter_labels)

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
        _maybe_save!(sys, integrator.ghosts, vsys, prbs, pints, probe_scratches,
                     sort_cutoff, sort_perm_buf, sort_key_buf,
                     to, global_step, save_interval_step, output_prefix, width, dt)
    end

    print_timer && show(to; allocations=true, compact=false)
    return to
end
