export AbstractTimeIntegrator, LeapFrogTimeIntegrator, RK4TimeIntegrator, time_integrate!, run_driver!

using ArgParse

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
struct LeapFrogTimeIntegrator{SYS<:Tuple, INTS<:Tuple, GHOSTS<:Tuple, T<:AbstractFloat} <: AbstractTimeIntegrator
    systems::SYS
    interactions::INTS
    ghosts::GHOSTS
    c::T
    h::T
end

"""
    LeapFrogTimeIntegrator(systems, interactions; ghosts=()) -> LeapFrogTimeIntegrator

Construct a `LeapFrogTimeIntegrator`.

- `systems`: an `AbstractParticleSystem` or iterable of `AbstractParticleSystem`s.
- `interactions`: a `SystemInteraction` or iterable of `SystemInteraction`s.

Raises `ArgumentError` if either collection is empty.
"""
function LeapFrogTimeIntegrator(systems, interactions; ghosts=())
    sys  = systems      isa AbstractParticleSystem ? (systems,)      : Tuple(systems)
    ints = interactions isa SystemInteraction      ? (interactions,) : Tuple(interactions)
    gsts = ghosts       isa GhostEntry             ? (ghosts,)       : Tuple(ghosts)
    isempty(sys)  && throw(ArgumentError("systems must not be empty"))
    isempty(ints) && throw(ArgumentError("interactions must not be empty"))
    T = eltype(eltype(first(sys).x))
    c = T(maximum(ps.c           for ps   in sys))
    h = T(minimum(inter.kernel.h for inter in ints))
    LeapFrogTimeIntegrator{typeof(sys), typeof(ints), typeof(gsts), T}(sys, ints, gsts, c, h)
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
struct RK4TimeIntegrator{SYS<:Tuple, INTS<:Tuple, GHOSTS<:Tuple, T<:AbstractFloat} <: AbstractTimeIntegrator
    systems::SYS
    interactions::INTS
    ghosts::GHOSTS
    c::T
    h::T
end

function RK4TimeIntegrator(systems, interactions; ghosts=())
    sys  = systems      isa AbstractParticleSystem ? (systems,)      : Tuple(systems)
    ints = interactions isa SystemInteraction      ? (interactions,) : Tuple(interactions)
    gsts = ghosts       isa GhostEntry             ? (ghosts,)       : Tuple(ghosts)
    isempty(sys)  && throw(ArgumentError("systems must not be empty"))
    isempty(ints) && throw(ArgumentError("interactions must not be empty"))
    T = eltype(eltype(first(sys).x))
    c = T(maximum(ps.c           for ps   in sys))
    h = T(minimum(inter.kernel.h for inter in ints))
    RK4TimeIntegrator{typeof(sys), typeof(ints), typeof(gsts), T}(sys, ints, gsts, c, h)
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

# Generate ghosts, sort them, then build all interaction grids.
# Steps 3-5 are identical for both LeapFrog and RK4 since the grid is
# (re)built exactly once per timestep in both integrators.
function _prepare_grids!(ghosts, ints, sort_cutoff, sort_perm_buf, sort_key_buf,
                          ghost_scratches, to, ghost_labels, inter_labels)
    for (i, ge) in enumerate(ghosts)
        @timeit to ghost_labels[i].gen @timeit to ghost_labels[i].name generate_ghosts!(ge)
    end
    for (i, ge) in enumerate(ghosts)
        @timeit to ghost_labels[i].sort @timeit to ghost_labels[i].name sort_particles!(
            ge.ghost, sort_cutoff, sort_perm_buf, sort_key_buf, ghost_scratches[i])
    end
    for (i, inter) in enumerate(ints)
        @timeit to inter_labels[i].grid @timeit to inter_labels[i].name create_grid!(inter)
    end
end

# Run one full sweep pass: state updates, ghost stage updates, then interaction
# sweeps — in that order, for every stage.  Called once per timestep in
# LeapFrog; called once per RK stage in RK4 (wrapped in extra @timeit there).
function _sweep_all_stages!(sys, ghosts, ints, num_stages, to, ps_labels, ghost_labels, inter_labels)
    for stage in 1:num_stages
        for (i, ps) in enumerate(sys)
            length(ps.state_updater) == num_stages || continue
            @timeit to ps_labels[i].upd @timeit to ps_labels[i].name update_state!(ps, stage)
        end
        for (i, ge) in enumerate(ghosts)
            @timeit to ghost_labels[i].stage @timeit to ghost_labels[i].name update_ghost!(ge, stage)
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
function _maybe_save!(sys, ghosts, to, global_step, save_interval_step, output_prefix, width)
    if output_prefix !== nothing && global_step % save_interval_step == 0
        @timeit to "save h5" begin
            path = "$(output_prefix)_$(lpad(global_step, width, '0')).h5"
            d    = dirname(path)
            !isempty(d) && mkpath(d)
            h5open(path, "w") do f
                for ps in sys
                    write_h5(ps, create_group(f, ps.name))
                end
                for ge in ghosts
                    write_h5(ge.ghost, create_group(f, ge.ghost.name))
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
    T     = typeof(integrator.c)
    dt    = T(CFL) * integrator.h / integrator.c

    num_stages = length(integrator.interactions[1].pfns)
    @assert all(length(inter.pfns) == num_stages for inter in integrator.interactions) "All interactions must have the same number of stages (pfns length), got: $(map(inter -> length(inter.pfns), integrator.interactions))"
    for ps in sys
        n_upd = length(ps.state_updater)
        if n_upd != num_stages
            @warn "ParticleSystem \"$(ps.name)\" has $n_upd state updater(s) but num_stages=$num_stages; stages $(n_upd + 1) and later will skip the state update"
        end
    end

    # Pre-allocate q0 buffers: one typed tuple of arrays per system.
    # map on a Tuple preserves element types (unlike a comprehension, which
    # would produce Vector{Any} when systems have different concrete types).
    q0_bufs = map(_make_q0_bufs, sys)

    # Pre-allocate sort infrastructure.  All interactions share the same cutoff
    # (enforced at construction), so 2h is the canonical cell-lattice spacing.
    # perm_buf and key_buf are shared across all sort calls within a timestep.
    # sys_scratches is a tuple of scratch tuples (one per real system, fixed size).
    # ghost_scratches is a vector of scratch tuples (one per ghost, resized each step).
    sort_cutoff    = T(2) * integrator.h
    sort_nd        = first(sys).ndims
    sort_max_n     = maximum(ps.n for ps in sys)
    sort_perm_buf  = Vector{Int}(undef, sort_max_n)
    sort_key_buf   = Vector{SVector{sort_nd,Int}}(undef, sort_max_n)
    sys_scratches  = map(_make_sort_scratch, sys)
    ghost_scratches = [_make_empty_sort_scratch(ge.ghost) for ge in integrator.ghosts]

    # Pre-compute @timeit labels to avoid string interpolation allocations in the loop.
    # Sub-labels are short because they appear nested under the parent name.
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

        # ---- 1. Sort real particle systems by cell index -------------------
        # Sorting before saving q0 keeps q0 and dqdt in the same index space
        # throughout the step, so the full-step q = q0 + dt·dqdt is correct.
        # Positions come from the previous step's update, so spatially nearby
        # particles are already nearly sorted — the cost is low.
        _sort_all_systems!(sys, sys_scratches, sort_cutoff, sort_perm_buf, sort_key_buf, to, ps_labels, 1)

        # ---- 2. Save initial values ----------------------------------------
        @timeit to "save q0" _save_q0_all!(sys, q0_bufs)

        # ---- 3-5. Generate ghosts, sort, build grids -----------------------
        _prepare_grids!(integrator.ghosts, ints, sort_cutoff, sort_perm_buf, sort_key_buf,
                        ghost_scratches, to, ghost_labels, inter_labels)

        # ---- 6. Half-step --------------------------------------------------
        for (i, ps) in enumerate(sys)
            @timeit to ps_labels[i].mid @timeit to ps_labels[i].name _halfstep_ps!(ps, dt / 2)
        end

        # ---- 7. Update ghost kinematics (v, rho) ---------------------------
        for (i, ge) in enumerate(integrator.ghosts)
            @timeit to ghost_labels[i].kin @timeit to ghost_labels[i].name update_ghost_kinematics!(ge)
        end

        # ---- 8. Sweep ------------------------------------------------------
        _sweep_all_stages!(sys, integrator.ghosts, ints, num_stages, to,
                           ps_labels, ghost_labels, inter_labels)

        # ---- 9. Full-step: update q = q0 + dt·dqdt -----------------------
        _fullstep_q_all!(sys, q0_bufs, dt, to, ps_labels, 1)

        # ---- 10. XSPH velocity correction ----------------------------------
        _xsph_correction!(sys, ints, to, ps_labels, inter_labels)

        # ---- 11. Update positions: x += dt·v -------------------------------
        _update_positions_all!(sys, dt, to, ps_labels, 1)

        # ---- 12. Print -----------------------------------------------------
        _maybe_print!(sys, to, global_step, print_interval_step, dt)

        # ---- 13. Save ------------------------------------------------------
        _maybe_save!(sys, integrator.ghosts, to, global_step, save_interval_step, output_prefix, width)
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
    sys  = integrator.systems
    ints = integrator.interactions
    T    = typeof(integrator.c)
    dt   = T(CFL) * integrator.h / integrator.c

    num_stages = length(integrator.interactions[1].pfns)
    @assert all(length(inter.pfns) == num_stages for inter in integrator.interactions) "All interactions must have the same number of stages (pfns length), got: $(map(inter -> length(inter.pfns), integrator.interactions))"
    for ps in sys
        n_upd = length(ps.state_updater)
        if n_upd != num_stages
            @warn "ParticleSystem \"$(ps.name)\" has $n_upd state updater(s) but num_stages=$num_stages; stages $(n_upd + 1) and later will skip the state update"
        end
    end

    # RK4 stage coefficients (typed to T to avoid conversions in the loop).
    # advance[s]: multiply dt by this to get the intermediate state from q0.
    # weight[s]:  accumulation weight for the weighted sum.
    rk4_advance = (T(0),   T(0.5), T(0.5), T(1.0))
    rk4_weight  = (T(1/6), T(1/3), T(1/3), T(1/6))

    q0_bufs  = map(_make_q0_bufs,  sys)
    acc_bufs = map(_make_acc_bufs, sys)

    sort_cutoff      = T(2) * integrator.h
    sort_nd          = first(sys).ndims
    sort_perm_buf    = Vector{Int}(undef, maximum(ps.n for ps in sys))
    sort_key_buf     = Vector{SVector{sort_nd,Int}}(undef, maximum(ps.n for ps in sys))
    sys_scratches    = map(_make_sort_scratch, sys)
    ghost_scratches  = [_make_empty_sort_scratch(ge.ghost) for ge in integrator.ghosts]

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

        # ---- 1. Sort real particle systems by cell index -------------------
        @timeit to "sort" _sort_all_systems!(sys, sys_scratches, sort_cutoff, sort_perm_buf, sort_key_buf, to, ps_labels, 1)

        # ---- 2. Save q0 and zero accumulators ------------------------------
        @timeit to "save q0" _save_q0_all!(sys, q0_bufs)
        _zero_acc_all!(sys, acc_bufs)

        # ---- 3-5. Generate ghosts, sort, build grids -----------------------
        _prepare_grids!(integrator.ghosts, ints, sort_cutoff, sort_perm_buf, sort_key_buf,
                        ghost_scratches, to, ghost_labels, inter_labels)

        # ---- 6-8. Four RK stages -------------------------------------------
        for rk_iter in 1:4
            rk_label = "rk$rk_iter"

            # ---- 6. Advance q from q0 (stages 2-4); always reset dqdt ------
            # Stage 1: q is already q0; just reset dqdt to source.
            # Stages 2-4: q = q0 + advance_coeff*dt * dqdt, then reset dqdt.
            if rk_iter == 1
                @timeit to "rk stage" @timeit to rk_label _reset_dqdt_all!(sys)
            else
                _rk4_advance_all!(sys, q0_bufs, rk4_advance[rk_iter] * dt, to, ps_labels, 1)
            end

            # ---- 7. Update ghost kinematics --------------------------------
            for (i, ge) in enumerate(integrator.ghosts)
                @timeit to ghost_labels[i].kin @timeit to ghost_labels[i].name update_ghost_kinematics!(ge)
            end

            # ---- 8. State updates then sweeps (all updates before any sweep)
            @timeit to "sweep" @timeit to rk_label _sweep_all_stages!(
                sys, integrator.ghosts, ints, num_stages, to, ps_labels, ghost_labels, inter_labels)

            # Accumulate weighted dqdt into acc_bufs.
            _acc_dqdt_all!(sys, acc_bufs, rk4_weight[rk_iter])
        end

        # ---- 9. Apply accumulated RK4 update: q = q0 + dt * acc -----------
        _apply_acc_all!(sys, q0_bufs, acc_bufs, dt, to, ps_labels, 1)

        # ---- 10. XSPH velocity correction -----------------------------------
        _xsph_correction!(sys, ints, to, ps_labels, inter_labels)

        # ---- 11. Update positions: x += dt·v --------------------------------
        _update_positions_all!(sys, dt, to, ps_labels, 1)

        # ---- 12. Print -------------------------------------------------------
        _maybe_print!(sys, to, global_step, print_interval_step, dt)

        # ---- 13. Save --------------------------------------------------------
        _maybe_save!(sys, integrator.ghosts, to, global_step, save_interval_step, output_prefix, width)
    end

    print_timer && show(to; allocations=true, compact=false)
    return to
end

# ---------------------------------------------------------------------------
# Interactive driver helpers
# ---------------------------------------------------------------------------

function _prompt_int(label::AbstractString, default::Int)
    print("  $label [$default]: ")
    s = strip(readline())
    isempty(s) && return default
    v = tryparse(Int, s)
    v === nothing && (println("  Invalid, keeping $default."); return default)
    return v
end

function _prompt_float(label::AbstractString, default::Float64)
    print("  $label [$default]: ")
    s = strip(readline())
    isempty(s) && return default
    v = tryparse(Float64, s)
    v === nothing && (println("  Invalid, keeping $default."); return default)
    return v
end

function _prompt_prefix(default)
    dflt_str = default === nothing ? "none" : string(default)
    print("  Output prefix [$dflt_str] (\"none\" to disable): ")
    s = strip(readline())
    isempty(s) && return default
    s == "none" && return nothing
    return s
end

# ---------------------------------------------------------------------------
# run_driver!
# ---------------------------------------------------------------------------

"""
    run_driver!(integrator, num_timesteps, print_interval_step,
                save_interval_step, CFL, output_prefix; interactive=true)

High-level simulation driver wrapping any `AbstractTimeIntegrator`.

## Interactive mode (`interactive=true`, default)

Runs `num_timesteps` steps, prints a per-batch timing summary, then asks:

    Steps to continue [0 to stop]:

Enter a positive integer to keep going, or 0 (or non-integer) to stop.
After entering a step count the driver asks whether to change settings;
entering `y` prompts for each of `print_interval_step`, `save_interval_step`,
`CFL`, and `output_prefix` — press Enter to keep the current value.

A cumulative timing table is printed when the simulation finishes.

## Non-interactive mode (`interactive=false`)

Runs `num_timesteps` steps and exits. Equivalent to `time_integrate!` but
with the same global step-counter bookkeeping (HDF5 files are numbered from
`step_offset + 1`).

## HDF5 file numbering

Files are always numbered by the *global* step counter, so they are
contiguous across multiple interactive batches.
"""
function run_driver!(
    integrator::AbstractTimeIntegrator,
    num_timesteps::Int,
    print_interval_step::Int,
    save_interval_step::Int,
    CFL::Real,
    output_prefix;
    interactive::Bool = true,
)
    global_step = 0
    to          = TimerOutput()
    cur_n       = num_timesteps
    cur_print   = print_interval_step
    cur_save    = save_interval_step
    cur_CFL     = Float64(CFL)
    cur_prefix  = output_prefix
    cur_interactive = interactive

    # Parse CLI flags from ARGS if any.
    # Defaults shown in the help text are the values passed to run_driver!.
    if !isempty(ARGS)
        ap = ArgParseSettings(; description = "Grasph SPH driver")
        @add_arg_table! ap begin
            "--steps", "-s"
                help    = "number of timesteps to run"
                arg_type = Int
                default = cur_n
            "--print-freq", "-p"
                help    = "print every N steps"
                arg_type = Int
                default = cur_print
            "--save-freq", "-f"
                help    = "save every N steps"
                arg_type = Int
                default = cur_save
            "--cfl", "-c"
                help    = "CFL number"
                arg_type = Float64
                default = cur_CFL
            "--non-interactive", "-n"
                help    = "disable interactive prompt between batches"
                action  = :store_true
            "--output-prefix", "-o"
                help    = "output file prefix"
                arg_type = String
                default = cur_prefix
        end
        parsed = parse_args(ARGS, ap)
        cur_n           = parsed["steps"]
        cur_print       = parsed["print-freq"]
        cur_save        = parsed["save-freq"]
        cur_CFL         = parsed["cfl"]
        cur_prefix      = parsed["output-prefix"]
        cur_interactive = interactive && !parsed["non-interactive"]
    end

    while true
        time_integrate!(
            integrator, cur_n, cur_print, cur_save, cur_CFL, cur_prefix;
            step_offset = global_step,
            print_timer = false,
            to          = to,
        )
        global_step += cur_n

        println("\n--- Batch complete (total steps: $global_step) ---")
        show(to; allocations=true, compact=false)
        println()

        cur_interactive || break

        print("\nSteps to continue [0 to stop]: ")
        line   = strip(readline())
        n_more = tryparse(Int, line)
        (n_more === nothing || n_more <= 0) && break
        cur_n = n_more

        print("Change settings? [y/N]: ")
        if strip(readline()) in ("y", "Y")
            cur_print  = _prompt_int("Print every N steps", cur_print)
            cur_save   = _prompt_int("Save every N steps",  cur_save)
            cur_CFL    = _prompt_float("CFL", cur_CFL)
            cur_prefix = _prompt_prefix(cur_prefix)
        end
    end

    println("\n=== Accumulated timing ===")
    show(to; allocations=true, compact=false)
    println()
    nothing
end
