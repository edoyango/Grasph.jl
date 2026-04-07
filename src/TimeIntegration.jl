export LeapFrogTimeIntegrator, time_integrate!, run_driver!

# ---------------------------------------------------------------------------
# Integrator struct
# ---------------------------------------------------------------------------

"""
    LeapFrogTimeIntegrator

Leapfrog time integrator for one or more `AbstractParticleSystem`s.

Conjugate pairs `(q, dqdt)` are driven by `ps.pairs`. At each step:

1. Save each `q`.
2. **Half-step**: `q += 0.5 dt * dqdt`, reset `dqdt ← source`, call
   `update_state!`.
3. Rebuild the cell list and sweep all interactions.
4. **Full-step**: `q = q₀ + dt * dqdt`, `x += dt * v`.
"""
struct LeapFrogTimeIntegrator{SYS<:Tuple, INTS<:Tuple, GHOSTS<:Tuple, T<:AbstractFloat}
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

# ---------------------------------------------------------------------------
# Per-system step helpers — generic over ps.pairs
# ---------------------------------------------------------------------------

# Type-stable field accessor: S is a compile-time constant, so getfield is fully inferred.
@inline _getf(ps, ::Val{S}) where {S} = getfield(ps, S)

# Source (reset) value for each known derivative field; zero for user-added pairs.
# Called with a Val{S} dqdt key, so dispatch is always type-stable.
@inline _source_for(ps::AbstractParticleSystem, ::Val{:dvdt})          = ps.source_v
@inline _source_for(ps::AbstractParticleSystem, ::Val{:drhodt})        = ps.source_rho
@inline _source_for(ps::AbstractParticleSystem, ::Val{name}) where {name} =
    zero(eltype(getfield(ps, name)))

# Barrier functions: type-stable hot loops called with concrete array types.
@inline function _halfstep_pair!(q, dqdt, src, half_dt)
    @inbounds @fastmath @batch for i in eachindex(q)
        q[i]    += half_dt * dqdt[i]
        dqdt[i]  = src
    end
end

@inline function _fullstep_pair!(q, q0, dqdt, dt)
    @inbounds @fastmath @batch for i in eachindex(q)
        q[i] = q0[i] + dt * dqdt[i]
    end
end

# Type-stable tuple walk over ps.pairs (which stores Val-encoded field names).
# Base.tail recursion ensures first(pairs) always has a concrete type — the same
# pattern used for _sweep_pfns! and _update_state_pfns!.
_halfstep_pairs!(ps, ::Tuple{}, half_dt) = nothing
@inline function _halfstep_pairs!(ps, pairs::Tuple, half_dt)
    q_val, dqdt_val = first(pairs)
    _halfstep_pair!(_getf(ps, q_val), _getf(ps, dqdt_val), _source_for(ps, dqdt_val), half_dt)
    _halfstep_pairs!(ps, Base.tail(pairs), half_dt)
end
@inline _halfstep_ps!(ps::AbstractParticleSystem, half_dt) =
    _halfstep_pairs!(ps, getfield(ps, :pairs), half_dt)

# Build a typed tuple of q0 buffers (one copy per pair).
# Returning a Tuple rather than a Vector preserves element types for _fullstep_pairs!.
_make_q0_bufs(ps, ::Tuple{}) = ()
@inline function _make_q0_bufs(ps, pairs::Tuple)
    q_val = first(first(pairs))
    (copy(_getf(ps, q_val)), _make_q0_bufs(ps, Base.tail(pairs))...)
end
@inline _make_q0_bufs(ps::AbstractParticleSystem) = _make_q0_bufs(ps, getfield(ps, :pairs))

@inline function _parallel_copy!(dst, src)
    @inbounds @batch for i in eachindex(dst)
        dst[i] = src[i]
    end
end

# Save current q values into the pre-allocated buffers.
_save_q0_pairs!(ps, ::Tuple{}, ::Tuple{}) = nothing
@inline function _save_q0_pairs!(ps, pairs::Tuple, bufs::Tuple)
    q_val = first(first(pairs))
    _parallel_copy!(first(bufs), _getf(ps, q_val))
    _save_q0_pairs!(ps, Base.tail(pairs), Base.tail(bufs))
end

# Full-step: q = q0 + dt * dqdt, walking pairs and bufs in lockstep.
_fullstep_pairs!(ps, ::Tuple{}, ::Tuple{}, dt) = nothing
@inline function _fullstep_pairs!(ps, pairs::Tuple, bufs::Tuple, dt)
    q_val, dqdt_val = first(pairs)
    _fullstep_pair!(_getf(ps, q_val), first(bufs), _getf(ps, dqdt_val), dt)
    _fullstep_pairs!(ps, Base.tail(pairs), Base.tail(bufs), dt)
end
@inline _fullstep_ps!(ps::AbstractParticleSystem, q0_bufs, dt) =
    _fullstep_pairs!(ps, getfield(ps, :pairs), q0_bufs, dt)

@inline function _update_positions!(ps::AbstractParticleSystem, dt)
    x = ps.x; v = ps.v
    @inbounds @fastmath @batch for i in eachindex(x)
        x[i] += dt * v[i]
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
    # _make_q0_bufs uses Base.tail recursion so each buffer has a concrete type.
    q0_bufs = [_make_q0_bufs(ps) for ps in sys]

    # Pre-compute @timeit labels to avoid string interpolation allocations in the loop
    ps_labels = [(mid="mid-step [$(ps.name)]", 
                  full="full-step [$(ps.name)]", 
                  upd="state update [$(ps.name)]") for ps in sys]
    
    inter_labels = []
    for inter in ints
        ps_a  = inter.system_a
        label = is_coupled(inter) ? "$(ps_a.name)×$(inter.system_b.name)" : ps_a.name
        push!(inter_labels, (grid="grid [$label]", sweep="sweep [$label]"))
    end

    ghost_labels = [(gen="ghost gen [$(ge.ghost.name)]", 
                     kin="ghost kin [$(ge.ghost.name)]", 
                     stage="ghost stage [$(ge.ghost.name)]") for ge in integrator.ghosts]

    width = ndigits(step_offset + num_timesteps)

    for itimestep in 1:num_timesteps
        global_step = step_offset + itimestep

        # ---- 1. Save initial values ----------------------------------------
        for (i, ps) in enumerate(sys)
            _save_q0_pairs!(ps, getfield(ps, :pairs), q0_bufs[i])
        end

        # ---- 2. Generate ghosts (positions only) ---------------------------
        for (i, ge) in enumerate(integrator.ghosts)
            @timeit to ghost_labels[i].gen generate_ghosts!(ge)
        end

        # ---- 3. Create grids -----------------------------------------------
        for (i, inter) in enumerate(ints)
            @timeit to inter_labels[i].grid create_grid!(inter)
        end

        # ---- 4. Half-step --------------------------------------------------
        for (i, ps) in enumerate(sys)
            @timeit to ps_labels[i].mid _halfstep_ps!(ps, dt / 2)
        end

        # ---- 5. Update ghost kinematics (v, rho) ---------------------------
        for (i, ge) in enumerate(integrator.ghosts)
            @timeit to ghost_labels[i].kin update_ghost_kinematics!(ge)
        end

        # ---- 6. Sweep ------------------------------------------------------
        for stage in 1:num_stages
            for (i, ps) in enumerate(sys)
                length(ps.state_updater) == num_stages || continue
                @timeit to ps_labels[i].upd update_state!(ps, stage)
            end

            for (i, ge) in enumerate(integrator.ghosts)
                @timeit to ghost_labels[i].stage update_ghost!(ge, stage)
            end

            for (i, inter) in enumerate(ints)
                @timeit to inter_labels[i].sweep sweep!(inter, stage)
            end
        end

        # ---- 7. Full-step --------------------------------------------------
        for (i, ps) in enumerate(sys)
            @timeit to ps_labels[i].full begin
                _fullstep_ps!(ps, q0_bufs[i], dt)
                _update_positions!(ps, dt)
            end
        end

        # ---- 8. Print ------------------------------------------------------
        if global_step % print_interval_step == 0
            @timeit to "print summary" begin
                println("\nStep $global_step")
                for ps in sys
                    print_summary(ps)
                end
            end
        end

        # ---- 9. Save -------------------------------------------------------
        if output_prefix !== nothing && global_step % save_interval_step == 0
            @timeit to "save h5" begin
                path = "$(output_prefix)_$(lpad(global_step, width, '0')).h5"
                d    = dirname(path)
                !isempty(d) && mkpath(d)
                h5open(path, "w") do f
                    for ps in sys
                        write_h5(ps, create_group(f, ps.name))
                    end
                    for ge in integrator.ghosts
                        write_h5(ge.ghost, create_group(f, ge.ghost.name))
                    end
                end
            end
        end
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

High-level simulation driver wrapping `LeapFrogTimeIntegrator`.

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
    integrator::LeapFrogTimeIntegrator,
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

    # Parse CLI flags from ARGS if any
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--steps" || arg == "-s"
            if i + 1 <= length(ARGS)
                cur_n = parse(Int, ARGS[i+1])
                i += 1
            end
        elseif arg == "--print-freq" || arg == "-p"
            if i + 1 <= length(ARGS)
                cur_print = parse(Int, ARGS[i+1])
                i += 1
            end
        elseif arg == "--save-freq" || arg == "-f"
            if i + 1 <= length(ARGS)
                cur_save = parse(Int, ARGS[i+1])
                i += 1
            end
        elseif arg == "--cfl" || arg == "-c"
            if i + 1 <= length(ARGS)
                cur_CFL = parse(Float64, ARGS[i+1])
                i += 1
            end
        elseif arg == "--non-interactive" || arg == "-n"
            cur_interactive = false
        end
        i += 1
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
