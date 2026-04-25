export Stage, run_driver!

using ArgParse

"""
    Stage(integrator, num_steps, CFL, label)

One phase of a multi-stage simulation driven by [`run_driver!`](@ref).

- `integrator`: any `AbstractTimeIntegrator`.
- `num_steps`: timesteps to run in this stage.
- `CFL`: Courant number; timestep is `dt = CFL * h / c`.
- `label`: short identifier (letters/digits/`-`/`_`) used to name this stage's
  CLI flags, e.g. `label="damping"` produces `--damping-steps` and `--damping-cfl`.
"""
struct Stage{I<:AbstractTimeIntegrator}
    integrator::I
    num_steps::Int
    CFL::Float64
    label::String
end

Stage(integrator::AbstractTimeIntegrator, num_steps::Integer, CFL::Real, label::AbstractString) =
    Stage(integrator, Int(num_steps), Float64(CFL), String(label))

_effective_stages(stages) = [s for s in stages if s !== nothing]

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

# Unique particle systems across all effective stages, preserving first-seen order.
function _unique_systems(stages_eff)
    seen = IdDict{Any,Bool}()
    out  = Any[]
    for st in stages_eff
        for ps in st.integrator.systems
            if !haskey(seen, ps)
                seen[ps] = true
                push!(out, ps)
            end
        end
    end
    return out
end

# Load particle state from an HDF5 checkpoint into the systems used by `stages_eff`.
# Returns the global step read from the file.
function _load_restart!(stages_eff, path::AbstractString)
    restart_step = 0
    h5open(path, "r") do f
        if haskey(HDF5.attrs(f), "step")
            restart_step = Int(HDF5.attrs(f)["step"][])
        else
            @warn "Restart file $path has no \"step\" attribute; assuming step 0"
        end
        for ps in _unique_systems(stages_eff)
            if haskey(f, ps.name)
                read_h5!(ps, f[ps.name])
            else
                @warn "Restart file $path has no group for system \"$(ps.name)\"; leaving it untouched"
            end
        end
    end
    return restart_step
end

# Parse CLI flags shared by all stages and per-stage --<label>-steps / --<label>-cfl.
# Returns a NamedTuple of resolved values. When ARGS is empty, the supplied
# Julia defaults are returned unchanged.
function _parse_driver_args(stages_eff, print_freq, save_freq, output_prefix,
                            interactive, restart)
    resolved_steps  = [st.num_steps for st in stages_eff]
    resolved_cfls   = [st.CFL       for st in stages_eff]
    resolved_print  = print_freq
    resolved_save   = save_freq
    resolved_prefix = output_prefix
    resolved_inter  = interactive
    resolved_restart = restart

    isempty(ARGS) && return (; steps = resolved_steps, cfls = resolved_cfls,
                              print_freq = resolved_print, save_freq = resolved_save,
                              output_prefix = resolved_prefix,
                              interactive = resolved_inter,
                              restart = resolved_restart)

    ap = ArgParseSettings(; description = "Grasph SPH multi-stage driver")
    @add_arg_table! ap begin
        "--print-freq", "-p"
            help    = "print every N steps"
            arg_type = Int
            default = resolved_print
        "--save-freq", "-f"
            help    = "save every N steps"
            arg_type = Int
            default = resolved_save
        "--output-prefix", "-o"
            help    = "output file prefix (pass \"none\" to disable)"
            arg_type = String
            default  = resolved_prefix === nothing ? "none" : String(resolved_prefix)
        "--non-interactive", "-n"
            help    = "disable the tail \"steps to continue\" prompt"
            action  = :store_true
        "--restart", "-r"
            help    = "resume from an HDF5 checkpoint (reads the \"step\" attr)"
            arg_type = String
            default  = resolved_restart === nothing ? "" : String(resolved_restart)
    end
    for (i, st) in enumerate(stages_eff)
        add_arg_table!(ap,
            "--$(st.label)-steps",
            Dict(:help => "steps for stage \"$(st.label)\"",
                 :arg_type => Int,
                 :default  => resolved_steps[i]))
        add_arg_table!(ap,
            "--$(st.label)-cfl",
            Dict(:help => "CFL for stage \"$(st.label)\"",
                 :arg_type => Float64,
                 :default  => resolved_cfls[i]))
    end

    parsed = parse_args(ARGS, ap)
    resolved_print  = parsed["print-freq"]
    resolved_save   = parsed["save-freq"]
    prefix_str      = parsed["output-prefix"]
    resolved_prefix = (prefix_str == "none" || isempty(prefix_str)) ? nothing : prefix_str
    resolved_inter  = interactive && !parsed["non-interactive"]
    rs = parsed["restart"]
    resolved_restart = isempty(rs) ? nothing : rs
    for (i, st) in enumerate(stages_eff)
        resolved_steps[i] = parsed["$(st.label)-steps"]
        resolved_cfls[i]  = parsed["$(st.label)-cfl"]
    end

    return (; steps = resolved_steps, cfls = resolved_cfls,
              print_freq = resolved_print, save_freq = resolved_save,
              output_prefix = resolved_prefix,
              interactive = resolved_inter,
              restart = resolved_restart)
end

"""
    run_driver!(stages, print_interval_step, save_interval_step, output_prefix;
                interactive=true, restart=nothing)

Multi-stage simulation driver.

- `stages`: iterable of `Union{Stage, Nothing}`. `nothing` entries are skipped
  and have no CLI flags generated for them. Each `Stage` carries its own
  integrator, step count, and CFL.
- `print_interval_step`, `save_interval_step`, `output_prefix`: shared across
  all stages. HDF5 files are numbered by the cumulative global step.
- `interactive`: when `true`, after the final stage the driver prompts
  `Steps to continue [0 to stop]:` and extends the final stage. No prompt
  between stages.
- `restart`: path to an HDF5 file produced by a prior run. The `step`
  attribute sets the global step counter; particle state is loaded via
  `read_h5!`; stages whose cumulative end ≤ restart step are skipped and the
  straddling stage resumes for the remainder.

## CLI knobs

Each `Stage` with label `L` adds `--L-steps N` and `--L-cfl F`. Global flags:
`--print-freq`, `--save-freq`, `--output-prefix`, `--non-interactive`, and
`--restart PATH`. When `ARGS` is empty, CLI parsing is skipped.
"""
function run_driver!(
    stages,
    print_interval_step::Int,
    save_interval_step::Int,
    output_prefix;
    interactive::Bool = true,
    restart::Union{AbstractString, Nothing} = nothing,
)
    stages_eff = _effective_stages(stages)
    isempty(stages_eff) && throw(ArgumentError("run_driver! requires at least one non-nothing Stage"))

    # Warn on duplicate labels — they'd collide on CLI.
    labels = [st.label for st in stages_eff]
    if length(unique(labels)) != length(labels)
        throw(ArgumentError("stage labels must be unique, got: $labels"))
    end

    parsed = _parse_driver_args(stages_eff, print_interval_step, save_interval_step,
                                output_prefix, interactive, restart)

    cur_print   = parsed.print_freq
    cur_save    = parsed.save_freq
    cur_prefix  = parsed.output_prefix
    cur_interactive = parsed.interactive

    # Apply restart (if any) and determine per-stage offsets / remaining steps.
    global_step = 0
    if parsed.restart !== nothing
        global_step = _load_restart!(stages_eff, parsed.restart)
        println("Resumed from $(parsed.restart) at global step $global_step")
    end

    to = TimerOutput()

    cumulative = 0
    for (i, st) in enumerate(stages_eff)
        n_steps   = parsed.steps[i]
        cfl       = parsed.cfls[i]
        stage_end = cumulative + n_steps

        if n_steps <= 0
            println("\n=== Stage \"$(st.label)\" skipped (num_steps=$n_steps) ===")
            cumulative = stage_end
            continue
        end

        if global_step >= stage_end
            println("\n=== Stage \"$(st.label)\" already completed in restart (ends at step $stage_end) ===")
            cumulative = stage_end
            continue
        end

        remaining = stage_end - max(global_step, cumulative)
        println("\n=== Stage \"$(st.label)\" ($remaining steps, CFL=$cfl) ===")

        time_integrate!(
            st.integrator, remaining, cur_print, cur_save, cfl, cur_prefix;
            step_offset = global_step,
            print_timer = false,
            to          = to,
        )
        global_step += remaining
        cumulative   = stage_end

        println("\n--- Stage \"$(st.label)\" complete (total steps: $global_step) ---")
        show(to; allocations=true, compact=false)
        println()
    end

    # Tail prompt extends the last effective stage's integrator/CFL.
    if cur_interactive
        last_stage = stages_eff[end]
        cur_CFL    = parsed.cfls[end]
        while true
            print("\nSteps to continue [0 to stop]: ")
            line   = strip(readline())
            n_more = tryparse(Int, line)
            (n_more === nothing || n_more <= 0) && break

            print("Change settings? [y/N]: ")
            if strip(readline()) in ("y", "Y")
                cur_print  = _prompt_int("Print every N steps", cur_print)
                cur_save   = _prompt_int("Save every N steps",  cur_save)
                cur_CFL    = _prompt_float("CFL", cur_CFL)
                cur_prefix = _prompt_prefix(cur_prefix)
            end

            println("\n=== Extending stage \"$(last_stage.label)\" ($n_more steps, CFL=$cur_CFL) ===")
            time_integrate!(
                last_stage.integrator, n_more, cur_print, cur_save, cur_CFL, cur_prefix;
                step_offset = global_step,
                print_timer = false,
                to          = to,
            )
            global_step += n_more

            println("\n--- Extension complete (total steps: $global_step) ---")
            show(to; allocations=true, compact=false)
            println()
        end
    end

    println("\n=== Accumulated timing ===")
    show(to; allocations=true, compact=false)
    println()
    nothing
end
