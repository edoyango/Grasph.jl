using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm, dot

# ---------------------------------------------------------------------------
# Cluster C extension: in-context validation of `onesided=true` for
# Trapdoor.jl's interaction shape — the richest in the whole script survey.
#
# Mirrors the established pattern in test/test_onesided_sweep.jl (sections 5-6,
# `_dambreak_like` / "short-run trajectory equivalence" / "long-run physical
# invariants"), but built around Trapdoor.jl's unique wiring instead:
#
#   - ElastoPlasticParticleSystem soil (ND=2, NS=4, 4-stage state_updater)
#   - TWO VirtualParticleSystems (bottom_virt; trapdoor_static_virt /
#     trapdoor_moving_virt sharing one source, differing only in prescribed_v)
#   - a free-slip, self-referencing GhostParticleSystem (walls_ghost, stress
#     reflected via HouseholderReflect)
#   - a Ghost×Virtual coupled SystemInteraction (ghost_bottom: system_a is
#     itself a ghost) — unique to this script
#   - ProbeParticleSystems fed by InterpolateFieldFn
#   - TWO chained integrators (settling phase w/ Γ damping, then moving phase)
#
# Trapdoor.jl is inherently 2D-only (HouseholderReflect's Voigt formulas and
# ElastoPlasticStressUpdater both hard-require ND=2/NS=4), so unlike
# `_dambreak_like` the builder below takes no `ndims` parameter.
#
# Geometry is a fully deterministic regular grid (Trapdoor.jl itself uses no
# randomness at all), scaled down ~15-20x from the real script:
#   real:    264×100 soil, 40-col trapdoor (112 cols left), 10 boundary layers
#   reduced:  16×8   soil,  4-col trapdoor (  6 cols left),  3 boundary layers
# ---------------------------------------------------------------------------

const _TD_dx             = 0.05
const _TD_h_sph          = 1.2 * _TD_dx
const _TD_rho0           = 1600.0
const _TD_art_visc_alpha = 0.1
const _TD_art_visc_beta  = 0.0

const _TD_E        = 10.0e6
const _TD_nu       = 0.33
const _TD_phi      = 39.0 * π / 180.0
const _TD_psi      = 19.0 * π / 180.0
const _TD_cohesion = 0.0

const _TD_c_sound = sqrt(_TD_E * (1 - _TD_nu) / (_TD_rho0 * (1 + _TD_nu) * (1 - 2*_TD_nu)))

const _TD_nx       = 16    # soil columns  (real script: 264)
const _TD_ny       = 8     # soil rows     (real script: 100)
const _TD_n_layers = 3     # boundary thickness, in particle layers (real: 10)
const _TD_ntd_x    = 4     # trapdoor width, columns (real: 40)
const _TD_nleft_x  = (_TD_nx - _TD_ntd_x) ÷ 2
const _TD_pad      = 1     # bottom-flank overhang beyond soil left/right edges (real: 5)
const _TD_trapdoor_vel = -0.005
const _TD_n_probes = 3     # stress probes across the trapdoor top (real: 5)

# ---------------------------------------------------------------------------
# Builder — called independently (not deepcopy'd) for the "old"/coloured and
# "new"/onesided runs, so both start from bit-identical state with no
# aliasing risk between the two.
# ---------------------------------------------------------------------------

function _trapdoor_like(onesided::Bool)
    dx, h_sph, rho0, c_sound = _TD_dx, _TD_h_sph, _TD_rho0, _TD_c_sound
    nx, ny, n_layers = _TD_nx, _TD_ny, _TD_n_layers
    ntd_x, nleft_x, pad = _TD_ntd_x, _TD_nleft_x, _TD_pad

    n_soil    = nx * ny
    soil_mass = rho0 * dx * dx

    soil = ElastoPlasticParticleSystem(
        "soil", n_soil, 2, 4, soil_mass, c_sound;
        source_v      = [0.0, -9.81],
        state_updater = (
            nothing,
            ZeroFieldUpdater(:strain_rate, :vorticity),
            ElastoPlasticStressUpdater(_TD_E, _TD_nu, _TD_phi, _TD_psi, _TD_cohesion),
            nothing,
        ),
    )
    let k = 1
        for i in 0:nx-1, j in 0:ny-1
            soil.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx)
            k += 1
        end
    end
    fill!(soil.v, zero(SVector{2,Float64}))
    soil.rho .= rho0
    update_state!(soil, 3)

    # --- Static bottom boundary (left + right flanks, outside trapdoor) ---
    n_bottom = (nx - ntd_x + 2*pad) * n_layers
    bottom_source = StressParticleSystem("bottom_source", n_bottom, 2, 4, soil_mass, c_sound)
    let k = 1
        for j in 1:n_layers
            for i in -pad:nleft_x-1                     # left flank
                bottom_source.x[k] = SVector((i + 0.5) * dx, -(j - 0.5) * dx)
                k += 1
            end
            for i in nleft_x+ntd_x:nx+pad-1              # right flank
                bottom_source.x[k] = SVector((i + 0.5) * dx, -(j - 0.5) * dx)
                k += 1
            end
        end
    end
    bottom_source.rho .= rho0
    fill!(bottom_source.v, zero(SVector{2,Float64}))

    _trapdoor_updater = (
        nothing,
        (VirtualNormUpdater(SVector(0.0, 0.0), :rho), PrescribedVelocityUpdater()),
        nothing,
        VirtualNormUpdater(SVector(0.0, 0.0), :stress),
    )

    bottom_virt = VirtualParticleSystem(
        bottom_source, "bottom_virt", n_bottom, 2, soil_mass, c_sound;
        zero_fields   = (:v, :rho, :stress),
        state_updater = _trapdoor_updater,
    )

    # --- Trapdoor boundary: two VirtualParticleSystems share one source ---
    n_trapdoor = ntd_x * n_layers
    trapdoor_source = StressParticleSystem("trapdoor_source", n_trapdoor, 2, 4, soil_mass, c_sound)
    let k = 1
        for j in 1:n_layers
            for i in nleft_x:nleft_x+ntd_x-1
                trapdoor_source.x[k] = SVector((i + 0.5) * dx, -(j - 0.5) * dx)
                k += 1
            end
        end
    end
    trapdoor_source.rho .= rho0
    fill!(trapdoor_source.v, zero(SVector{2,Float64}))

    trapdoor_static_virt = VirtualParticleSystem(
        trapdoor_source, "trapdoor_static_virt", n_trapdoor, 2, soil_mass, c_sound;
        zero_fields   = (:v, :rho, :stress),
        state_updater = _trapdoor_updater,
    )
    trapdoor_moving_virt = VirtualParticleSystem(
        trapdoor_source, "trapdoor_moving_virt", n_trapdoor, 2, soil_mass, c_sound;
        zero_fields   = (:v, :rho, :stress),
        prescribed_v  = SVector(0.0, _TD_trapdoor_vel),
        state_updater = _trapdoor_updater,
    )

    # --- Left/right ghost walls (free-slip: stress reflected each stage 3) ---
    walls_ghost = GhostParticleSystem(soil,
        nothing,
        nothing,
        GhostCopier(:stress => HouseholderReflect()),
        nothing,
    )
    walls_entry = GhostEntry(
        walls_ghost, 3.0 * h_sph,
        (SVector(1.0,  0.0), SVector(0.0, 0.0)),                   # left  wall at x = 0
        (SVector(-1.0, 0.0), SVector(Float64(nx * dx), 0.0)),      # right wall at x = nx*dx
    )

    # --- Interactions (4-stage tuples, matching soil's state_updater stages) ---
    kernel     = CubicSplineKernel(h_sph; ndims=2)
    sr_pfn     = StrainRateVorticityPfn()
    kin_pfn    = CauchyFluidPfn(_TD_art_visc_alpha, _TD_art_visc_beta, h_sph)
    interp_rho = InterpolateFieldFn(:rho; accumulate_wsum=true)
    interp_str = InterpolateFieldFn(:stress; accumulate_wsum=false)

    soil_self            = SystemInteraction(kernel, (nothing,    sr_pfn,  nothing,    kin_pfn), soil; onesided=onesided)
    soil_bottom          = SystemInteraction(kernel, (interp_rho, sr_pfn,  interp_str, kin_pfn), soil, bottom_virt; onesided=onesided)
    ghost_bottom         = SystemInteraction(kernel, (interp_rho, nothing, interp_str, nothing), walls_ghost, bottom_virt; onesided=onesided)
    soil_trapdoor_static = SystemInteraction(kernel, (interp_rho, sr_pfn,  interp_str, kin_pfn), soil, trapdoor_static_virt; onesided=onesided)
    soil_trapdoor_moving = SystemInteraction(kernel, (interp_rho, sr_pfn,  interp_str, kin_pfn), soil, trapdoor_moving_virt; onesided=onesided)
    soil_walls           = SystemInteraction(kernel, (nothing,    sr_pfn,  nothing,    kin_pfn), soil, walls_ghost; onesided=onesided)

    # --- Stress probes across the top of the trapdoor ---
    n_td_probes = _TD_n_probes
    td_probe_positions = [
        SVector(nleft_x * dx + (i-1) * (ntd_x * dx / (n_td_probes - 1)), 0.0)
        for i in 1:n_td_probes
    ]
    td_probe_updater = VirtualNormUpdater(SVector(1.0, 1.0), :stress)

    td_probe_static = ProbeParticleSystem(
        "td_probe_static", td_probe_positions;
        extras        = (stress = [zero(SVector{4,Float64}) for _ in 1:n_td_probes],),
        state_updater = td_probe_updater,
    )
    td_probe_moving = ProbeParticleSystem(
        "td_probe_moving", td_probe_positions;
        extras        = (stress = [zero(SVector{4,Float64}) for _ in 1:n_td_probes],),
        state_updater = td_probe_updater,
        prescribed_v  = SVector(0.0, _TD_trapdoor_vel),
    )
    probe_static_int = SystemInteraction(kernel, InterpolateFieldFn(:stress), soil, td_probe_static; onesided=onesided)
    probe_moving_int = SystemInteraction(kernel, InterpolateFieldFn(:stress), soil, td_probe_moving; onesided=onesided)

    # --- Integrators: settling (damped, static trapdoor) then moving ---
    integrator_static = LeapFrogTimeIntegrator(
        [soil],
        [soil_self, soil_bottom, ghost_bottom, soil_trapdoor_static, soil_walls];
        ghosts             = (walls_entry,),
        virtual_systems    = (bottom_virt, trapdoor_static_virt),
        probes             = (td_probe_static,),
        probe_interactions = (probe_static_int,),
        Γ                  = 0.002,
    )
    integrator_moving = LeapFrogTimeIntegrator(
        [soil],
        [soil_self, soil_bottom, ghost_bottom, soil_trapdoor_moving, soil_walls];
        ghosts             = (walls_entry,),
        virtual_systems    = (bottom_virt, trapdoor_moving_virt),
        probes             = (td_probe_moving,),
        probe_interactions = (probe_moving_int,),
    )

    return (
        soil = soil, n_soil = n_soil, n_bottom = n_bottom, n_trapdoor = n_trapdoor,
        bottom_virt = bottom_virt,
        trapdoor_static_virt = trapdoor_static_virt, trapdoor_moving_virt = trapdoor_moving_virt,
        walls_ghost = walls_ghost, walls_entry = walls_entry,
        integrator_static = integrator_static, integrator_moving = integrator_moving,
        td_probe_static = td_probe_static, td_probe_moving = td_probe_moving,
        probe_static_int = probe_static_int, probe_moving_int = probe_moving_int,
        h_sph = h_sph,
    )
end

# ---------------------------------------------------------------------------
# Probe measurement — normally only triggered inside `_maybe_save!` at save
# cadence (see TimeIntegration.jl). Our runs pass `output_prefix=nothing` to
# suppress all I/O, so we invoke the probe-measurement machinery directly
# (mirror -> sort-by-cell -> grid -> zero -> sweep -> state-update ->
# sort-by-id) at a controlled point instead of threading a real HDF5 save
# through the run.
# ---------------------------------------------------------------------------

function _measure_trapdoor_probes!(soil, probe, probe_int, h_sph)
    sort_cutoff   = 2.0 * h_sph
    perm_buf      = Vector{Int}(undef, soil.n)
    key_buf       = Vector{UInt64}(undef, soil.n)
    probe_scratch = Grasph._make_sort_scratch(probe)
    Grasph._measure_probes!((probe,), (probe_int,), sort_cutoff, perm_buf, key_buf, (probe_scratch,))
    return nothing
end

# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

_elemscale(v::AbstractVector{<:Real})    = max(maximum(abs, v), 1.0)
_elemscale(v::AbstractVector{<:SVector}) = max(maximum(norm, v), 1.0)
_elemdiff(a::AbstractVector{<:Real},    b::AbstractVector{<:Real})    = maximum(abs.(a .- b))
_elemdiff(a::AbstractVector{<:SVector}, b::AbstractVector{<:SVector}) = maximum(norm.(a .- b))

_allfinite(v::AbstractVector{<:Real})    = all(isfinite, v)
_allfinite(v::AbstractVector{<:SVector}) = all(x -> all(isfinite, x), v)

# ---------------------------------------------------------------------------
# Two-stage driver: settling (integrator_static, Γ-damped) then moving
# (integrator_moving), continuing from the same soil system — mirrors how
# Trapdoor.jl chains its two `Stage(...)` entries through `run_driver!`.
# ---------------------------------------------------------------------------

function _run_trapdoor_two_stage!(built, n_settle, n_move)
    # A print/save interval of just-past-this-stage's step count (as used in
    # the single-stage template) can coincidentally divide the *global* step
    # once `step_offset` is nonzero (stage 2 here) — use a value larger than
    # the combined step count so I/O never fires in either stage.
    quiet = n_settle + n_move + 1
    time_integrate!(built.integrator_static, n_settle, quiet, quiet, 0.1, nothing;
                     print_timer = false)
    time_integrate!(built.integrator_moving, n_move, quiet, quiet, 0.1, nothing;
                     step_offset = n_settle, print_timer = false)
    return built
end

# ---------------------------------------------------------------------------
# 1. Short-run trajectory equivalence: onesided=true vs onesided=false,
#    through the full settling -> moving two-stage sequence (not just one
#    stage in isolation), since the Ghost×Virtual pairing (ghost_bottom) and
#    the two-integrator handoff are exactly what's unique here.
# ---------------------------------------------------------------------------

@testset "trapdoor: short-run trajectory equivalence onesided=true vs onesided=false (settling -> moving)" begin
    n_settle, n_move = 30, 30

    built_old = _trapdoor_like(false)
    built_new = _trapdoor_like(true)

    @test built_old.n_soil == built_new.n_soil == _TD_nx * _TD_ny

    _run_trapdoor_two_stage!(built_old, n_settle, n_move)
    _run_trapdoor_two_stage!(built_new, n_settle, n_move)

    soil_old, soil_new = built_old.soil, built_new.soil

    x_scale     = max(maximum(norm.(soil_old.x)), 1.0)
    v_scale     = max(maximum(norm.(soil_old.v)), 1.0)
    rho_scale   = max(maximum(abs.(soil_old.rho)), 1.0)
    stress_scale = max(maximum(norm.(soil_old.stress)), 1.0)
    strain_scale = max(maximum(norm.(soil_old.strain)), 1.0)
    vort_scale   = max(maximum(abs.(soil_old.vorticity)), 1.0)

    x_diff      = _elemdiff(soil_old.x, soil_new.x)
    v_diff      = _elemdiff(soil_old.v, soil_new.v)
    rho_diff    = _elemdiff(soil_old.rho, soil_new.rho)
    stress_diff = _elemdiff(soil_old.stress, soil_new.stress)
    strain_diff = _elemdiff(soil_old.strain, soil_new.strain)
    strainp_diff = _elemdiff(soil_old.strain_p, soil_new.strain_p)
    vort_diff   = _elemdiff(soil_old.vorticity, soil_new.vorticity)

    @test _allfinite(soil_old.x) && _allfinite(soil_new.x)
    @test _allfinite(soil_old.rho) && _allfinite(soil_new.rho)
    @test _allfinite(soil_old.stress) && _allfinite(soil_new.stress)

    # Measured (see this harness's development notes): all fields agree at
    # true floating-point-reordering level (relative diffs ~1e-13-1e-19), not
    # merely "close" — i.e. this soil grid's strain rates never straddle the
    # elastoplastic yield surface closely enough for the branch in
    # ElastoPlasticStressUpdater to diverge between the two sweep orders.
    # 1e-9 leaves ~1000x headroom over the observed diffs while still being
    # tight enough to catch a real regression.
    @test x_diff       < 1e-9 * x_scale
    @test v_diff        < 1e-9 * v_scale
    @test rho_diff       < 1e-9 * rho_scale
    @test stress_diff    < 1e-9 * stress_scale
    @test strain_diff    < 1e-9 * strain_scale
    @test strainp_diff   < 1e-9 * strain_scale
    @test vort_diff       < 1e-9 * vort_scale

    # Probe measurement (not exercised during the run itself since
    # save_interval_step was chosen to suppress it) — diff the moving-phase
    # probe's accumulated stress field between the two runs.
    _measure_trapdoor_probes!(soil_old, built_old.td_probe_moving, built_old.probe_moving_int, built_old.h_sph)
    _measure_trapdoor_probes!(soil_new, built_new.td_probe_moving, built_new.probe_moving_int, built_new.h_sph)

    probe_stress_old = built_old.td_probe_moving.stress
    probe_stress_new = built_new.td_probe_moving.stress
    @test _allfinite(probe_stress_old)
    @test _allfinite(probe_stress_new)
    # Sanity: the probes actually picked up non-vacuous stress (otherwise the
    # diff check below would pass trivially on all-zero data).
    @test any(v -> norm(v) > 0, probe_stress_old)

    probe_scale = max(maximum(norm.(probe_stress_old)), 1.0)
    @test _elemdiff(probe_stress_old, probe_stress_new) < 1e-9 * probe_scale
end

# ---------------------------------------------------------------------------
# 2. Long-run physical invariants (onesided=true path only): settling then an
#    extended moving-only run, checking for NaN/Inf and a sanity bound on
#    position magnitude (catches a particle leaking past a boundary/ghost
#    wall and blowing up).
# ---------------------------------------------------------------------------

@testset "trapdoor: long-run physical invariants (onesided=true)" begin
    n_settle, n_move_long = 30, 100

    built = _trapdoor_like(true)
    _run_trapdoor_two_stage!(built, n_settle, n_move_long)

    soil = built.soil

    @test _allfinite(soil.rho)
    @test _allfinite(soil.x)
    @test _allfinite(soil.v)
    @test _allfinite(soil.stress)
    @test _allfinite(soil.strain)
    @test _allfinite(soil.strain_p)
    @test _allfinite(soil.vorticity)

    @test _allfinite(built.bottom_virt.rho)
    @test _allfinite(built.bottom_virt.stress)
    @test _allfinite(built.trapdoor_moving_virt.stress)

    # No particle should have travelled absurdly far given the short run and
    # small domain (~0.8m x 0.4m soil block) — a broken boundary/ghost
    # coupling (particles leaking through a wall to -Inf) would blow this up.
    @test all(x -> all(abs.(x) .< 20.0), soil.x)

    _measure_trapdoor_probes!(soil, built.td_probe_moving, built.probe_moving_int, built.h_sph)
    probe_stress = built.td_probe_moving.stress
    @test _allfinite(probe_stress)
end
