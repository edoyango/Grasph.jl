using Test
using Grasph
using StaticArrays
using LinearAlgebra: norm

# ---------------------------------------------------------------------------
# Cluster G: onesided=true vs coloured, real-integrator equivalence test for
# CantileverBeam.jl's INTENDED interaction shape.
#
# CantileverBeam.jl, as currently committed to this repo, is BROKEN and does
# not run:
#   - `CubicSplineKernel(; ndims=2)` throws -- `h` is a required positional
#     argument, not a keyword.
#   - `SystemInteraction(...; h=h_sph)` passes an invalid `h=` keyword;
#     `SystemInteraction`'s only keywords are `velocity_adjust_pairwise_fn`,
#     `onesided`, and `ka`.
# This harness does NOT copy the script's calls verbatim. It reconstructs the
# INTENDED shape with corrected calls (`kernel = CubicSplineKernel(h_sph;
# ndims=2)`, no `h=` kwarg anywhere), while keeping every other physical
# choice (particle system types, state updaters, pfns, gravity/E/nu) as the
# script intends.
#
# This is the only shape (of the 11 experiment scripts) combining:
#   - an `ElastoPlasticParticleSystem` self-interaction (2-stage:
#     `StrainRateVorticityPfn` then `CauchyFluidPfn`, driven by
#     `ZeroFieldUpdater`/`HookeLawStressUpdater`) coupled to a
#     `DynamicBoundarySystem`-wrapped fixed end,
#   - a `ProbeParticleSystem` built via the *mirror-target* constructor
#     (`ProbeParticleSystem(name, real_system; extras=...)`, not an explicit
#     position list), coupled through `NeighborCountFn` -- a `WritesB()` pfn
#     that writes into `system_b` (the probe), not `system_a` (the beam).
# Running this through a real `LeapFrogTimeIntegrator`/`time_integrate!` loop
# (rather than the standalone per-pfn harnesses in test_onesided_sweep.jl)
# exercises the full stage loop, the DynamicBoundarySystem-specific
# `pfn_contribution` methods, and the probe reverse-sweep path together, in
# the same wiring the real script uses.
# ---------------------------------------------------------------------------

const CFL_num = 0.1

# ---------------------------------------------------------------------------
# Shared builder -- deterministic regular grid (no RNG needed: unlike the
# dambreak-like template, every particle position here is a pure function of
# its grid index, so calling this twice independently already yields
# bit-identical starting state).
#
# Reduced scale: beam 20x6 = 120 particles (script: 250x20 = 5000); fixed
# boundary 1 layer x 8 rows = 8 particles (script: 3 layers x 26 rows = 78).
# Same dx/h_sph ratio, same physical constants (E, nu, rho0, gravity,
# artificial-viscosity coefficients, boundary beta) as the script.
# ---------------------------------------------------------------------------

function _cantilever_like(; n_beam_x=20, n_beam_y=6, n_bnd_layers=1)
    dx             = 0.02
    h_sph          = 1.2 * dx
    rho0           = 1000.0
    E              = 1.0e9
    nu             = 0.3
    c_beam         = sqrt(E * (1 - nu) / (rho0 * (1 + nu) * (1 - 2*nu)))
    art_visc_alpha = 0.1
    art_visc_beta  = 0.0
    gravity        = SVector(0.0, -9.81)

    n_beam    = n_beam_x * n_beam_y
    beam_mass = rho0 * dx^2

    beam = ElastoPlasticParticleSystem(
        "beam", n_beam, 2, 4, beam_mass, c_beam;
        source_v      = gravity,
        state_updater = (
            ZeroFieldUpdater(:strain_rate, :vorticity),
            HookeLawStressUpdater(E, nu),
        ),
    )
    let k = 1
        for j in 0:n_beam_y-1, i in 0:n_beam_x-1
            beam.x[k] = SVector((i + 0.5) * dx, (j + 0.5) * dx)
            k += 1
        end
    end
    fill!(beam.v,      zero(SVector{2,Float64}))
    fill!(beam.stress, zero(SVector{4,Float64}))
    beam.rho .= rho0
    beam.p   .= 0.0

    # Fixed-end boundary at x = 0 -- 1-row margin above/below the beam,
    # mirroring the script's `n_bnd_layers * (n_beam_y + 2*n_bnd_layers)`.
    n_fix = n_bnd_layers * (n_beam_y + 2 * n_bnd_layers)
    fix_inner = BasicParticleSystem("fix", n_fix, 2, beam_mass, c_beam)
    let k = 1
        for layer in 1:n_bnd_layers, iy in -n_bnd_layers:n_beam_y+n_bnd_layers-1
            fix_inner.x[k] = SVector(-(layer - 0.5) * dx, (iy + 0.5) * dx)
            k += 1
        end
    end
    fix_inner.rho .= rho0
    fill!(fix_inner.v, zero(SVector{2,Float64}))
    # normal (1,0) points right (into the beam domain); point on the plane x = 0.
    fix_dyn = DynamicBoundarySystem(fix_inner, SVector(1.0, 0.0), SVector(0.0, 0.0), 3.0)

    kernel     = CubicSplineKernel(h_sph; ndims=2)
    sr_pfn     = StrainRateVorticityPfn()
    cauchy_pfn = CauchyFluidPfn(art_visc_alpha, art_visc_beta, h_sph)

    beam_probe = ProbeParticleSystem(
        "beam_probe", beam;                       # mirror-target constructor form
        extras = (nbr_count = zeros(Int, n_beam),),
    )

    return (; kernel, sr_pfn, cauchy_pfn, beam, fix_inner, fix_dyn, beam_probe, n_beam, n_fix)
end

function _build_integrator(s; onesided::Bool)
    beam_self = SystemInteraction(s.kernel, (s.sr_pfn, s.cauchy_pfn), s.beam; onesided=onesided)
    beam_fix  = SystemInteraction(s.kernel, (s.sr_pfn, s.cauchy_pfn), s.beam, s.fix_dyn; onesided=onesided)
    probe_nbr = SystemInteraction(s.kernel, NeighborCountFn(:nbr_count), s.beam, s.beam_probe; onesided=onesided)

    LeapFrogTimeIntegrator(
        [s.beam, s.fix_inner], [beam_self, beam_fix];
        probes             = (s.beam_probe,),
        probe_interactions = (probe_nbr,),
    )
end

# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

_elemscale(v::AbstractVector{<:Real})    = max(maximum(abs, v), 1.0)
_elemscale(v::AbstractVector{<:SVector}) = max(maximum(norm, v), 1.0)
_elemdiff(a::AbstractVector{<:Real}, b::AbstractVector{<:Real})    = maximum(abs.(a .- b))
_elemdiff(a::AbstractVector{<:SVector}, b::AbstractVector{<:SVector}) = maximum(norm.(a .- b))

_has_nan(v::AbstractVector{<:Real})    = any(isnan, v)
_has_nan(v::AbstractVector{<:SVector}) = any(x -> any(isnan, x), v)

# ---------------------------------------------------------------------------
# 1. Short-run trajectory equivalence -- onesided=true vs onesided=false,
#    through a real LeapFrogTimeIntegrator (2-stage: StrainRateVorticityPfn,
#    CauchyFluidPfn), beam self + beam<->DynamicBoundarySystem fixed end.
# ---------------------------------------------------------------------------

@testset "short-run trajectory equivalence: onesided=true vs onesided=false" begin
    s_old = _cantilever_like()
    s_new = _cantilever_like()

    integ_old = _build_integrator(s_old; onesided=false)
    integ_new = _build_integrator(s_new; onesided=true)

    nsteps = 80
    outdir = mktempdir()

    # save_interval_step == nsteps triggers exactly one probe measurement
    # (via `_measure_probes!`, inside `_maybe_save!`) at the final step.
    # Probes are only ever measured at save cadence with a non-`nothing`
    # output_prefix -- this is the only way to exercise NeighborCountFn's
    # WritesB() reverse-sweep pass through the real driver loop.
    time_integrate!(integ_old, nsteps, nsteps + 1, nsteps, CFL_num,
                     joinpath(outdir, "coloured"); print_timer=false)
    time_integrate!(integ_new, nsteps, nsteps + 1, nsteps, CFL_num,
                     joinpath(outdir, "onesided"); print_timer=false)

    beam_old, beam_new = s_old.beam, s_new.beam

    # Positions/velocities/density tightest (matches test_onesided_sweep.jl's
    # dambreak-like short-run tolerances); dvdt/drhodt/stress/strain_rate/
    # vorticity are forces/rates recomputed fresh each sweep from the
    # (very slightly) diverged x/v/rho, so a somewhat looser tolerance is
    # appropriate for them too.
    tolerances = (
        x           = 1e-8,
        v           = 1e-6,
        rho         = 1e-6,
        dvdt        = 1e-6,
        drhodt      = 1e-6,
        stress      = 1e-6,
        strain_rate = 1e-6,
        vorticity   = 1e-6,
    )

    for (field, tol) in pairs(tolerances)
        va, vb = getproperty(beam_old, field), getproperty(beam_new, field)
        @test !_has_nan(va)
        @test !_has_nan(vb)
        @test _elemdiff(va, vb) < tol * _elemscale(va)
    end

    # NeighborCountFn is WritesB() -- it writes into the PROBE (system_b),
    # never the beam (system_a). A broken WritesB reverse-sweep pass would be
    # completely invisible to the beam-field checks above; this is the
    # dedicated check for it. The regular beam grid (dx spacing) has no pairs
    # sitting near the kernel cutoff boundary (cutoff = 2h = 2.4dx; no
    # integer grid offset (a,b) satisfies a^2+b^2 = 5.76), and the beam barely
    # moves over this many steps (dt ~ CFL*h/c_beam is tiny for this stiff
    # material), so the neighbour count is expected to be exactly reproduced,
    # not merely close.
    @test s_old.beam_probe.nbr_count == s_new.beam_probe.nbr_count
    @test sum(s_old.beam_probe.nbr_count) > 0   # sanity: the sweep found neighbours at all
end

# ---------------------------------------------------------------------------
# 2. Long-run physical invariants (onesided=true path only)
# ---------------------------------------------------------------------------

@testset "long-run physical invariants (onesided=true)" begin
    s = _cantilever_like()
    integ = _build_integrator(s; onesided=true)

    nsteps = 200   # 2.5x the short-run step count
    time_integrate!(integ, nsteps, nsteps + 1, nsteps + 1, CFL_num, nothing; print_timer=false)

    beam = s.beam
    for field in (:x, :v, :dvdt, :stress, :strain_rate, :vorticity)
        v = getproperty(beam, field)
        @test !_has_nan(v)
    end
    @test all(!isnan, beam.rho)
    @test all(!isinf, beam.rho)
    @test all(!isnan, beam.drhodt)
    @test all(x -> all(!isinf, x), beam.x)
    @test all(x -> all(!isinf, x), beam.v)

    # Loose sanity bound: the beam spans ~0.4m x 0.12m at this reduced scale.
    # A broken fixed-end coupling (e.g. a particle leaking past the boundary
    # and blowing up under the unopposed gravity load) would send positions
    # to absurd magnitudes well before this bound is reached.
    @test all(x -> all(abs.(x) .< 50.0), beam.x)
end
