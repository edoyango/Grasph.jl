using Test
using Grasph

@testset "Grasph" begin
    include("test_particles.jl")
    include("test_read_h5.jl")
    include("test_kernels.jl")
    include("test_interaction_init.jl")
    include("test_interaction_grid.jl")
    include("test_interaction_sweep.jl")
    include("test_interaction_sweep_3d.jl")
    include("test_time_integration.jl")
    include("test_ghost_particles.jl")
    include("test_ghost_updates.jl")
    include("test_ghost_copier.jl")
    include("test_boundary_particles.jl")
    include("test_elasto_plastic.jl")
    include("test_ep_updater.jl")
    include("test_vorticity.jl")
    include("test_probe_particles.jl")
end
