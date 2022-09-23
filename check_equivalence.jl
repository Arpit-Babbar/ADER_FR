using TrixiBase: trixi_include
using Suppressor
final_time = 5.0
cfls = [0.226, 0.117, 0.072, 0.049]
filename_periodic = "src/run_const_linadv1d.jl"
for degree in 1:3
    ndofs = 240
    nx = Int(ndofs / (degree + 1))

    println("Running ADER with degree N=$degree")
    error, u1 = @suppress trixi_include(filename_periodic, solver = "ader",
                                        degree = degree, final_time = final_time, nx = nx,
                                        cfl = cfls[degree])
    u1_ader = copy(u1)
    cp("error.txt","error_ader_$degree.txt", force = true)
    rm("output_ader_$degree", force=true, recursive = true)
    cp("output","output_ader_$degree", force=true)

    println("Running LW-D2 with degree N=$degree")
    error, u1_ = @suppress trixi_include(filename_periodic, solver = "lwfr",
                                         degree = degree, final_time = final_time, nx = nx,
                                         cfl = cfls[degree])
    u1_lwfr = copy(u1_)
    cp("error.txt","error_lwfr_$degree.txt", force = true)
    rm("output_lwfr_$degree", force=true, recursive = true)
    cp("output","output_lwfr_$degree", force=true)
    @show norm(u1_ader - u1_lwfr, Inf)
    @assert norm(u1_ader - u1_lwfr, Inf) < 1e-13 norm(u1_ader - u1_lwfr, Inf)

    println("Running LW-D1 with degree N=$degree")
    @suppress trixi_include(filename_periodic, solver = "lwfr", diss = 1,
                            degree = degree, final_time = final_time, nx = nx, cfl = cfls[degree])
    cp("error.txt","error_lwfr_d1_$degree.txt", force = true)
    rm("output_lwfr_d1_$degree", force=true, recursive = true)
    cp("output","output_lwfr_d1_$degree", force=true)
end

final_time = 5.0
cfls = [0.226, 0.117, 0.072, 0.049]
filename_dirichlet = "src/run_const_linadv1d_dirichlet.jl"
for degree in 1:3
    ndofs = 240
    nx = Int(ndofs / (degree + 1))

    println("Running ADER with Dirichlet bc degree N=$degree")
    error, u1 = @suppress trixi_include(filename_dirichlet, solver = "ader", degree = degree,
                                        final_time = final_time, nx = nx, cfl = cfls[degree])
    u1_ader = copy(u1)
    cp("error.txt","error_dirichlet_ader_$degree.txt", force = true)
    rm("output_ader_$degree", force=true, recursive = true)
    cp("output","output_ader_$degree", force=true)

    println("Running LW-D2 with Dirichlet bc, degree N=$degree")
    error, u1_ = @suppress trixi_include(filename_dirichlet, solver = "lwfr", degree = degree,
                                         final_time = final_time, nx = nx, cfl = cfls[degree])
    u1_lwfr = copy(u1_)
    cp("error.txt","error_dirichlet_lwfr_$degree.txt", force = true)
    rm("output_lwfr_$degree", force=true, recursive = true)
    cp("output","output_lwfr_$degree", force=true)
    @show norm(u1_ader - u1_lwfr, Inf)
    @assert norm(u1_ader - u1_lwfr, Inf) < 1e-12

    println("Running LW-D1 with Dirichlet bc, degree N=$degree")
    @suppress trixi_include(filename_dirichlet, solver = "lwfr", diss = 1, degree = degree,
                            final_time = final_time, nx = nx, cfl = cfls[degree])
    cp("error.txt","error_dirichlet_lwfr_d1_$degree.txt", force = true)
    rm("output_lwfr_d1_$degree", force=true, recursive = true)
    cp("output","output_lwfr_d1_$degree", force=true)
end
