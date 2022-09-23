push!(LOAD_PATH,".")

ngrids = 3      # Number of grid levels
Error_l1, Error_l2 = zeros(ngrids), zeros(ngrids)
#------------------------------------------------------------------------------
out = include("run_burg1d.jl")
Error_l1[1], Error_l2[1] = out["l1_error"], out["l2_error"]
#------------------------------------------------------------------------------
for k=2:ngrids
   global grid_size = 2*grid_size
   local param = FR.Parameters(grid_size, cfl, tvbM, save_iter_interval,
                               save_time_interval, compute_error_interval)
   local out = FR.solve(equation, problem, scheme, param)
   Error_l1[k], Error_l2[k] = out["l1_error"], out["l2_error"]

   println("l1 convergence rate = ", log(Error_l1[k-1]/Error_l1[k])/log(2))
   println("l2 convergence rate = ", log(Error_l2[k-1]/Error_l2[k])/log(2))
end
