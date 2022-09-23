#-------------------------------------------------------------------------------
# Compute cell residual for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual!(eq, grid, op, scheme, t, dt, u1, res, Fb, Ub)
   N = op["degree"]
   if N == 1
      compute_cell_residual_1!(eq, grid, op, scheme, t, dt, u1, res, Fb, Ub)
   elseif N == 2
      compute_cell_residual_2!(eq, grid, op, scheme, t, dt, u1, res, Fb, Ub)
   elseif N == 3
      compute_cell_residual_3!(eq, grid, op, scheme, t, dt, u1, res, Fb, Ub)
   else
      println("compute_cell_residual: Not implemented for degree > 1")
      @assert false
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Apply LWFR scheme and solve the problem
# N = degree of solution space
#-------------------------------------------------------------------------------
function solve_lwfr(eq, problem, scheme, param)
   tick()
   println("Solving ",eq.name," using LWFR")

   # Make 1D/2D grid
   grid = make_grid(problem, param)

   # Make fr operators
   N, sol_pts, cor_fun = scheme["degree"], scheme["solution_points"],
                         scheme["correction_function"]
   op = fr_operators(N, sol_pts, cor_fun)

   # Allocate memory
   u1, ua, res, Fb, Ub = setup_arrays_lwfr(grid, scheme)

   # Set initial condition
   set_initial_condition!(u1, grid, op, problem)

   # Compute cell average for initial condition
   compute_cell_average!(ua, u1, grid, problem, op)

   # Initialize counters
   iter, t, fcount = 0, 0.0, 0

   # Save initial solution to file
   fcount = write_soln("sol", fcount, iter, t, grid, op, ua, u1)

   # Choose CFL number
   cfl = param["cfl"]
   if cfl > 0.0
      @printf("CFL: specified value = %f\n", cfl)
   else
      cfl = get_cfl(scheme)
      @printf("CFL: based on stability = %f\n", cfl)
   end

   # Compute initial error norm
   error_norm = compute_error(problem, grid, op, u1, t)

   println("Starting time stepping")
   final_time = problem["final_time"]
   while t < final_time
      dt = compute_time_step(eq, grid, cfl, ua)
      dt = adjust_time_step(problem, param, t, dt)
      compute_cell_residual!(eq, grid, op, scheme, t, dt, u1, res, Fb, Ub)
      update_ghost_values_lwfr!(problem, eq, grid, op, Fb, Ub, t, dt)
      compute_face_residual!(eq, grid, op, scheme, t, dt, Fb, Ub, ua, res)
      update_solution!(u1, res)
      compute_cell_average!(ua, u1, grid, problem, op)
      apply_limiter!(grid, scheme, param, op, ua, u1)
      t += dt; iter += 1
      @printf("iter,dt,t = %5d %12.4e %12.4e\n", iter, dt, t)
      if save_solution(problem, param, t, iter)
         fcount = write_soln("sol", fcount, iter, t, grid, op, ua, u1)
      end
      if (param["compute_error_interval"] > 0 &&
          mod(iter,param["compute_error_interval"]) == 0)
         error_norm = compute_error(problem, grid, op, u1, t)
      end
   end
   error_norm = compute_error(problem, grid, op, u1, t)
   println("Elapsed time in seconds = ", tok())
   return error_norm, u1
end
