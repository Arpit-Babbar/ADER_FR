import NLsolve: nlsolve

#-------------------------------------------------------------------------------
# Apply LWFR scheme and solve the problem
# N = degree of solution space
#-------------------------------------------------------------------------------
function solve_ader(eq, problem, scheme, param)
   tick()
   println("Solving ",eq.name," using ADER")

   # Make 1D/2D grid
   grid = make_grid(problem, param)

   # Make fr operators(if you can make ADER-FR instead of ADER-DG)
   N, sol_pts, cor_fun = scheme["degree"], scheme["solution_points"],
                         scheme["correction_function"]
   op = fr_operators(N, sol_pts, cor_fun)

   @assert N <= 3 "Only implemented for degree <= 3"

   # Allocate memory
   q, u1, qa, ua, res, Fb, Ub = setup_arrays_ader(grid, scheme)

   # Set initial condition
   set_initial_condition!(u1, grid, op, problem)
   # for τ=1:N+1
   #    q[:,τ,1:end-1]=u1
   # end

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
      # Add compute_lwfr residual here to get the initial guess for iterative solver
      for τ=1:N+1
         q[:,τ,1:end-1]=u1
      end
      compute_predictor!(eq, grid, op, dt, u1, q)
      compute_cell_average!(ua, q[:,1,1:end-1], grid, problem, op)
      # fcount = write_soln("sol", fcount, iter, t, grid, op, ua, q[:,1,1:end-1])
      compute_cell_residual_ader!(eq, grid, op, scheme, dt, q, res, Fb, Ub)
      compute_cell_average_ader!(qa, q, grid, problem, op)
      compute_cell_average!(ua, u1, grid, problem, op)
      update_ghost_values_ader!(problem, eq, grid, op, Fb, Ub, t, dt)
      compute_face_residual_ader!(eq, grid, op, scheme, dt, Fb, Ub, qa, res)
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
