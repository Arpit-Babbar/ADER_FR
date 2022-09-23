#------------------------------------------------------------------------------
function compute_residual_rkfr!(eq, problem, grid, op, scheme, t, dt, u1,
                                Fb, ub, ua, res)
   compute_cell_residual_rkfr!(eq, grid, op, scheme, t, dt, u1, res, Fb, ub)
   update_ghost_values_rkfr!(problem, eq, grid, op, Fb, ub, t)
   compute_face_residual!(eq, grid, op, scheme, t, dt, Fb, ub, ua, res)
   return nothing
end

# DiffEq callback function for time step. We also save solution in this.
# Makes use of two global variables: iter, fcount
function dtFE(u, p, t)
   eq, problem, scheme, param, grid, op, Fb, ub, ua, res = p

   dt = compute_time_step(eq, grid, param["cfl"], ua)
   dt = adjust_time_step(problem, param, t, dt)
   if save_solution(problem, param, t, iter)
      global fcount = write_soln("sol", fcount, iter, t, grid, op, ua, u)
   end
   @printf("iter,dt,t   = %5d %12.4e %12.4e \n", iter, dt, t)
   if (param["compute_error_interval"] > 0 &&
       mod(iter,param["compute_error_interval"]) == 0)
      error_norm = compute_error(problem, grid, op, u, t)
   end
   global iter +=1
   return dt
end

# DiffEq limiter functions
function stage_limiter!(u, integrator, p, t)
   eq, problem, scheme, param, grid, op, Fb, ub, ua, res = p
   compute_cell_average!(ua, u, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u)
end
step_limiter!(u, integrator, p, t) = nothing

#------------------------------------------------------------------------------
# 2nd order, 2-stage SSPRK
#------------------------------------------------------------------------------
function apply_ssprk22!(eq, problem, param, grid, op, scheme,
                        t, dt, u0, u1, Fb, ub, ua, res)
   r1 = view_res(grid, res)
   # Stage 1
   ts = t
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpy!(-1.0, r1, u1)         # u1 = u1 - res
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   # Stage 2
   ts = t + dt
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpy!(-1.0, r1, u1)         # u1 = u1 - res
   axpby!(0.5, u0, 0.5, u1)    # u1 = u0 + u1
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   return nothing
end

#------------------------------------------------------------------------------
# 3rd order, 3-stage SSPRK
#------------------------------------------------------------------------------
function apply_ssprk33!(eq, problem, param, grid, op, scheme,
                        t, dt, u0, u1, Fb, ub, ua, res)
   r1 = view_res(grid, res)
   # Stage 1
   ts = t
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpy!(-1.0, r1, u1)                     # u1 = u1 - res
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   # Stage 2
   ts = t + dt
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpy!(-1.0, r1, u1)                     # u1 = u1 - res
   axpby!(0.75, u0, 0.25, u1)              # u1 = (3/4)u0 + (1/4)u1
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   # Stage 3
   ts = t + 0.5*dt
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpy!(-1.0, r1, u1)                     # u1 = u1 - res
   axpby!(1.0/3.0, u0, 2.0/3.0, u1)        # u1 = (1/3)u0 + (2/3)u1
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   return nothing
end

#------------------------------------------------------------------------------
# z = a*x + y
#------------------------------------------------------------------------------
function axpyz!(a, x::SubArray{Real,4}, y, z)
   n1, n2, nx, ny = size(x)
   @inbounds Threads.@threads for ij in CartesianIndices((1:nx, 1:ny))
      i, j = ij[1], ij[2]
      for jj=1:n2, ii=1:n1
         z[ii,jj,i,j] = a * x[ii,jj,i,j] + y[ii,jj,i,j]
      end
   end
   return nothing
end

#--------------------------------------------------------
# Four stage, third order SSPRK
#------------------------------------------------------
function apply_ssprk43!(eq, problem, param, grid, op, scheme,
                        t, dt, u0, u1,Fb, ub, ua, res)
   utmp   = copy(u0)
   r1 = view_res(grid, res)
   # Stage 1
   ts = t
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpy!(-0.5, r1, u1)                     # u1 = u1 - res
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   # Stage 2
   ts = t + 0.5*dt
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpy!(-0.5, r1, u1)                     # u1 = u1 - res
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   # Stage 3
   ts = t + dt
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpy!(-0.5, r1, u1)                     # u1 = u1 - res
   axpby!(2/3, u0, 1/3, u1)                # u1 = 2/3*u0 + 1/3*u1
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   # Stage 4
   ts = t + 0.5*dt
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpy!(-0.5, r1, u1)                    # u1 = u1 - res
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
return nothing
end

#------------------------------------------------------------------------------
# Classical RK4
#------------------------------------------------------------------------------
function apply_rk4!(eq, problem, param, grid, op, scheme, t,
                    dt, u0, u1,Fb, ub, ua, res)
   r1 = view_res(grid, res)
   utmp   = copy(u0)
   # Stage 1
   ts   = t
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpyz!(-0.5, r1, u0, u1)       # u1   = u0 - 0.5*r1
   axpy!(-1.0/6.0, r1, utmp)      # utmp = utmp - (1/6)*r1
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   # Stage 2
   ts  = t + 0.5*dt
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpyz!(-0.5, r1, u0, u1)       # u1   = u0 - 0.5*r1
   axpy!(-1.0/3.0, r1, utmp)      # utmp = utmp - (1/3)*r1
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   # Stage 3
   ts  = t + 0.5*dt
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpyz!(-1.0, r1, u0, u1)       # u1   = u0 - r1
   axpy!(-1.0/3.0, r1, utmp)      # utmp = utmp - (1/3)*r1
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   # Stage 4
   ts  = t + dt
   compute_residual_rkfr!(eq, problem, grid, op, scheme, ts, dt, u1,
                          Fb, ub, ua, res)
   axpyz!(-1.0/6.0, r1, utmp, u1) # u1   = utmp - (1/6)*r1
   compute_cell_average!(ua, u1, grid, problem, op)
   apply_limiter!(grid, scheme, param, op, ua, u1)
   return nothing
end

#------------------------------------------------------------------------------
function solve_rkfr(eq, problem, scheme, param)
   tick()
   println("Solving ",eq.name," using RKFR")

   # Make grid
   grid = make_grid(problem, param)

   # Make fr operators
   N, sol_pts, cor_fun = scheme["degree"], scheme["solution_points"],
                         scheme["correction_function"]
   op = fr_operators(N, sol_pts, cor_fun)

   u0, u1, ua, res, Fb, ub = setup_arrays_rkfr(grid, scheme)

   # Set initial condition
   set_initial_condition!(u1, grid, op, problem)

   # Compute cell average for initial condition
   compute_cell_average!(ua, u1, grid, problem, op)

   # Initialize counters
   t = 0.0
   global iter, fcount = 0, 0
   final_time = problem["final_time"]

   # Choose CFL number
   cfl = param["cfl"]
   if cfl > 0.0
      @printf("CFL: specified value = %f\n", cfl)
   else
      cfl = get_cfl(scheme)
      @printf("CFL: based on stability = %f\n", cfl)
   end
   param["cfl"] = cfl

   # Fifth order: use DifferentialEquations
   if N == 4
      println("Using DifferentialEquations")
      p = (eq, problem, scheme, param, grid, op, Fb, ub, ua, res)
      copyto!(u0, u1)
      tspan = (0.0, final_time)
      odeprob = ODEProblem(compute_residual_rkfr!, u0, tspan, p)
      dt = compute_time_step(eq, grid, cfl, ua)
      callback_dt = StepsizeLimiter(dtFE, safety_factor=1.0, max_step=true)
      callback = (callback_dt)
      sol = DifferentialEquations.solve(odeprob,
                                        SSPRK54(stage_limiter!, step_limiter!),
                                        dt=dt, adaptive=false,
                                        callback=callback,
                                        saveat=final_time, dense=false,
                                        save_start=false, save_everystep=false)
      copyto!(u1, sol[1])
      t = sol.t[1]
      error_norm = compute_error(problem, grid, op, u1, t)
      tock()
      return error_norm, u1
   end

   # Save initial solution to file
   fcount = write_soln("sol", fcount, iter, t, grid, op, ua, u1)

   # Compute initial error norm
   error_norm = compute_error(problem, grid, op, u1, t)

   println("Starting time stepping")
   while t < final_time
      dt = compute_time_step(eq, grid, cfl, ua)
      dt = adjust_time_step(problem, param, t, dt)
      copyto!(u0, u1) # u0 = u1
      if N == 1
         apply_ssprk22!(eq, problem, param, grid, op, scheme, t, dt, u0, u1,
                        Fb, ub, ua, res)
      elseif N == 2
         apply_ssprk33!(eq, problem, param, grid, op, scheme, t, dt, u0, u1,
                        Fb, ub, ua, res)
      elseif N == 3
         apply_rk4!(eq, problem, param, grid, op, scheme, t, dt, u0, u1,
                    Fb, ub, ua, res)
      else
         println("Not implemented for degree ", N)
         @assert false
      end
      compute_cell_average!(ua, u1, grid, problem, op)
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
