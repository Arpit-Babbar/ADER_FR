module FR1D

using Plots
using DelimitedFiles
include("FR.jl")
include("ADER1D.jl")
include("LWFR1D.jl")
include("RKFR1D.jl")
#-------------------------------------------------------------------------------
# Set initial condition by interpolation in all real cells
#-------------------------------------------------------------------------------
function set_initial_condition!(u, grid, op, problem)
   println("Setting initial condition")
   initial_value = problem["initial_value"]
   nx = grid.size
   xg = op["xg"]
   nd = length(xg)
   for i=1:nx
      dx = grid.dx[i] # cell size
      xc = grid.xc[i] # cell center
      for ii=1:nd
         x = xc - 0.5 * dx + xg[ii] * dx
         u[ii,i] = initial_value(x)
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Compute cell average in all real cells
#-------------------------------------------------------------------------------
function compute_cell_average!(ua, u, grid, problem, op)
   nx = grid.size
   wg = op["wg"]
   Vl, Vr = op["Vl"], op["Vr"]
   nd = length(wg)
   for i=1:nx
      u1 = @view u[:,i]
      ua[i] = dot(wg, u1)
   end
   # Update ghost values of ua by periodicity or with face averages
   if problem["periodic_x"]
      ua[0]    = ua[nx]
      ua[nx+1] = ua[1]
   else
      u1 = @view u[:,1]
      ua[0] = dot(u1,Vl)
      u1 = @view u[:,nx]
      ua[nx] = dot(u1,Vr)
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Choose cfl based on degree and correction function
#-------------------------------------------------------------------------------
function get_cfl(scheme)
   solver, diss, degree, cor_fun = ( scheme["solver"], scheme["diss"],
                                     scheme["degree"],
                                     scheme["correction_function"] )
   @assert ((degree > 0 && degree < 5) || scheme["solver"] == "ader") "Invalid degree"
   if solver == "lwfr"
      if diss == 1
         cfl_radau = [0.226, 0.117, 0.072, 0.049]
         cfl_g2    = [0.465, 0.204, 0.116, 0.060]
      elseif diss == 2
         cfl_radau = [0.333, 0.170, 0.103, 0.069]
         cfl_g2    = [1.000, 0.333, 0.170, 0.103]
      end
   elseif solver == "rkfr"
      cfl_radau = [0.333, 0.209, 0.145, 0.110]
      cfl_g2    = [1.0, 0.45, 0.2875, 0.212]
   elseif solver == "ader"
      cfl_radau = [0.333, 0.170, 0.103, 0.069, 0.01, 0.001, 0.001]
      cfl_g2    = [1.000, 0.333, 0.170, 0.103, 0.01, 0.001, 0.001]
   end
   # Reduce this cfl by a small amount
   safety_factor = 0.98
   if cor_fun == "radau"
      return safety_factor * cfl_radau[degree]
   elseif cor_fun == "g2"
      if solver == "rkfr"
         println("Warning - Using CFL of LWFR for RKFR")
      end
      return safety_factor * cfl_g2[degree]
   else
      println("get_cfl: unknown correction function")
      @assert false
   end
end

#-------------------------------------------------------------------------------
# Compute dt using cell average
#-------------------------------------------------------------------------------
function compute_time_step(eq, grid, cfl, ua)
   speed = eq.speed
   nx    = grid.size
   xc    = grid.xc
   dx    = grid.dx
   den   = 0.0
   for i=1:nx
      sx = speed(xc[i], ua[i], eq)
      den    = max(den, abs(sx)/dx[i] + 1.0e-12)
   end
   dt = cfl / den
   return dt
end

#-------------------------------------------------------------------------------
# Compute flux f at all solution points in one cell
#-------------------------------------------------------------------------------
@inline function compute_flux!(eq, flux, x, u, f)
   n = length(x)
   for i=1:n
      f[i] = flux(x[i], u[i], eq)
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Interpolate average flux and solution to the two faces of cell
#-------------------------------------------------------------------------------
@inline function interpolate_to_face!(Vl, Vr, F, U, Fb, Ub)
   nd = length(F)
   # Interpolate flux
   Fb[1] = BLAS.dot(nd, F, 1, Vl, 1)
   Fb[2] = BLAS.dot(nd, F, 1, Vr, 1)

   # Interpolate U
   Ub[1] = dot(u, Vl)
   Ub[2] = dot(u, Vr)
   return nothing
end

@inline function interpolate_to_face!(Vl, Vr, U, Ub)
   nd = length(U)

   # Interpolate U
   Ub[1] = dot(U, Vl)
   Ub[2] = dot(U, Vr)
   return nothing
end

#-------------------------------------------------------------------------------
# Add numerical flux to residual
#-------------------------------------------------------------------------------
function compute_face_residual!(eq, grid, op, scheme, t, dt, Fb, Ub, ua, res)
   bl, br = op["bl"], op["br"]
   nx = grid.size
   dx = grid.dx
   xf = grid.xf
   num_flux = scheme["numerical_flux"]
   RealT = eltype(dx)

   lamx = OffsetArray(zeros(RealT, nx+2), OffsetArrays.Origin(0))
   lamx[1:nx] = dt ./ dx

   # Vertical faces, x flux
   for i=1:nx+1
      # Face between i-1 and i
      x = xf[i]
      Fn = num_flux(x, ua[i-1], ua[i],
                     Fb[2,i-1], Fb[1,i],
                     Ub[2,i-1], Ub[1,i], eq, 1)
      r = @view res[:, i-1]
      axpy!(lamx[i-1]*Fn, br, r)
      r = @view res[:, i]
      axpy!(lamx[i]*Fn, bl, r)
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Fill some data in ghost cells using periodicity
#-------------------------------------------------------------------------------
function update_ghost_values_periodic!(problem, Fb, Ub)
   nx =  size(Fb,2)-2
   if problem["periodic_x"]
      # Left ghost cells
      copyto!(Ub, CartesianIndices((2:2, 0:0)),
              Ub, CartesianIndices((2:2, nx:nx)))
      copyto!(Fb, CartesianIndices((2:2, 0:0)),
              Fb, CartesianIndices((2:2, nx:nx)))

      # Right ghost cells
      copyto!(Ub, CartesianIndices((1:1, nx+1:nx+1)),
              Ub, CartesianIndices((1:1, 1:1)))
      copyto!(Fb, CartesianIndices((1:1, nx+1:nx+1)),
              Fb, CartesianIndices((1:1, 1:1)))
   end

   # TODO: add neumann and dirichlet bc
   left, right = problem["boundary_condition"]

   return nothing
end

#-------------------------------------------------------------------------------
# Limiter function
#-------------------------------------------------------------------------------
function apply_limiter_tvb!(grid, param, op, ua, u1)
   nx = grid.size
   xg, wg = op["xg"], op["wg"]
   Vl, Vr = op["Vl"], op["Vr"]
   tvbM = param["tvbM"]
   nd = length(wg)
   # Loop over cells
   for i=1:nx
      # face values
      ul, ur  = 0.0, 0.0
      for ii=1:nd
         ul += u1[ii, i] * Vl[ii]
         ur += u1[ii, i] * Vr[ii]
      end
      # slopes b/w centres and faces
      dul, dur = ua[i] - ul, ur - ua[i]
      # minmod to detect jumps'
      Mdx2 = tvbM * grid.dx[i]^2
      dulm = minmod(dul, ua[i] - ua[i-1], ua[i+1] - ua[i], Mdx2)
      durm = minmod(dur, ua[i] - ua[i-1], ua[i+1] - ua[i], Mdx2)
      # limit if jumps are detected
      if (abs(dul-dulm)>1e-06 || abs(dur-durm)>1e-06)
         dux = 0.5 * (dulm+durm)
         for ii=1:nd
            u1[ii,i] = ua[i] + 2.0 * (xg[ii]-0.5) * dux
         end
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Compute error norm
#-------------------------------------------------------------------------------
function compute_error(problem, grid, op, u1, t)
   xmin, xmax = grid.domain
   xg         = op["xg"]
   wg = op["wg"]
   nd         = length(xg)
   utype = eltype(u1)

   # TODO: Assuming periodicity
   exact_solution = problem["exact_solution"]

   nq     = nd + 3    # number of quadrature points in each direction
   xq, wq = weights_and_points(nd+3, "gl")
   # xq, wq = xg, wg

   nq = length(xq)

   V = Vandermonde_lag(xg, xq) # matrix evaluating at `xq`
                               # using values at solution points `xg`
   nx = grid.size
   xc = grid.xc
   dx = grid.dx

   l1_error, l2_error, energy  = 0.0, 0.0, 0.0
   for i=1:nx
         ue = zeros(utype, length(xq)) # exact solution
         x = xc[i] - 0.5 * dx[i] .+ dx[i] * xq
         for i=1:nq
            ue[i] = exact_solution(x[i], t)
         end
         @views un  = V * u1[:,i]
         du         = abs.(un - ue)
         # l1         = dx[i] * BLAS.dot(nq, du, 1, wq, 1)
         l1         = dx[i] * dot(du, wq)
         @. du      = du^2
         # l2         = dx[i] * BLAS.dot(nq, du, 1, wq, 1)
         l2         = dx[i] * dot(du, wq)
         @. du      = un^2
         # e          = dx[i] * BLAS.dot(nq, du, 1, wq, 1)
         e          = dx[i] * dot(du, wq)
         l1_error += l1
         l2_error += l2
         energy += e
   end
   domain_size = (xmax - xmin)
   l1_error = l1_error/domain_size
   l2_error = sqrt(l2_error/domain_size)
   energy   = energy/domain_size
   # println("Energy = $energy")
   @printf(error_file, "%.32e %.32e %.32e %.32e\n", t, l1_error, l2_error, energy)
   return Dict("l1_error" => l1_error, "l2_error" => l2_error,
               "energy" => energy)
end

#-------------------------------------------------------------------------------
function write_soln(base_name, fcount, iter, time, grid, op, z, u1, ndigits=3)
   # Clear and re-create output directory
   if fcount == 0
      run(`rm -rf output`)
      run(`mkdir output`)
   end
   filename = get_filename(base_name, ndigits, fcount)
   xc = grid.xc
   nx = grid.size
   xg = op["xg"]
   u = @view z[1:end-1]
   out_avg = hcat(xc, u)
   writedlm("output/"*filename*".txt",out_avg, " ") # average solution
   p = plot(xc, u, title="t = $time, iter=$iter")
   xlabel!(p, "x"); ylabel!(p, "u")
   png("output/"*filename*".png")
   x = reduce(vcat, [grid.xf[i] .+ grid.dx[i]*xg for i=1:nx])
   u = vec(u1)
   out = hcat(x, u)
   p = plot(x, u, title="t = $time, iter=$iter")
   xlabel!(p, "x"); ylabel!(p, "u")
   png("output/ptwise_"*filename*".png")
   writedlm("output/ptwise_"*filename*".txt", out, " ") # pointwise solution
   println("Wrote ", filename*".png, "*filename*".txt, ptwise_"
           *filename*".txt to output.")
   fcount += 1
   return fcount
end

end
