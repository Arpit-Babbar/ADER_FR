function view_res(grid, res)
   nx = grid.size
   r1 = @view res[:,1:nx]
   return r1
end

#------------------------------------------------------------------------------
function setup_arrays_rkfr(grid, scheme, uEltype = Float64)
   gArray(nx) = OffsetArray(zeros(uEltype, nx+2),
                            OffsetArrays.Origin(0))
   gArray(n1,nx) = OffsetArray(zeros(uEltype, n1,nx+2),
                                     OffsetArrays.Origin(1,0))
   # Allocate memory
   N = scheme["degree"]
   nd = N + 1
   nx = grid.size
   u0  = zeros(uEltype, nd,nx)
   u1  = zeros(uEltype, nd,nx)
   ua  = gArray(nx)
   res = gArray(nd, nx)
   Fb  = gArray(2,nx)
   ub  = gArray(2,nx)
   return u0, u1, ua, res, Fb, ub
end

#------------------------------------------------------------------------------
function update_ghost_values_rkfr!(problem, equation, grid, op, Fb, Ub, t)
   update_ghost_values_periodic!(problem, Fb, Ub)

   if problem["periodic_x"]
      return nothing
   end

   nx = grid.size
   flux, eq = equation["flux"], equation["eq"]
   xf = grid.xf
   left, right = problem["boundary_condition"]
   boundary_value = problem["boundary_value"]

   if left == dirichlet
      x = xf[1]
      ub = boundary_value(x,t)
      Ub[2, 0] = ub
      Fb[2, 0] = flux(x,ub,eq)
   elseif left == neumann
      Ub[2, 0] = Ub[1, 1]
      Fb[2, 0] = Fb[1, 1]
   else
      println("Incorrect bc specified at left.")
      @assert false
   end

   if right == dirichlet
      x = xf[nx+1]
      ub = boundary_value(x,t)
      Ub[1, nx+1] = ub
      Fb[1, nx+1] = flux(x,ub,eq)
   elseif right == neumann
      Ub[1, nx+1] = Ub[2, nx]
      Fb[1, nx+1] = Fb[2, nx]
   else
      println("Incorrect bc specified at right.")
      @assert false
   end

   return nothing
end

#------------------------------------------------------------------------------
function compute_cell_residual_rkfr!(eq, grid, op, scheme, t, dt, u1, res, Fb,
                                     ub)
   flux           = eq.flux
   xg, D1, Vl, Vr = op["xg"], op["D1"], op["Vl"], op["Vr"]
   RealT = eltype(xg)
   uEltype = eltype(u1)
   nx  = grid.size
   nd  = length(xg)
   bflux_ind = scheme["bflux"]
   for i=1:nx
      dx = grid.dx[i]
      xc = grid.xc[i]
      lamx = dt/dx
      x   = Array{RealT}(undef,nd)
      f   = Array{uEltype}(undef,nd)
      # Solution points
      @. x       = xc - 0.5 * dx + xg * dx
      u = @view u1[:,i]
      # Compute flux at all solution points
      compute_flux!(eq, flux, x, u, f)
      res[:,i] = lamx * D1 * f
      @views interpolate_to_face!(Vl, Vr, u, ub[:,i])
      if bflux_ind == extrapolate
         @views interpolate_to_face!(Vl, Vr, f, Fb[:,i])
      else
         xl, xr = grid.xf[i], grid.xf[i+1]
         Fb[1,i] = flux(xl, ub[1,i], eq)
         Fb[2,i] = flux(xr, ub[2,i], eq)
      end
   end
   return nothing
end

#------------------------------------------------------------------------------
# For use with DifferentialEquations
#------------------------------------------------------------------------------
function compute_residual_rkfr!(du, u::Array{<:Real,2}, p, t)
   eq, problem, scheme, param, grid, op, Fb, ub, ua, res = p
   dt = -1.0
   compute_residual_rkfr!(eq, problem, grid, op, scheme, t, dt, u,
                          Fb, ub, ua, res)
   nd = scheme["degree"] + 1
   nx = grid.size
   copyto!(du,  CartesianIndices((1:nd,1:nx)),
           res, CartesianIndices((1:nd,1:nx)))
   return nothing
end

function axpyz!(a, x::SubArray{<:Real,2}, y, z)
   n, nx = size(x)
   for i=1:nx
      for ii=1:n
         z[ii,i] = a * x[ii,i] + y[ii,i]
      end
   end
   return nothing
end
