using TensorOperations
using StaticArrays
using Kronecker
using NLsolve: fixedpoint

function setup_arrays_ader(grid, scheme, RealT = Float64)
   gArray(nx) = OffsetArray(zeros(RealT, nx+2),
                            OffsetArrays.Origin(0))
   gArray(n,nx) = OffsetArray(zeros(RealT, n, nx+2),
                              OffsetArrays.Origin(1,0))
   gArray(n1,n2,nx) = OffsetArray(zeros(RealT, n1,n2,nx+2),
                                  OffsetArrays.Origin(1,1,0))
   # Allocate memory
   N   = scheme["degree"]
   nd  = N + 1
   nx  = grid.size
   q   = gArray(nd,nd,nx) # predictor
   u1  = zeros(RealT, nd,nx)
   qa  = gArray(nd,nx)
   ua  = gArray(nx)
   res = gArray(nd,nx)
   Fb  = gArray(2,nd,nx)
   Ub  = gArray(2,nd,nx)
   return q, u1, qa, ua, res, Fb, Ub
end

function compute_cell_average_ader!(qa, q, grid, problem, op)
   nx = grid.size
   wg = op["wg"]
   Vl, Vr = op["Vl"], op["Vr"]
   nd = length(wg)
   for i=1:nx
      for τ=1:nd
         u1 = @view q[:,τ,i]
         qa[τ,i] = dot(wg, u1)
      end
   end
   # Update ghost values of ua by periodicity or with face averages
   if problem["periodic_x"]
      qa[:,0]    = qa[:,nx]
      qa[:,nx+1] = qa[:,1]
   else
      for τ=1:nd
         u1 = @view q[:,τ,1]
         qa[τ,0] = dot(u1,Vl)
         u1 = @view q[:,τ,nx]
         qa[τ,nx] = dot(u1,Vr)
      end
   end


   return nothing
end

function update_ghost_values_ader!(problem, eq, grid, op, Fb, Ub, t, dt)
   nx = size(Fb,3)-2
   nd = size(Fb,2)
   if problem["periodic_x"]
      # Left ghost cells
      for τ=1:nd
         copyto!(Ub, CartesianIndices((2:2, τ:τ, 0:0)),
                 Ub, CartesianIndices((2:2, τ:τ, nx:nx)))
         copyto!(Fb, CartesianIndices((2:2, τ:τ, 0:0)),
                 Fb, CartesianIndices((2:2, τ:τ, nx:nx)))

         # Right ghost cells
         copyto!(Ub, CartesianIndices((1:1, τ:τ, nx+1:nx+1)),
                 Ub, CartesianIndices((1:1, τ:τ, 1:1)))
         copyto!(Fb, CartesianIndices((1:1, τ:τ, nx+1:nx+1)),
                 Fb, CartesianIndices((1:1, τ:τ, 1:1)))
      end
      return nothing
   end
   nx = grid.size
   nd, xg, wg = op["degree"]+1, op["xg"], op["wg"]
   flux     = eq.flux
   dx, xf = grid.dx, grid.xf
   left, right = problem["boundary_condition"]
   boundary_value = problem["boundary_value"]

   if left == dirichlet
         x = xf[1]
         for τ=1:nd
            tq = t + xg[τ]*dt
            bvalue = boundary_value(x,tq)
            Ub[2,τ,0] = bvalue
            Fb[2,τ,0] = flux(x,bvalue,eq)
         end
   elseif left == neumann
      for τ = 1:nd
         Ub[2, τ, 0] = Ub[1, τ, 1]
         Fb[2, τ, 0] = Fb[1, τ, 1]
      end
   else
      println("Incorrect bc specified at left.")
      @assert false
   end

   if right == dirichlet
      x  = xf[nx+1]
      for τ = 1:nd
         tq = t + xg[τ]*dt
         bvalue = boundary_value(x,tq)
         Ub[2, τ, 0] = bvalue
         Fb[2, τ, 0] = flux(x,bvalue,eq)
      end
   elseif right == neumann
      for τ = 1:nd
         Ub[1, τ, nx+1] = Ub[2, τ, nx]
         Fb[1, τ, nx+1] = Fb[2, τ, nx]
      end
   else
      println("Incorrect bc specified at right.")
      @assert false
   end
end

function compute_predictor!(eq, grid, op, dt, u1, q)
   Vl, Vr, S, Sdw = op["Vl"], op["Vr"], op["S"], op["Sdw"] # Stiffness matrix
   ader_factor = op["ader_factor"]
   xg, wg = op["xg"], op["wg"]
   nx = grid.size
   nd = length(Vl)
   flux = eq.flux
   function f!(rhs, Q, x, u, lamx)
      rhs .= zero(eltype(rhs))
      for τ in 1:nd, ξ in 1:nd
         rhs[ξ,τ] += -u[ξ]*Vl[τ] + Q[ξ, τ]
         for k in 1:nd
            rhs[ξ,τ] += (
               - Q[ξ,k]*S[k,τ]                   # Time derivative term
               + Q[ξ,k]*Vr[k]*Vr[τ]              # Time boundary terms
            )
         end
         f_node = flux(x, Q[ξ, τ], eq)
         for ξξ in 1:nd
            # F += S * f for each variable
            # F[ξξ,τ] = ∑_ξ S[ξξ,ξ] * f[ξ,τ] for each variable
            rhs[ξξ, τ] += lamx * S[ξξ, ξ] * f_node * wg[τ] / wg[ξξ]
         end
      end
   end
   for i=1:nx
      u = @view u1[:,i]
      dx = grid.dx[i]
      x  = grid.xf[i] .+ xg*dx
      lamx = dt/dx
      sol = @views fixedpoint(
                           (F,Q)->f!(F,Q,x,u,lamx), # equation F = 0 to be solved
                            q[:,:,i],               # initial guess
                            ftol = 5e-16,
                            # method = :anderson,
                            # m = 0,
			    method = :newton
                          )
      # @assert sol.f_converged == true
      q[:,:,i] = sol.zero
   end
end

function compute_cell_residual_ader!(eq, grid, op, scheme, dt, q, res, Fb, Ub)
   flux           = eq.flux
   xg, wg, D1, Vl, Vr = op["xg"], op["wg"], op["D1"], op["Vl"], op["Vr"]

   # Dm = op["Dm"]
   nx  = grid.size
   nd  = length(xg)
   bflux_ind = scheme["bflux"]
   fill!(res, zero(eltype(res)))
   for i=1:nx
      for τ=1:nd
         dx = grid.dx[i]
         xc = grid.xc[i]
         lamx = dt/dx
         # Solution points
         x   = Array{eltype(xc)}(undef,nd)
         f   = Array{eltype(Fb)}(undef,nd)
         @. x       = xc - 0.5 * dx + xg * dx
         u = @view q[:,τ,i]
         # Compute flux at all solution points
         compute_flux!(eq, flux, x, u, f)
         res[:,i] += lamx * wg[τ] * D1 * f
         @views interpolate_to_face!(Vl, Vr, u, Ub[:, τ, i])
         if bflux_ind == extrapolate
            @views interpolate_to_face!(Vl, Vr, f, Fb[:, τ, i])
         else
            xl, xr = grid.xf[i], grid.xf[i+1]
            Fb[1,τ,i] = flux(xl, Ub[1,τ,i], eq)
            Fb[2,τ,i] = flux(xr, Ub[2,τ,i], eq)
         end
      end
   end
   return nothing
end

function compute_face_residual_ader!(eq, grid, op, scheme, dt, Fb, Ub, qa, res)
   bl, br = op["bl"], op["br"]
   xg, wg = op["xg"], op["wg"]
   nd = length(xg)
   nx = grid.size
   dx = grid.dx
   xf = grid.xf
   num_flux = scheme["numerical_flux"]

   lamx = OffsetArray(zeros(eltype(dt), nx+2), OffsetArrays.Origin(0))
   lamx[1:nx] = dt ./ dx

   # Vertical faces, x flux
   for i=1:nx+1
      for τ=1:nd
         x  = xf[i]
         Fn = num_flux(x, qa[τ,i-1], qa[τ,i],
                       Fb[2,τ,i-1], Fb[1,τ,i],
                       Ub[2,τ,i-1], Ub[1,τ,i], eq, 1)
         r = @view res[:,i-1]
         axpy!(lamx[i-1]*Fn*wg[τ], br, r)
         r = @view res[:,i]
         axpy!(lamx[i]*Fn*wg[τ], bl, r)
      end
   end
   return nothing
end
