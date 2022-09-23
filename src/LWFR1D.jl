#-------------------------------------------------------------------------------
# Allocate solution arrays needed by LWFR in 1d
#-------------------------------------------------------------------------------
function setup_arrays_lwfr(grid, scheme, uEltype = Float64)
   gArray(nx) = OffsetArray(zeros(uEltype, nx+2),
                            OffsetArrays.Origin(0))
   gArray(n1,nx) = OffsetArray(zeros(uEltype, n1,nx+2),
                               OffsetArrays.Origin(1,0))
   # Allocate memory
   N = scheme["degree"]
   nd = N + 1
   nx = grid.size
   u1  = zeros(uEltype, nd,nx)
   ua  = gArray(nx)
   res = gArray(nd, nx)
   Fb  = gArray(2,nx)
   Ub  = gArray(2,nx)
   return u1, ua, res, Fb, Ub
end

#-------------------------------------------------------------------------------
# Fill ghost values
#-------------------------------------------------------------------------------
function update_ghost_values_lwfr!(problem, eq, grid, op, Fb, Ub, t, dt)
   update_ghost_values_periodic!(problem, Fb, Ub)

   if problem["periodic_x"]
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
         ub, fb = 0.0, 0.0
         for l=1:nd
            tq = t + xg[l]*dt
            bvalue = boundary_value(x,tq)
            ub += bvalue * wg[l]
            fb += flux(x,bvalue,eq) * wg[l]
         end
         Ub[2, 0] = ub
         Fb[2, 0] = fb
   elseif left == neumann
      Ub[2, 0] = Ub[1, 1]
      Fb[2, 0] = Fb[1, 1]
   else
      println("Incorrect bc specified at left.")
      @assert false
   end

   if right == dirichlet
      x  = xf[nx+1]
      ub, fb = 0.0, 0.0
      for l=1:nd
         tq = t + xg[l]*dt
         bvalue = boundary_value(x,tq)
         ub += bvalue * wg[l]
         fb += flux(x,bvalue,eq) * wg[l]
      end
      Ub[1, nx+1] = ub
      Fb[1, nx+1] = fb
   elseif right == neumann
      Ub[1, nx+1] = Ub[2, nx]
      Fb[1, nx+1] = Fb[2, nx]
   else
      println("Incorrect bc specified at right.")
      @assert false
   end
   return nothing
end

@inline function interpolate_to_face_diss1!(Vl, Vr, u, U, Ub)
   nd = length(U)

   # Interpolate u
   Ub[1] = dot(u, Vl)
   Ub[2] = dot(u, Vr)
   return nothing
end

@inline function interpolate_to_face_diss2!(Vl, Vr, u, U, Ub)
   nd = length(U)

   # Interpolate u
   # Ub[1] = dot(u, Vl)
   # Ub[2] = dot(u, Vr)
   Ub[1] = dot(U, Vl)
   Ub[2] = dot(U, Vr)
   return nothing
end
#-------------------------------------------------------------------------------
# Compute cell residual for degree=1 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_1!(eq, grid, op, scheme, t, dt, u1, res, Fb, Ub)
   flux       = eq.flux
   xg, Dm, D1 = op["xg"], op["Dm"], op["D1"]
   Vl, Vr     = op["Vl"], op["Vr"]
   nd         = length(xg)
   nx         = grid.size
   bflux      = scheme["bflux"]
   if scheme["diss"] == 1
      interpolate_soln_to_face! = interpolate_to_face_diss1!
   else
      interpolate_soln_to_face! = interpolate_to_face_diss2!
   end
   RealT = eltype(grid.dx)
   uEltype = eltype(u1)
   for i=1:nx # Loop over cells
      dx = grid.dx[i]
      xc = grid.xc[i]
      lamx = dt/dx
      # Some local variables
      x = Array{RealT}(undef,nd)
      f = Array{uEltype}(undef,nd)
      F = Array{uEltype}(undef,nd)
      ut, U  = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      up, um = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      # Solution points
      @. x = xc - 0.5 * dx + xg * dx
      u = @view u1[:,i]
      # Compute flux at all solution points
      compute_flux!(eq, flux, x, u, f)
      mul!(ut, Dm, f, -lamx, false)
      for ix=1:nd # Loop over solution points
         um[ix] = u[ix] - ut[ix]
         up[ix] = u[ix] + ut[ix]
         fm = flux(x[ix], um[ix], eq)
         fp = flux(x[ix], up[ix], eq)
         ft = 0.5 * (fp - fm)
         F[ix] = f[ix] + 0.5 * ft
         U[ix] = u[ix] + 0.5 * ut[ix]
      end
      r = @view res[:, i]
      mul!(r, D1, F, lamx, false)
      # Interpolate to faces
      @views interpolate_soln_to_face!(Vl, Vr, u, U, Ub[:,i])
      if bflux == extrapolate
         @views interpolate_to_face!(Vl, Vr, F, Fb[:,i])
      else
         xl, xr = grid.xf[i], grid.xf[i+1]
         ul  = dot(u, Vl)
         ur = dot(u, Vr)
         upl, uml = dot(up, Vl), dot(um, Vl)
         upr, umr = dot(up, Vr), dot(um, Vr)
         fml, fpl = flux(xl, uml, eq), flux(xl, upl, eq)
         fmr, fpr = flux(xr, umr, eq), flux(xr, upr, eq)
         fl  = flux(xl, ul, eq)
         fr  = flux(xr, ur, eq)
         ftl = 0.5 * (fpl - fml)
         ftr = 0.5 * (fpr - fmr)
         Fb[1,i] = fl + 0.5 * ftl
         Fb[2,i] = fr + 0.5 * ftr
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=2 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_2!(eq, grid, op, scheme, t, dt, u1, res, Fb, Ub)
   flux       = eq.flux
   xg, Dm, D1 = op["xg"], op["Dm"], op["D1"]
   Vl, Vr     = op["Vl"], op["Vr"]
   nd         = length(xg)
   nx         = grid.size
   bflux      = scheme["bflux"]
   if scheme["diss"] == 1
      interpolate_soln_to_face! = interpolate_to_face_diss1!
   else
      interpolate_soln_to_face! = interpolate_to_face_diss2!
   end
   RealT = eltype(grid.dx)
   uEltype = eltype(u1)
   for i=1:nx
      dx = grid.dx[i]
      xc = grid.xc[i]
      lamx = dt/dx
      # Some local variables
      x  = Array{RealT}(undef,nd)
      f  = Array{uEltype}(undef,nd)
      ft = Array{uEltype}(undef,nd)
      ut,utt = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      um,up  = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      F      = Array{uEltype}(undef,nd)
      U      = Array{uEltype}(undef,nd)

      # Solution points
      @. x = xc - 0.5 * dx + xg * dx
      u = @view u1[:,i]
      # Compute flux at all solution points
      compute_flux!(eq, flux, x, u, f)
      mul!(ut, Dm, f, -lamx, false)
      # computes and stores ft, gt and puts them in respective place
      for ix=1:nd # Loop over solution points
         um[ix] = u[ix] - ut[ix]
         up[ix] = u[ix] + ut[ix]
         fm = flux(x[ix], um[ix], eq)
         fp = flux(x[ix], up[ix], eq)
         ft[ix] = 0.5 * (fp - fm)
         F[ix] = f[ix] + 0.5 * ft[ix]
         U[ix] = u[ix] + 0.5 * ut[ix]
      end
      mul!(utt, Dm, ft, -lamx, false)
      # computes ftt, gtt and puts them in respective place; no need to store
      for ix=1:nd # Loop over solution points
         um[ix] += 0.5 * utt[ix]
         up[ix] += 0.5 * utt[ix]
         fm = flux(x[ix], um[ix], eq)
         fp = flux(x[ix], up[ix], eq)
         ftt = fp - 2.0 * f[ix] + fm
         F[ix] += 1.0/6.0 * ftt
         U[ix] += 1.0/6.0 * utt[ix]
      end
      r = @view res[:,i]
      mul!(r, D1, F, lamx, false)
      # Interpolate to faces
      @views interpolate_soln_to_face!(Vl, Vr, u, U, Ub[:,i])
      # @views interpolate_to_face!(Vl, Vr, U, Ub[:,i])
      if bflux == extrapolate
         @views interpolate_to_face!(Vl, Vr, F, Fb[:,i])
      else
         xl, xr = grid.xf[i], grid.xf[i+1]
         ul = dot(u, Vl)
         ur = dot(u, Vr)
         upl, uml = dot(up, Vl), dot(um, Vl)
         upr, umr = dot(up, Vr), dot(um, Vr)
         fl       = flux(xl, ul, eq)
         fr       = flux(xr, ur, eq)
         fml, fpl = flux(xl, uml, eq), flux(xl, upl, eq)
         fmr, fpr = flux(xr, umr, eq), flux(xr, upr, eq)
         ftl  = 0.5 * (fpl - fml)
         ftr  = 0.5 * (fpr - fmr)
         fttl = fpl - 2.0 * fl + fml
         fttr = fpr - 2.0 * fr + fmr
         Fb[1,i] = fl + 0.5 * ftl + (1.0/6.0) * fttl
         Fb[2,i] = fr + 0.5 * ftr + (1.0/6.0) * fttr
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=3 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_3!(eq, grid, op, scheme, t, dt, u1, res, Fb, Ub)
   flux       = eq.flux
   xg, Dm, D1 = op["xg"], op["Dm"], op["D1"]
   Vl, Vr     = op["Vl"], op["Vr"]
   nd         = length(xg)
   nx         = grid.size
   bflux      = scheme["bflux"]
   if scheme["diss"] == 1
      interpolate_soln_to_face! = interpolate_to_face_diss1!
   else
      interpolate_soln_to_face! = interpolate_to_face_diss2!
   end
   RealT = eltype(grid.dx)
   uEltype = eltype(u1)
   for i=1:nx # Loop over cells
      dx = grid.dx[i]
      xc = grid.xc[i]
      lamx = dt/dx
      # Some local variables
      x       = Array{RealT}(undef,nd)
      um,up   = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      umm,upp = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      f       = Array{uEltype}(undef,nd)
      ft      = Array{uEltype}(undef,nd)
      ftt     = Array{uEltype}(undef,nd)
      ut,utt  = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      uttt    = Array{uEltype}(undef,nd)
      F       = Array{uEltype}(undef,nd)
      U       = Array{uEltype}(undef,nd)

      # Solution points
      @. x = xc - 0.5 * dx + xg * dx
      u = @view u1[:,i]
      # Compute flux at all solution points
      compute_flux!(eq, flux, x, u, f)
      mul!(ut, Dm, f, -lamx, false)
      # computes and stores ft, gt and puts them in respective place
      for ix=1:nd # Loop over solution points
         um[ix]  = u[ix] - ut[ix]
         up[ix]  = u[ix] + ut[ix]
         umm[ix] = u[ix] - 2.0 * ut[ix]
         upp[ix] = u[ix] + 2.0 * ut[ix]
         fm  = flux(x[ix], um[ix], eq)
         fp  = flux(x[ix], up[ix], eq)
         fmm = flux(x[ix], umm[ix], eq)
         fpp = flux(x[ix], upp[ix], eq)
         ft[ix] = 1.0/12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
         F[ix]  = f[ix] + 0.5 * ft[ix]
         U[ix]  = u[ix] + 0.5 * ut[ix]
      end
      mul!(utt, Dm, ft, -lamx, false)
      # computes ftt, gtt and puts them in respective place and stores them
      for ix=1:nd # Loop over solution points
         um[ix] += 0.5 * utt[ix]
         up[ix] += 0.5 * utt[ix]
         fm = flux(x[ix], um[ix], eq)
         fp = flux(x[ix], up[ix], eq)
         ftt[ix] = fp - 2.0 * f[ix] + fm
         F[ix] += 1.0/6.0 * ftt[ix]
         U[ix] += 1.0/6.0 * utt[ix]
      end
      mul!(uttt, Dm, ftt, -lamx, false)
      # computes fttt, gttt and puts them in respective place; no need to store
      for ix=1:nd # Loop over solution points
         um[ix]  -= 1.0/6.0 * uttt[ix]
         up[ix]  += 1.0/6.0 * uttt[ix]
         umm[ix] += 2.0 * utt[ix] - 4.0/3.0 * uttt[ix]
         upp[ix] += 2.0 * utt[ix] + 4.0/3.0 * uttt[ix]
         fm = flux(x[ix], um[ix], eq)
         fp = flux(x[ix], up[ix], eq)
         fmm = flux(x[ix], umm[ix], eq)
         fpp = flux(x[ix], upp[ix], eq)
         fttt = 0.5 * (fpp - 2.0 * fp + 2.0 * fm - fmm)
         F[ix] += 1.0/24.0 * fttt
         U[ix] += 1.0/24.0 * uttt[ix]
      end
      r = @view res[:,i]
      mul!(r, D1, F, lamx, false)
      # Interpolate to faces
      @views interpolate_soln_to_face!(Vl, Vr, u, U, Ub[:,i])
      #@views interpolate_to_face!(Vl, Vr, U, Ub[:,i])
      if bflux == extrapolate
         @views interpolate_to_face!(Vl, Vr, F, Fb[:,i])
      else
         xl, xr = grid.xf[i], grid.xf[i+1]
         # ul = dot(u, Vl)
         # ur = dot(u, Vr)
         ul = dot(u,Vl)
         ur = dot(u,Vr)
         upl, uml = dot(up, Vl), dot(um, Vl)
         upr, umr = dot(up, Vr), dot(um, Vr)
         uppl, umml = dot(upp, Vl), dot(umm, Vl)
         uppr, ummr = dot(upp, Vr), dot(umm, Vr)
         fl         = flux(xl, ul, eq)
         fr         = flux(xr, ur, eq)
         fml, fpl   = flux(xl, uml, eq), flux(xl, upl, eq)
         fmr, fpr   = flux(xr, umr, eq), flux(xr, upr, eq)
         fmml, fppl = flux(xl, umml, eq), flux(xl, uppl, eq)
         fmmr, fppr = flux(xr, ummr, eq), flux(xr, uppr, eq)
         ftl = 1.0/12.0 * (-fppl + 8.0 * fpl - 8.0 * fml + fmml)
         ftr = 1.0/12.0 * (-fppr + 8.0 * fpr - 8.0 * fmr + fmmr)
         fttl  = fpl - 2.0 * fl + fml
         fttr  = fpr - 2.0 * fr + fmr
         ftttl = 0.5 * (fppl - 2.0 * fpl + 2.0 * fml - fmml)
         ftttr = 0.5 * (fppr - 2.0 * fpr + 2.0 * fmr - fmmr)
         Fb[1,i] = fl + 0.5 * ftl + (1.0/6.0) * fttl + (1.0/24.0) * ftttl
         Fb[2,i] = fr + 0.5 * ftr + (1.0/6.0) * fttr + (1.0/24.0) * ftttr
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# Compute cell residual for degree=4 case and for all real cells
#-------------------------------------------------------------------------------
function compute_cell_residual_4!(eq, grid, op, scheme, t, dt, u1, res, Fb, Ub)
   flux       = eq.flux
   xg, Dm, D1 = op["xg"], op["Dm"], op["D1"]
   Vl, Vr     = op["Vl"], op["Vr"]
   nd         = length(xg)
   nx         = grid.size
   bflux      = scheme["bflux"]
   if scheme["diss"] == 1
      interpolate_soln_to_face! = interpolate_to_face_diss1!
   else
      interpolate_soln_to_face! = interpolate_to_face_diss2!
   end
   RealT = eltype(grid.dx)
   uEltype = eltype(u1)
   for i=1:nx # Loop over cells
      dx = grid.dx[i]
      xc = grid.xc[i]
      lamx = dt/dx
      # Some local variables
      x          = Array{RealT}(undef,nd)
      um,up      = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      umm,upp    = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      f          = Array{uEltype}(undef,nd)
      ft         = Array{uEltype}(undef,nd)
      ftt        = Array{uEltype}(undef,nd)
      fttt       = Array{uEltype}(undef,nd)
      ut,utt     = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      uttt,utttt = Array{uEltype}(undef,nd), Array{uEltype}(undef,nd)
      F          = Array{uEltype}(undef,nd)
      U          = Array{uEltype}(undef,nd)

      # Solution points
      @. x = xc - 0.5 * dx + xg * dx
      u = @view u1[:,i]
      # Compute flux at all solution points
      compute_flux!(eq, flux, x, u, f)
      mul!(ut, Dm, f, -lamx, false)
      # computes and stores ft, gt and puts them in respective place
      for ix=1:nd # Loop over solution points
         um[ix]  = u[ix] - ut[ix]
         up[ix]  = u[ix] + ut[ix]
         umm[ix] = u[ix] - 2.0 * ut[ix]
         upp[ix] = u[ix] + 2.0 * ut[ix]
         fm      = flux(x[ix], um[ix], eq)
         fp      = flux(x[ix], up[ix], eq)
         fmm     = flux(x[ix], umm[ix], eq)
         fpp     = flux(x[ix], upp[ix], eq)
         ft[ix]  = 1.0/12.0 * (-fpp + 8.0 * fp - 8.0 * fm + fmm)
         F[ix] = f[ix] + 0.5 * ft[ix]
         U[ix] = u[ix] + 0.5 * ut[ix]
      end
      mul!(utt, Dm, ft, -lamx, false)
      # computes ftt, gtt and puts them in respective place and stores them
      for ix=1:nd # Loop over solution points
         um[ix]  += 0.5 * utt[ix]
         up[ix]  += 0.5 * utt[ix]
         umm[ix] += 2.0 * utt[ix]
         upp[ix] += 2.0 * utt[ix]
         fm  = flux(x[ix], um[ix], eq)
         fp  = flux(x[ix], up[ix], eq)
         fmm = flux(x[ix], umm[ix], eq)
         fpp = flux(x[ix], upp[ix], eq)
         ftt[ix] = 1.0/12.0 * (-fpp + 16.0 * fp - 30.0 * f[ix]
                                    + 16.0 * fm - fmm)
         F[ix] += 1.0/6.0 * ftt[ix]
         U[ix] += 1.0/6.0 * utt[ix]
      end
      mul!(uttt, Dm, ftt, -lamx, false)
      # computes and stores fttt, gttt; and puts them in respective place
      for ix=1:nd # Loop over solution points
         um[ix]  -= 1.0/6.0 * uttt[ix]
         up[ix]  += 1.0/6.0 * uttt[ix]
         umm[ix] -= 4.0/3.0 * uttt[ix]
         upp[ix] += 4.0/3.0 * uttt[ix]
         fm       = flux(x[ix], um[ix], eq)
         fp       = flux(x[ix], up[ix], eq)
         fmm      = flux(x[ix], umm[ix], eq)
         fpp      = flux(x[ix], upp[ix], eq)
         fttt[ix] = 0.5 * (fpp - 2.0 * fp + 2.0 * fm - fmm)
         F[ix] += 1.0/24.0 * fttt[ix]
         U[ix] += 1.0/24.0 * uttt[ix]
      end
      mul!(utttt, Dm, fttt, -lamx, false)
      # computes fttt, gttt and puts them in respective place; no need to store
      for ix=1:nd # Loop over solution points
         um[ix]  += 1.0/24.0 * utttt[ix]
         up[ix]  += 1.0/24.0 * utttt[ix]
         umm[ix] += 2.0/3.0 * utttt[ix]
         upp[ix] += 2.0/3.0 * utttt[ix]
         fm       = flux(x[ix], um[ix], eq)
         fp       = flux(x[ix], up[ix], eq)
         fmm      = flux(x[ix], umm[ix], eq)
         fpp      = flux(x[ix], upp[ix], eq)
         ftttt    = 0.5 * (fpp - 4.0 * fp + 6.0 * f[ix] - 4.0 * fm + fmm)
         F[ix]   += 1.0/120.0 * ftttt
         U[ix]   += 1.0/120.0 * utttt[ix]
      end
      r = @view res[:,i]
      mul!(r, D1, F, lamx, false)
      # Interpolate to faces
      @views interpolate_soln_to_face!(Vl, Vr, u, U, Ub[:,i])
      if bflux == extrapolate
         @views interpolate_to_face!(Vl, Vr, F, Fb[:,i])
      else
         xl, xr = grid.xf[i], grid.xf[i+1]
         ul = dot(u, Vl)
         ur = dot(u, Vr)
         upl, uml = dot(up, Vl), dot(um, Vl)
         upr, umr = dot(up, Vr), dot(um, Vr)
         uppl, umml = dot(upp, Vl), dot(umm, Vl)
         uppr, ummr = dot(upp, Vr), dot(umm, Vr)
         fl         = flux(xl, ul, eq)
         fr         = flux(xr, ur, eq)
         fml, fpl   = flux(xl, uml, eq), flux(xl, upl, eq)
         fmr, fpr   = flux(xr, umr, eq), flux(xr, upr, eq)
         fmml, fppl = flux(xl, umml, eq), flux(xl, uppl, eq)
         fmmr, fppr = flux(xr, ummr, eq), flux(xr, uppr, eq)
         ftl    = 1.0/12.0 * (-fppl + 8.0 * fpl - 8.0 * fml + fmml)
         ftr    = 1.0/12.0 * (-fppr + 8.0 * fpr - 8.0 * fmr + fmmr)
         fttl   = 1.0/12.0 * (-fppl + 16.0 * fpl - 30.0 * fl + 16.0 * fml
                               - fmml)
         fttr   = 1.0/12.0 * (-fppr + 16.0 * fpr - 30.0 * fr + 16.0 * fmr
                               - fmmr)
         ftttl  = 0.5 * (fppl - 2.0 * fpl + 2.0 * fml - fmml)
         ftttr  = 0.5 * (fppr - 2.0 * fpr + 2.0 * fmr - fmmr)
         fttttl = 0.5 * (fppl - 4.0 * fpl + 6.0 * fl - 4.0 * fml + fmml)
         fttttr = 0.5 * (fppr - 4.0 * fpr + 6.0 * fr - 4.0 * fmr + fmmr)
         Fb[1,i] = ( fl + 0.5*ftl + (1.0/6.0)*fttl + (1.0/24.0)*ftttl
                   + (1.0/120.0)*fttttl )
         Fb[2,i] = ( fr + 0.5*ftr + (1.0/6.0)*fttr + (1.0/24.0)*ftttr
                   + (1.0/120.0)*fttttr )
      end
   end
   return nothing
end

#-------------------------------------------------------------------------------
# For real cells: u1 = u1 - res
# Note: u1 does not have ghosts, res has ghosts
#-------------------------------------------------------------------------------
function update_solution!(u1, res)
   nx = size(u1,2)
   r1 = @view res[:,1:nx]
   axpy!(-1.0, r1, u1) # u1 = (-1.0)*r1 + u1
   return nothing
end
