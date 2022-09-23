module Basis

using FastGaussQuadrature
using StaticArrays
using Printf
using DoubleFloats
using ArbNumerics
using Kronecker: ⊗

#-------------------------------------------------------------------------------
# Legendre polynomials on [-1,+1]
#-------------------------------------------------------------------------------
function Legendre(n, x)
   if n == 0
      value = 1.0
   elseif n == 1
      value = x
   else
      value = ((2.0*n-1.0)/n * x * Legendre(n-1, x)
               - (n-1.0)/n * Legendre(n-2, x))
   end

   return value
end

#-------------------------------------------------------------------------------
# Derivative of Legendre
#-------------------------------------------------------------------------------
function dLegendre(n, x)
   if n == 0
      value = 0.0
   elseif n == 1
      value = 1.0
   else
      value = n * Legendre(n-1, x) + x * dLegendre(n-1, x)
   end

   return value
end

#-------------------------------------------------------------------------------
# Return n points and weights for the interval [0,1]
#-------------------------------------------------------------------------------
function weights_and_points(n, type)
   if type == "gl"
      x, w = gausslegendre(n)
   elseif type == "gll"
      x, w = gausslobatto(n)
   else
      println("Unknown solution points")
      @assert false
   end
   w *= 0.5
   x  = 0.5*(x .+ 1.0)
   return x, w
end

#-------------------------------------------------------------------------------
# xp = set of grid points
# Returns i'th Lagrange polynomial value at x
#-------------------------------------------------------------------------------
function Lagrange(i, xp, x)
   value = 1.0
   n     = length(xp)
   for j=1:n
      if j != i
         value *= (x - xp[j]) / (xp[i] - xp[j])
      end
   end
   return value
end

#-------------------------------------------------------------------------------
# Vandermonde Matrix for Lagrange polynomials
# xp: grid points
# x:  evaluation points
#-------------------------------------------------------------------------------
function Vandermonde_lag(xp, x)
   n = length(xp)
   m = length(x)
   V = zeros(typeof(Lagrange(1, xp, x[1])), m, n)
   for j=1:n
      for i=1:m
         V[i,j] = Lagrange(j, xp, x[i])
      end
   end
   return V
end

#-------------------------------------------------------------------------------
function barycentric_weights(x)
   n = length(x)
   w = ones(eltype(x), n)

   for j=2:n
      for k in 1:j-1
         w[k] *= x[k] - x[j] # all i > j cases
         w[j] *= x[j] - x[k] # all i < j cases
      end
   end

   value = 1.0 ./ w
   return value
end

#-------------------------------------------------------------------------------
# Differentiation matrix
# D[i,j] = l_j'(x_i)
#-------------------------------------------------------------------------------
function diff_mat(x)
   w = barycentric_weights(x)
   n = length(x)
   D = zeros(eltype(w), n, n)

   for j=1:n
      for i=1:n
         if j != i
            D[i,j] = (w[j]/w[i]) * 1.0/(x[i]-x[j])
            D[i,i]-= D[i,j]
         end
      end
   end
   return D
end

#-------------------------------------------------------------------------------
# Stiffness matrix
# S[i,j] = ∫ ℓi*ℓj'
#-------------------------------------------------------------------------------
function stiffness_matrix(x, w, Dm) # TODO - Do better
   n = length(x)
   RealT = eltype(w)
   S, Sdw = zeros(RealT, n, n), zeros(RealT, n, n)
   for j=1:n
      for i=1:n
         S[i,j] = Dm[i,j]*w[i]
         Sdw[i,j] = Dm[i,j]
      end
   end
   return S, Sdw
end


#-------------------------------------------------------------------------------
# FR Radau correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function gl_radau(k, x)
    value = 0.5 * (-1)^k * (Legendre(k,x) - Legendre(k+1,x))
    return value
end

function gr_radau(k,x)
    value = 0.5 * (Legendre(k,x) + Legendre(k+1,x))
    return value
end

#-------------------------------------------------------------------------------
# Derivatives of FR Radau correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function dgl_radau(k, x)
    value = 0.5 * (-1)^k * (dLegendre(k,x) - dLegendre(k+1,x))
    return value
end

function dgr_radau(k, x)
    value = 0.5 * (dLegendre(k,x) + dLegendre(k+1,x))
    return value
end

#-------------------------------------------------------------------------------
# FR g2 correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function gl_g2(k, x)
   value = 0.5 * (-1)^k * (Legendre(k,x) - ((k+1.0)*Legendre(k-1,x) +
                                             k*Legendre(k+1,x))/(2.0*k+1.0))
   return value
end

function gr_g2(k,x)
   value = gl_g2(k,-x)
   return value
end

#-------------------------------------------------------------------------------
# Derivatives of FR g2 correction functions
# x is in [-1,1]
#-------------------------------------------------------------------------------
function dgl_g2(k, x)
   value = 0.5 * (-1)^k * (1.0 - x) * dLegendre(k,x)
   return value
end

function dgr_g2(k, x)
   value = -dgl_g2(k,-x)
   return value
end

#-------------------------------------------------------------------------------
# sol_pts = gl, gll
# N       = degree
#-------------------------------------------------------------------------------
function fr_operators(N, sol_pts, cor_fun, RealT = Float64)
   println("Setting up differentiation operators")
   @printf("   Degree     = %d\n", N)
   @printf("   Sol points = %s\n", sol_pts)
   @printf("   Cor fun    = %s\n", cor_fun)

   nd = N + 1 # number of dofs
   xg, wg = weights_and_points(nd, sol_pts)

   # Required to evaluate solution at face
   Vl, Vr = zeros(RealT, nd), zeros(RealT, nd)
   for i=1:nd
      Vl[i] = Lagrange(i, xg, 0.0)
      Vr[i] = Lagrange(i, xg, 1.0)
   end

   # Correction terms
   if cor_fun == "radau"
      dgl, dgr = dgl_radau, dgr_radau
   elseif cor_fun == "g2"
      dgl, dgr = dgl_g2, dgr_g2
   else
      prinln("Unknown cor_fun = ",cor_fun)
      @assert false
   end

   bl, br = zeros(RealT, nd), zeros(RealT, nd)
   for i=1:nd
      bl[i] = 2.0 * dgl(N, 2.0*xg[i]-1.0)
      br[i] = 2.0 * dgr(N, 2.0*xg[i]-1.0)
   end

   # Differentiation matrix
   Dm = diff_mat(xg)
   D1 = Dm - bl * Vl' - br * Vr'


   # Stiffness matrix
   S, Sdw = stiffness_matrix(xg, wg, Dm)
   ader_factor = inv( Vr⊗Vr' - S )

   op = Dict("degree" => N,
             "xg" => SVector{nd,RealT}(xg), "wg" => SVector{nd,RealT}(wg),
             "Vl" => SVector{nd,RealT}(Vl), "Vr" =>SVector{nd,RealT}(Vr),
             "bl" => SVector{nd,RealT}(bl), "br" => SVector{nd,RealT}(br),
             "Dm" => SMatrix{nd,nd,RealT}(Dm), "D1" => SMatrix{nd,nd,RealT}(D1),
             "S"  => SMatrix{nd,nd,RealT}(S),
             "Sdw" => SMatrix{nd,nd,RealT}(Sdw),
             "ader_factor" => SMatrix{nd,nd,RealT}(ader_factor))

   return op
end

export weights_and_points
export fr_operators
export Vandermonde_lag

end
