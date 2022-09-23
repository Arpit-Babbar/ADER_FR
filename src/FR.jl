using ADER_FR.Basis
using ADER_FR.Grid
using OffsetArrays
using LinearAlgebra
using Printf
using WriteVTK
using TickTock
using DifferentialEquations
using FLoops
using DoubleFloats
using ArbNumerics

@enum BCType periodic dirichlet neumann
@enum BFluxType extrapolate evaluate

global error_file

#-------------------------------------------------------------------------------
# Create a dictionary of problem description
#-------------------------------------------------------------------------------
function Problem(domain::Any,
                 initial_value::Function,
                 boundary_value::Function,
                 boundary_condition::Vector{BCType},
                 final_time::Real,
                 exact_solution::Function)
   problem = Dict("domain" => domain,
                  "initial_value" => initial_value,
                  "boundary_value" => boundary_value,
                  "boundary_condition" => boundary_condition,
                  "final_time" => final_time,
                  "exact_solution" => exact_solution)
   if length(domain)==2
      @assert length(boundary_condition)==2 "Invalid Problem"
      left, right = boundary_condition
      if (left == periodic && right != periodic)
         println("Incorrect use of periodic bc")
         @assert false
      elseif left == periodic && right == periodic
         periodic_x = true
      else
         periodic_x = false
      end
      problem["periodic_x"]=periodic_x
      return problem
   elseif length(domain)==4
      @assert length(boundary_condition)==4 "Invalid Problem"
      left, right, bottom, top = boundary_condition

      if ((left == periodic && right != periodic) ||
         (left != periodic && right == periodic))
         println("Incorrect use of periodic bc")
         @assert false
      elseif left == periodic && right == periodic
         periodic_x = true
      else
         periodic_x = false
      end

      if ((bottom == periodic && top != periodic) ||
         (bottom != periodic && top == periodic))
         println("Incorrect use of periodic bc")
         @assert false
      elseif bottom == periodic && top == periodic
         periodic_y = true
      else
         periodic_y = false
      end
      problem["periodic_x"] = periodic_x
      problem["periodic_y"] = periodic_y
      return problem
   else
      @assert false,"Invalid domain"
   end
end

#-------------------------------------------------------------------------------
# Create a dictionary of parameters
#-------------------------------------------------------------------------------
function Parameters(grid_size::Union{Int64,Vector{Int64}},
                    cfl::Real,
                    tvbM::Real,
                    save_iter_interval::Int64,
                    save_time_interval::Real,
                    compute_error_interval::Int64)
   @assert (cfl >= 0.0) "cfl must be >= 0.0"
   @assert (save_iter_interval >= 0) "save_iter_interval must be >= 0"
   @assert (save_time_interval >= 0.0) "save_time_interval must be >= 0.0"
   @assert (!(save_iter_interval > 0 &&
   save_time_interval > 0.0)) "Both save_(iter,time)_interval > 0"
   param = Dict("grid_size" => grid_size,
                "cfl" => cfl,
                "tvbM" => tvbM,
                "save_iter_interval" => save_iter_interval,
                "save_time_interval" => save_time_interval,
                "compute_error_interval" => compute_error_interval)
   return param
end

#-------------------------------------------------------------------------------
# Create a dictionary of scheme description
#-------------------------------------------------------------------------------
function Scheme(solver::String,
		          diss::Int64,
                degree::Int64,
                solution_points::String,
                correction_function::String,
                numerical_flux::Function,
                limiter::String,
                bflux::BFluxType)
   @assert ((degree > 0 && degree < 5) || solver == "ader") "degree must be >=1 and <=4"
   @assert (solution_points=="gl" ||
            solution_points=="gll") "solution points must be gl or gll"
   Dict("solver" => solver,
	"diss"   => diss,
        "degree" => degree,
        "solution_points" => solution_points,
        "correction_function" => correction_function,
        "numerical_flux" => numerical_flux,
        "limiter" => limiter,
        "bflux" => bflux)
end

#-------------------------------------------------------------------------------
# C = a1 * A1 * B1 + a2 * A2 * B2
#-------------------------------------------------------------------------------
@inline function gemm!(a1, A1, B1, a2, A2, B2, C)
   mul!(C, A1, B1)         # C = A1 * B1
   mul!(C, A2, B2, a2, a1) # C = a1 * C + a2 * A2 * B2
   return nothing
end

#-------------------------------------------------------------------------------
# Limiter function
#-------------------------------------------------------------------------------
function apply_limiter!(grid, scheme, param, op, ua, u1)
   limiter = scheme["limiter"]
   if limiter == "tvb"
      apply_limiter_tvb!(grid, param, op, ua, u1)
   elseif limiter == "none"
      return nothing
   else
      println("Incorrect limiter, exiting...")
      @assert false
   end
end

function minmod(a, b, c, Mdx2)
   if abs(a) < Mdx2
      return a
   end
   s1, s2, s3 = sign(a), sign(b), sign(c)
   if (s1 != s2) || (s2 != s3)
      return 0.0
   else
      slope = s1 * min(abs(a),abs(b),abs(c))
      return slope
   end
end

#-------------------------------------------------------------------------------
# Return string of the form base_name00c with total number of digits = ndigits
#-------------------------------------------------------------------------------
function get_filename(base_name, ndigits, c)
    if c > 10^ndigits - 1
        println("get_filename: Not enough digits !!!")
        println("   ndigits =", ndigits)
        println("   c       =", c)
        @assert false
    end
    number = lpad(c, ndigits, "0")
    return string(base_name, number)
end

#-------------------------------------------------------------------------------
# Adjust dt to reach final time or the next time when solution has to be saved
#-------------------------------------------------------------------------------
function adjust_time_step(problem, param, t, dt)
   # Adjust to reach final time exactly
   final_time = problem["final_time"]
   if t + dt > final_time
      dt = final_time - t
      return dt
   end

   # Adjust to reach next solution saving time
   save_time_interval = param["save_time_interval"]
   if save_time_interval > 0.0
      next_save_time = ceil(t/save_time_interval) * save_time_interval
      # If t is not a plotting time, we check if the next time
      # would step over the plotting time to adjust dt
      if abs(t-next_save_time) > 1e-10 && t + dt - next_save_time > -1e-10
         dt = next_save_time - t
         return dt
      end
   end

   return dt
end

#-------------------------------------------------------------------------------
# Check if we have to save solution
#-------------------------------------------------------------------------------
function save_solution(problem, param, t, iter)
   # Save if we have reached final time
   final_time = problem["final_time"]
   if abs(t - final_time) < 1.0e-10
      return true
   end

   # Save after specified time interval
   save_time_interval = param["save_time_interval"]
   if save_time_interval > 0.0
      k1, k2 = ceil(t/save_time_interval), floor(t/save_time_interval)
      if (abs(t-k1*save_time_interval) < 1e-10 ||
          abs(t-k2*save_time_interval) < 1e-10)
         return true
      end
   end

   # Save after specified number of iterations
   save_iter_interval = param["save_iter_interval"]
   if save_iter_interval > 0
      if mod(iter, save_iter_interval) == 0
         return true
      end
   end

   return false
end

#-------------------------------------------------------------------------------
include("LWFR.jl")
include("RKFR.jl")
include("ADER.jl")
#-------------------------------------------------------------------------------
# Solve the problem
#-------------------------------------------------------------------------------
function solve(equation, problem, scheme, param)
   println("Number of julia threads = ", Threads.nthreads())
   println("Number of BLAS  threads = ", BLAS.get_num_threads())
   global error_file = open("error.txt", "w")
   if scheme["solver"] == "lwfr"
      out = solve_lwfr(equation, problem, scheme, param)
   elseif scheme["solver"] == "rkfr"
      out = solve_rkfr(equation, problem, scheme, param)
   elseif scheme["solver"] == "ader"
      out = solve_ader(equation, problem, scheme, param)
   else
      println("Solver not implemented")
      @assert false
   end
   close(error_file)
   return out
end
