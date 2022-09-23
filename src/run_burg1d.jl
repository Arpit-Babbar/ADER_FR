push!(LOAD_PATH,".")
import EqBurg1D; Eq = EqBurg1D
import FR1D; FR = FR1D
import Roots.find_zero

#------------------------------------------------------------------------------
xmin, xmax = 0.0, 2.0*pi
initial_value(x) = 0.2 * sin(x)
boundary_value(x,t) = 0.0 # dummy function
boundary_condition = [FR.periodic, FR.periodic]
final_time = 2.0

function exact_solution(x,t)
   implicit_eqn(u) = u - initial_value(x .- t*u)
   seed = initial_value(x)
   value = find_zero(implicit_eqn, seed)
   return value
end

degree = 1
solver = "ader"
diss = 2
solution_points = "gl"
correction_function = "radau"
limiter = "none"
bflux = FR.evaluate
numerical_flux = Eq.rusanov

nx = 20
cfl = 0.49
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0 # final_time / 10.0)
compute_error_interval = 0
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = FR.Problem(domain, initial_value, boundary_value, boundary_condition,
                     final_time, exact_solution)
equation = Eq.get_equation()
scheme = FR.Scheme(solver, diss, degree, solution_points, correction_function,
                   numerical_flux, limiter, bflux)
param = FR.Parameters(grid_size, cfl, tvbM, save_iter_interval,
                      save_time_interval, compute_error_interval)
#------------------------------------------------------------------------------

FR.solve(equation, problem, scheme, param)
