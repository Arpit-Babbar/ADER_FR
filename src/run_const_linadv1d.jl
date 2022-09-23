using StaticArrays
using ADER_FR
Eq = ADER_FR.EqLinAdv1D
FR = ADER_FR.FR1D
using ADER_FR.InitialValues
using DoubleFloats
using ArbNumerics
#------------------------------------------------------------------------------
xmin, xmax = -1.0, 1.0
velocity(x) = 5.0

function wpack1d(x)
   xmin, xmax = -1.0, 1.0
   L = xmax - xmin
   if x > xmax
      y = x - L*floor((x+xmax)/(xmax-xmin))
   elseif x < xmin
      y = x + L*floor((xmax-x)/(xmax-xmin))
   else
      y = x
   end
   value = sin(10.0*pi*y)*exp(-10*y^2)
   return value
end

# boundary_condition = [FR.dirichlet, FR.neumann]
boundary_condition = [FR.periodic, FR.periodic]
final_time = 1.0
# initial_value(x) =  sinpi(2.0*x)
initial_value(x) = wpack1d(x)
exact_solution(x,t) = initial_value(x-velocity(x) * t)
boundary_value(x,t) = exact_solution(x,t) # dummy function

degree = 4
solver = "lwfr"
diss = 2
solution_points = "gl"
correction_function = "radau"
numerical_flux = Eq.rusanov
limiter = "none"
bflux = FR.evaluate

nx = 20
cfl = 0.0
tvbM = 0.0
save_iter_interval = 0
save_time_interval = 0.0
compute_error_interval = 1
cfl_safety_factor = 0.98
#------------------------------------------------------------------------------
grid_size = nx
domain = [xmin, xmax]
problem = FR.Problem(domain, initial_value, boundary_value, boundary_condition,
                     final_time, exact_solution)
equation = Eq.get_equation(velocity)
scheme = FR.Scheme(solver, diss, degree, solution_points, correction_function,
                   numerical_flux, limiter, bflux)
param = FR.Parameters(grid_size, cfl, tvbM, save_iter_interval,
                      save_time_interval, compute_error_interval)
#------------------------------------------------------------------------------
error, u1 = FR.solve(equation, problem, scheme, param)

