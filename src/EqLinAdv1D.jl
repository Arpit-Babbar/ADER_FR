module EqLinAdv1D

struct LinAdv1D
   flux::Function
   speed::Function
   velocity::Function
   name::String
   numfluxes::Dict{String, Function}
end

# Upwind flux
function upwind(x,ual,uar,Fl,Fr,Ul,Ur,eq,dir)
   v = eq.velocity(x)
   F = (v > 0) ? Fl : Fr
   return F
end

# Rusanov flux
function rusanov(x,ual,uar,Fl,Fr,Ul,Ur,eq,dir)
   v = eq.velocity(x)
   F = 0.5*(Fl + Fr - abs(v)*(Ur - Ul))
   return F
end

function get_equation(velocity)
   flux(x,u, eq)  = velocity(x) * u
   speed(x,u, eq) = velocity(x)
   name = "1d Linear Advection Equation"
   numfluxes = Dict("upwind"  => upwind,
                    "rusanov" => rusanov)
   return LinAdv1D(flux, speed, velocity, name, numfluxes)
end

end
