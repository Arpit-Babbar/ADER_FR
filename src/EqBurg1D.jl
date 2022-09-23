module EqBurg1D

# No parameters yet
struct Burg1D
   flux::Function
   speed::Function
   name::String
   numfluxes::Dict{String, Function}
end

flux(x, u, eq)  = 0.5 * u^2
speed(x, u, eq) = u

# Rusanov flux
function rusanov(x,ual,uar,Fl,Fr,Ul,Ur,eq,dir)
   # lam = max |f'(u)| for u b/w ual and uar
   laml, lamr = speed(x, ual, eq), speed(x, uar, eq)
   lam = max(abs(laml), abs(lamr))
   F = 0.5*(Fl + Fr - lam*(Ur - Ul))
   return F
end

function roe(x,ual,uar,Fl,Fr,Ul,Ur,eq,dir)
   if abs(ual - uar) < 1e-10
      a = speed(x, ual, eq)
   else
      fl, fr = flux(x, ual, eq), flux(x, uar, eq)
      a = (fl - fr) / (ual - uar)
   end
   F = 0.5*(Fl + Fr - abs(a)*(Ur - Ul))
   return F
end

function get_equation()
   name = "1d Burger's equation"
   numfluxes = Dict("rusanov" => rusanov,
                    "roe"     => roe)
   Burg1D(flux, speed, name, numfluxes)
end

end
