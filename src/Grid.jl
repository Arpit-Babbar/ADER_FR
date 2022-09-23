module Grid

using Printf
using DoubleFloats
using ArbNumerics

struct CartesianGrid1D{RealT}
   domain::Vector{RealT}   # xmin,xmax
   size::Int64               # nx, ny
   xc::Array{RealT,1}      # x coord of cell center
   xf::Array{RealT,1}      # x coord of faces
   dx::Array{RealT,1}      # cell size along x
end

struct CartesianGrid2D{RealT}
   domain::Vector{RealT}   # xmin,xmax,ymin,ymax
   size::Vector{Int64}       # nx, ny
   xc::Array{RealT,1}      # x coord of cell center
   yc::Array{RealT,1}      # y coord of cell center
   xf::Array{RealT,1}      # x coord of faces
   yf::Array{RealT,1}      # y coord of faces
   dx::Array{RealT,1}      # cell size along x
   dy::Array{RealT,1}      # cell size along y
end

# 1D/2D Uniform Cartesian grid
function make_grid(problem, param)
   domain = problem["domain"]
   size = param["grid_size"]
   if length(domain)==2 && length(size)==1
      println("Making 1D uniform Cartesian grid")
      xmin, xmax = domain
      nx = size
      dx1 = (xmax - xmin)/nx
      xc = [LinRange(xmin+0.5*dx1, xmax-0.5*dx1, nx)...]
      @printf("   Grid size = %d \n", nx)
      @printf("   xmin,xmax = %e, %e\n", xmin, xmax)
      @printf("   dx        = %e\n", dx1)
      dx = dx1 .* ones(nx)
      xf = [LinRange(xmin, xmax, nx+1)...]
      return CartesianGrid1D(domain,size,xc,xf,dx)
   elseif length(domain)==4 && length(size)==2
      println("Type of domain = ",typeof(domain))
      size = param["grid_size"]
      println("Type of size = ",typeof(size))
      println("Making 2D uniform Cartesian grid")
      xmin,xmax,ymin,ymax = domain
      nx,ny = size
      dx1 = (xmax - xmin)/nx
      dy1 = (ymax - ymin)/ny
      xc = LinRange(xmin+0.5*dx1, xmax-0.5*dx1, nx)
      yc = LinRange(ymin+0.5*dy1, ymax-0.5*dy1, ny)
      @printf("   Grid size = %d x %d\n", nx, ny)
      @printf("   xmin,xmax = %e, %e\n", xmin, xmax)
      @printf("   ymin,ymax = %e, %e\n", ymin, ymax)
      @printf("   dx, dy    = %e, %e\n", dx1, dy1)
      dx = dx1 .* ones(nx)
      dy = dy1 .* ones(ny)
      xf = LinRange(xmin, xmax, nx+1)
      yf = LinRange(ymin, ymax, ny+1)
      return CartesianGrid2D(domain,size,xc,yc,xf,yf,dx,dy)
   else
      @assert false "Incorrect Grid"
   end
end

export make_grid

end