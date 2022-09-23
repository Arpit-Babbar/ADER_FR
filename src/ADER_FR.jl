module ADER_FR

src_dir = @__DIR__

include("$src_dir/Basis.jl")
include("$src_dir/Grid.jl")
include("$src_dir/EqLinAdv1D.jl")
include("$src_dir/EqBurg1D.jl")
include("$src_dir/InitialValues.jl")
include("$src_dir/FR1D.jl")

end
