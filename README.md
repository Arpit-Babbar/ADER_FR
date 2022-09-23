In the base directory of this repository, start `julia` as

```shell
julia --project=.
```
or by starting plain `julia` REPL and then entering `import Pkg; Pkg.activate(".")`. Install all dependencies (only needed the first time) with
```julia
julia> import Pkg; Pkg.instantiate()
```

Then, to reproduce the results, use

```julia
julia> include("check_equivalence.jl")
```

To plot the data, run the python script
```shell
python plot_equivalence_data.py
```
