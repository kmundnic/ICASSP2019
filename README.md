# ICASSP2019
Code accompanying the paper "Bluetooth based indoor localization using Triplet Embeddings"

This code runs in Julia v0.6 (tested on 0.6.4). The following packages are needed (run the following code to install):

```julia
Pkg.add("CSV")
Pkg.add("SCS")
Pkg.add("JuMP")
Pkg.add("JSON")
Pkg.add("Plots") # Will need a backend
Pkg.add("PyCall")
Pkg.add("PyPlot")
Pkg.add("DataFrames")
```

## Running the experiments
### Enabling multithreading
TripletEmbeddings.jl implements tSTE using threads to compute the gradient. To allow Julia to use more than one thread, run in the terminal `export JULIA_NUM_THREADS=8` before starting the Julia REPL.

### Experiments
To run, open a Julia REPL in the same directory and run:

```julia
include("experiments.jl")
```