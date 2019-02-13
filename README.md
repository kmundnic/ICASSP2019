# ICASSP2019
Code accompanying the paper "Bluetooth based indoor localization using Triplet Embeddings"

This code runs in Julia v0.6 (tested on 0.6.4). The following packages are needed (run the following code to install):

```julia
Pkg.add("CSV")
Pkg.add("SCS")
Pkg.add("JuMP")
Pkg.add("JSON")
Pkg.add("Plots"); pyplot() # Needs PyCall and and PyPlot
Pkg.add("DataFrames")
```

To run, open a Julia REPL in the same directory and run:

```julia
include("experiments.jl")
```