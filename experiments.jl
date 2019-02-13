include("algorithms.jl")

using CSV
using JSON

srand(1)
show_plots = false

path = "owl_events_by_unit"
unit = "floor6:north"
file = "2018-05-03.csv"

# Triplet embedding parameters
# We are using tSTE with a big α value,
# which in the limit is the same as STE,
# with σ^2 = 1.
# Numerically, α = 30 makes it a convex problem
# already.
no_dims = 2
λ = 0.0
α = 30.0

limits = [193 136] # RSSI min and max values
data = load_data(joinpath(path, unit, file), limits)
owls = sort(collect(unique(data[:receiverDirectory])))

# We read the positions and convert them to an array
# The values are in meters
X = load_owl_positions("positions/floor6-north-positions.csv")

g = g1 # We use this clipping function defined in algorithms.jl
number_of_receivers = Dict{String, Array{Int64,1}}()
mse = Dict{String, Array{Float64,1}}()
senders = ["floor6:north:A", "floor6:north:B", "floor6:north:C"]
# senders = ["floor6:north:A"]

for owl in senders
	@show(g, owl)

	# We find all the events where a particular owl is a sender
	pings = @view data[data[:deviceDirectory] .== owl, :]
	times = unique(pings[:timeStamp])

	# Find the index of the sender owl in the positions array
	# and the owls array (they have the same order)
	index = find(x -> x == owl, owls)

	# Remove the owl that will be the sender
	Y = X[find(x -> x != owl, owls), :]

	# Remove the same owl from the list
	receivers = filter(x -> x != owl, owls)

	number_of_receivers[owl] = zeros(Int64, size(times))
	mse[owl] = zeros(Float64, size(times))

	# We loop over every time, to capture every event
	counter = 0
	for timestamp in times

		counter += 1

		ping = pings[pings[:timeStamp] .== timestamp, :]
		number_of_receivers[owl][counter] = size(ping,1)

		# We need at least two receivers to locate the sender
		if number_of_receivers[owl][counter] > 1

			# We get the ordered list of RSSIs
			rssi = get_rssis_from_ping(receivers, ping)

			# We label the triplets from locations and RSSI info
			triplets = label(Y, rssi, g=g)

			if maximum(triplets) == size(X,1)
				# Compute the embedding
				# The last row of X_hat has the information related to the sender
				# We inialize the embedding so that we reduce the chance of calculating
				# a reflection of the embedding we're looking for
				X0 = [Y + 1 * randn(size(Y)); [mean(Y[:,1]) mean(Y[:,2])]]
				
				X_hat, _ = Embeddings.tSTE(triplets, no_dims, λ, α, 
					X0 = X0,
					max_iter = 1000, 
					project = false, 
					debug = false, 
					verbose = false)

				# Fit the estimated positions of the receivers to the true positions
				# We make sure that we compute the transformation only with the 
				# receivers and not the sender
				Z, _, A, B = procrustes(Y, X_hat[1:end-1,:])

				# Apply the transformation to the sender, so that it is in the same embedding
				# as the true positions of the receivers
				# This step has a loss that is independent of the triplet embedding approach
				E = apply_procrustes(X_hat, A, B)

				# Compute the estimation error
				# X has the real position, E has the estimated position
				mse[owl][counter] = norm(X[index,:] - E')

				if in_convex_hull(Y, E)
					mse[owl][counter] = norm(X[index,:] - E')
				else
					mse[owl][counter] = Inf
				end
			end
		end
	end

end

open(joinpath("results/", string("$(g)", "_mse.json")), "w") do io
	JSON.print(io, mse)
end

open(joinpath("results/", string("$(g)", "_receivers.json")), "w") do io
	JSON.print(io, number_of_receivers)
end

if show_plots
	plot_results(X, Y, Z, owl, index, rssi, E, show_plots=true)
end