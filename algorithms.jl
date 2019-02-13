using CSV
using SCS
using JuMP
using Plots; pyplot()
using DataFrames

include("TripletEmbeddings.jl/src/Embeddings.jl")
include("TripletEmbeddings.jl/src/utilities.jl")

function g1(x::Int64)
	if x < 136
		return 0
	elseif 136 <= x <= 193
		return x
	else
		return 193
	end
end

function g2(x::Int64)
	if x < 150
		return 0
	elseif 150 <= x <= 193
		return x
	else
		return 193
	end
end

function load_data(file::String)
	data = dropmissing(CSV.read(file, types=Dict(5 => String), allowmissing=:none))
	# data = data[limits[1] .>= data[:rssi] .>= limits[2], :] # Remove all the rows with invalid RRSI values

	return data
end

function load_data(file::String, limits::Array{Int64,2})
	data = dropmissing(CSV.read(file, types=Dict(5 => String), allowmissing=:none))
	data = data[limits[1] .>= data[:rssi] .>= limits[2], :] # Remove all the rows with invalid RRSI values

	return data
end

function load_owl_positions(file::String)
	return convert(Array{Float64,2}, CSV.read(file))
end

function label(X::Array{Float64,2}, rssis::Dict{Int64,Int64})::Array{Int64,2}
	# probability represents the probability of swapping the order of a
	# random triplet

	# We prealocate the possible total amount of triplets. Before returning,
	# we clip the array 'triplets' to the amount of nonzero elements.
	n = size(X, 1) + 1
	triplets = zeros(Int64, n*binomial(n-1, 2), 3)
	counter = 0

	D = distances(X, size(X,1))::Array{Float64,2}

	for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k && i < n && k < n
			if D[i,j] < D[i,k]
				counter +=1
        		@inbounds triplets[counter,:] = [i, j, k]
            elseif D[i,j] > D[i,k]
				counter += 1
				@inbounds triplets[counter,:] = [i, k, j]
			end
	    elseif i != j && i != k && i == n
	    	# Our anchor is the Jelly
	    	if haskey(rssis, j) && !haskey(rssis, k)
				counter +=1
	    		@inbounds triplets[counter,:] = [i, j, k]
	    	elseif !haskey(rssis, j) && haskey(rssis, k)
	    		counter +=1
	    		@inbounds triplets[counter,:] = [i, k, j]
	    	end
		end
	end

    return triplets[1:counter,:]
end

function label(X::Array{Float64,2}, rssis::Array{Int64,1}; g::Function = (x->x))::Array{Int64,2}
	# We prealocate the possible total amount of triplets. Before returning,
	# we clip the array 'triplets' to the amount of nonzero elements.
	n = size(X, 1) + 1
	triplets = zeros(Int64, n * binomial(n-1, 2), 3)
	counter = 0

	D = distances(X, size(X,1))::Array{Float64,2}

	for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k && i < n && k < n
			if D[i,j] < D[i,k]
				counter +=1
        		@inbounds triplets[counter,:] = [i, j, k]
            elseif D[i,j] > D[i,k]
				counter += 1
				@inbounds triplets[counter,:] = [i, k, j]
			end
	    elseif i != j && i != k && i == n
	    	# Our anchor is the Jelly
	    	if g(rssis[j]) > g(rssis[k])
				counter +=1
	    		@inbounds triplets[counter,:] = [i, j, k]
	    	elseif g(rssis[k]) > g(rssis[j])
	    		counter +=1
	    		@inbounds triplets[counter,:] = [i, k, j]
	    	end
		end
	end

    return triplets[1:counter,:]
end

function label2(X::Array{Float64,2}, rssis::Array{Int64,1}; g::Function = (x->x))::Array{Int64,2}
	# probability represents the probability of swapping the order of a
	# random triplet

	# We prealocate the possible total amount of triplets. Before returning,
	# we clip the array 'triplets' to the amount of nonzero elements.
	n = size(X, 1) + 1
	triplets = zeros(Int64, n * binomial(n-1, 2), 3)
	counter = 0

	D = distances(X, size(X,1))::Array{Float64,2}

	for k = 1:n, j = 1:k-1, i = 1:n
        if i != j && i != k && i < n && k < n
			if D[i,j] < D[i,k]
				counter +=1
        		@inbounds triplets[counter,:] = [i, j, k]
            elseif D[i,j] > D[i,k]
				counter += 1
				@inbounds triplets[counter,:] = [i, k, j]
			end
	    elseif i != j && i != k && i == n

			if max(g(rssis[j]), g(rssis[k])) >= 150
	    		# Our anchor is the Jelly
		    	if g(rssis[j]) > g(rssis[k])
					counter +=1
		    		@inbounds triplets[counter,:] = [i, j, k]
		    	elseif g(rssis[k]) > g(rssis[j])
		    		counter +=1
		    		@inbounds triplets[counter,:] = [i, k, j]
		    	end
		    end
		end
	end

    return triplets[1:counter,:]
end

function distances(X::Array{Float64,2})::Array{Float64,2}
    n = maximum(size(X))

    D = zeros(Float64, n, n)

    for j = 1:n, i = j:n
        @inbounds D[i,j] = norm(X[i,:] - X[j,:])^2
    end

    return D + D'
end

function centering_matrix(n::Int64)
	return eye(n) - ones(n,1)*ones(1,n)/n
end

function classicalMDS(D::Array{Float64,2}, d::Int64)
	n = size(D,1)
	@assert issymmetric(D)

	J = eye(n) - ones(n,1)*ones(1,n)/n

	λ, V = eig(- J * D * J/2)
	λ[d+1:end] = 0

	return diagm(sqrt.(real.(λ))) * V'
end

function apply_procrustes(X_hat::Array{Float64,2}, A::Array{Float64,2}, B::Array{Float64,2})
	X0_hat = X_hat[1:end-1,:] - repmat(mean(X_hat[1:end-1,:],1), size(X_hat[1:end-1,:],1), 1)
	ssqX_hat = sum(X0_hat.^2)
	normX_hat = sqrt(ssqX_hat)

	return ((X_hat - repmat(mean(X_hat[1:end-1,:],1), size(X_hat,1), 1)) / normX_hat * A  + repmat(B, size(X_hat,1), 1))[end,:]
end

function get_rssis_from_ping(owls::Array{String,1}, ping::DataFrame)
	rssi = zeros(Int64, size(owls))

	for i in 1:length(owls)
	    if owls[i] in ping[:receiverDirectory]
	        rssi[i] = ping[owls[i] .==  ping[:receiverDirectory], :rssi][1]
	    else
	        rssi[i] = 0
	    end
	end

	return rssi
end

function in_convex_hull(X::Array{Float64,2}, E::Array{Float64,1})
	# If the estimate lies in the convex hull, then the optimization
	# problem is feasible.

	# We define the points along the columns columns
	x = E'
	Z = X'

	m = Model(solver=SCSSolver(verbose=false))
	@variable(m, 0 <= lambda[j=1:size(Z,2)] <= 1)

	@constraint(m, inhull[i=1:length(x)], x[i] == sum(Z[i, j] * lambda[j] for j = 1:size(Z, 2)))
	@constraint(m, sum(lambda) == 1)
	status = solve(m)

	return status == :Optimal

end

function plot_locations(
	X::Array{Float64,2}, 
	Y::Array{Float64,2}, 
	Z::Array{Float64,2},
	i::Int64,
	owl::String,
	index::Int64,
	rssi::Array{Int64,1},
	E::Array{Float64,1};
	show_plots=true)

	# We cannot pass scalars to scatter(), so we create an array of arrays
	# for both.
	# I could create a recipe using Plots, but I haven't looked at it yet
	true_sender_x = Array{Float64,1}(1)
	true_sender_x[1] = X[index[1],1]
	true_sender_y = Array{Float64,1}(1)
	true_sender_y[1] = X[index[1],2]

	estimated_sender_x = Array{Float64,1}(1)
	estimated_sender_x[1] = E[1]
	estimated_sender_y = Array{Float64,1}(1)
	estimated_sender_y[1] = E[2]

	scatter([Y; X[index,:]'][:,1], [Y; X[index,:]'][:,2], 
		show=show_plots,
		reuse=false,
		markershape = :circle,
		label="True receiver positions", 
		title=string(split(owl, ':')[end], " i = $(i), error = $(norm(X[index,:] - E))"),
		xlabel="Position [m]",
		ylabel="Position [m]")

	scatter!(Z[:,1], Z[:,2],
		show=show_plots,
		markershape = :circle,
		label = "Estimated receiver positions")

	scatter!(true_sender_x, true_sender_y,
		show=show_plots,
		markershape = :circle,
	    label="True sender position")

	scatter!(estimated_sender_x, estimated_sender_y,
	    show=show_plots,
	    markershape = :circle,
	    label="Estimated sender position")

	for j in 1:length(rssi)
		if rssi[j] != 0
			p1 = [E[1]; Y[j,1]]
			p2 = [E[2]; Y[j,2]]
			plot!(p1, p2,
				show=show_plots,
				linewidth=0.5,
				color=:black,
				label="",
				# label=string(split(owls[j], ':')[end], " RSSI = $(rssi[j])"),
				style=:dash,
				annotations=(mean(p1), mean(p2), text("$(rssi[j])",8)))
		end
	end

end

function plot_locations(
	X::Array{Float64,2}, 
	Y::Array{Float64,2}, 
	Z::Array{Float64,2},
	owl::String,
	index::Int64,
	rssi::Array{Int64,1},
	E::Array{Float64,1};
	show_plots=true)

	# We cannot pass scalars to scatter(), so we create an array of arrays
	# for both.
	# I could create a recipe using Plots, but I haven't looked at it yet
	true_sender_x = Array{Float64,1}(1)
	true_sender_x[1] = X[index[1],1]
	true_sender_y = Array{Float64,1}(1)
	true_sender_y[1] = X[index[1],2]

	estimated_sender_x = Array{Float64,1}(1)
	estimated_sender_x[1] = E[1]
	estimated_sender_y = Array{Float64,1}(1)
	estimated_sender_y[1] = E[2]

	scatter([Y; X[index,:]'][:,1], [Y; X[index,:]'][:,2], 
		show=show_plots, 
		markershape = :circle,
		label="True receiver positions", 
		title=string(split(owl, ':')[end], " error = $(norm(X[index,:] - E))"),
		xlabel="Position [m]",
		ylabel="Position [m]")

	scatter!(Z[:,1], Z[:,2],
		show=show_plots,
		markershape = :circle,
		label = "Estimated receiver positions")

	scatter!(true_sender_x, true_sender_y,
		show=show_plots,
		markershape = :circle,
	    label="True sender position")

	scatter!(estimated_sender_x, estimated_sender_y,
	    show=show_plots,
	    markershape = :circle,
	    label="Estimated sender position")

	for j in 1:length(rssi)
		if rssi[j] != 0
			p1 = [E[1]; Y[j,1]]
			p2 = [E[2]; Y[j,2]]
			plot!(p1, p2,
				show=show_plots,
				linewidth=0.5,
				color=:black,
				label="",
				# label=string(split(owls[j], ':')[end], " RSSI = $(rssi[j])"),
				style=:dash,
				annotations=(mean(p1), mean(p2), text("$(rssi[j])",8)))
		end
	end

end