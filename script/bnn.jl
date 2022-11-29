"""
Bayesian NN tutorial from
https://turing.ml/dev/tutorials/03-bayesian-neural-network/
"""

using Pkg
Pkg.activate("bnn")


using Turing
using Flux
using Plots
using Random
using ReverseDiff

Turing.setadbackend(:reversediff)

"""
Data generation
"""

# Number of points to generate.
N = 80
M = round(Int, N / 4)
Random.seed!(1234)

# Generate artificial data.
x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
xt1s = Array([[x1s[i] + 0.5; x2s[i] + 0.5] for i in 1:M])
x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
append!(xt1s, Array([[x1s[i] - 5; x2s[i] - 5] for i in 1:M]))

x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
xt0s = Array([[x1s[i] + 0.5; x2s[i] - 5] for i in 1:M])
x1s = rand(M) * 4.5;
x2s = rand(M) * 4.5;
append!(xt0s, Array([[x1s[i] - 5; x2s[i] + 0.5] for i in 1:M]))

# Store all the data for later.
xs = [xt1s; xt0s]
ts = [ones(2 * M); zeros(2 * M)]

# Plot data points.
function plot_data()
    x1 = map(e -> e[1], xt1s)
    y1 = map(e -> e[2], xt1s)
    x2 = map(e -> e[1], xt0s)
    y2 = map(e -> e[2], xt0s)

    Plots.scatter(x1, y1; color="red", clim=(0, 1))
    return Plots.scatter!(x2, y2; color="blue", clim=(0, 1))
end

plot_data()




"""
Bayesian Neural Network
"""

nn_initial = Chain(Dense(2,3,tanh), Dense(3,2, tanh), Dense(2,1, σ))

parameters_initial, reconstruct = Flux.destructure(nn_initial)


"""
Probabilistic Model
"""

alpha = 0.09
sig = sqrt(1.0/alpha)
@model function bayes_nn(xs, ts, nparameters, reconstruct)
    # Wights and bias
    parameters ~ MvNormal(zeros(nparameters), sig .* ones(nparameters))

    # Construct NN
    nn = reconstruct(parameters)

    # Forward NN to make prediction
    preds = nn(xs)

    # Observe
    for i in 1:length(ts)
        ts[i] ~ Bernoulli(preds[i])
    end
end

# Perferm inference
N = 5000 
ch = sample(
    bayes_nn(hcat(xs...), ts, length(parameters_initial), reconstruct), 
    HMC(0.05, 4),
    N
)


"""
Validation
"""

theta = MCMCChains.group(ch, :parameters).value


# A helper to create NN from weights `theta` and run it through data `x`
nn_forward(x, theta) = reconstruct(theta)(x)

# Plot the data we have.
plot_data()

# Find the index that provided the highest log posterior in the chain.
_, i = findmax(ch[:lp])

# Extract the max row value from i.
i = i.I[1]

# Plot the posterior distribution with a contour plot
x1_range = collect(range(-6; stop=6, length=25))
x2_range = collect(range(-6; stop=6, length=25))
Z = [nn_forward([x1, x2], theta[i, :])[1] for x1 in x1_range, x2 in x2_range]
contour!(x1_range, x2_range, Z)