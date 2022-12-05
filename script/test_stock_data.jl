"""
Traing Bayesian neural network with stock data
"""

cd(@__DIR__)
using Pkg
Pkg.activate("bnn")
using Flux
using Plots
using ReverseDiff
using CSV
using DataFrames
using DelimitedFiles


"""
Data
"""

stock_n = "600118.SS_1"
data_pt = "../data/Bayesianneuralnet_stockmarket/code/datasets" 
tnpath = joinpath(data_pt , "$(stock_n)_test.txt")
ttpath = joinpath(data_pt , "$(stock_n)_train.txt")

train = hcat(readdlm(tnpath))
test = hcat(readdlm(ttpath))

plot(train[:, 1])
plot(test)

