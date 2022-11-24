
"""
Probablistic Model
"""

cd(@__DIR__)
using Turing 
using StatsPlots
using Optim 


@model function gdemo(x,y)
    s² ~ InverseGamma(2,3)
    m ~ Normal(0, sqrt(s²))
    x ~ Normal(m, sqrt(s²))
    y ~ Normal(m, sqrt(s²))
end

@time chn = sample(gdemo(1.5, 2), HMC(0.1, 5), MCMCThreads(),10000, 1)

@time chn = sample(gdemo(1.5, 2), HMC(0.1, 5),10000)

chn = sample(gdemo(1.5, 2), Prior(),10000)

describe(chn)

p = plot

mean(chn[:s²])