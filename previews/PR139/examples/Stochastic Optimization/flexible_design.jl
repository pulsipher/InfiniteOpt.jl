# # Power Network Flexibility Design
# Here we consider a stochastic flexibility design problem that is inspired
# from an example problem in Pulsipher JL, Zavala VM. *A scalable stochastic
# programming approach for the design of flexible systems*. Computers & Chemical
# Engineering. 2019 Sep 2;128:69-76. See the paper 
# [here](https://doi.org/10.1016/j.compchemeng.2019.05.033).

# ## Modeling and Solution

using InfiniteOpt, Clp, Distributions

## Set the covariance matrix for the uncertain parameters
θ_nom = [0.; 60.; 10.]
covar = [80. 0 0; 0 80. 0; 0 0 120.]

## Set the dimensions
n_z = 3; n_θ = 3; n_d = 3

## Set the problem parameters
c = ones(n_d) / sqrt(n_d)
c_max = 5
U = 10000
num_samples = 100 # more should be used (kept small for doc generation)

## Initialize the model
m = InfiniteModel(Clp.Optimizer)
set_silent(m)

## Set the uncertainty parameters
@infinite_parameter(m, θ[i = 1:n_θ] ~ MvNormal(θ_nom, covar), 
                    num_supports = num_samples)

## Initialize the variables
@variable(m, 0 <= y <= 1, Infinite(θ))
@variable(m, z[1:n_z], Infinite(θ))
@variable(m, d[1:n_d] >= 0)

## Set objective function
@objective(m, Max, expect(1 - y, θ))

## Set the line capacity constraints
@constraint(m, f1, -z[1] - 35 - d[1] <= y * U)
@constraint(m, f2, z[1] - 35 - d[1] <= y * U)
@constraint(m, f3, -z[2] - 50 - d[2] <= y * U)
@constraint(m, f4, z[1] - 50 - d[2] <= y * U)

## Set the generator capacity constraints
@constraint(m, f5, -z[3] <= y * U)
@constraint(m, f6, z[3] - 100 - d[3] <= y * U)

## Set the node balance constraints
@constraint(m, h1, z[1] - θ[1] == 0)
@constraint(m, h2, -z[1] -z[2] + z[3] - θ[2] == 0)
@constraint(m, h3, z[2] - θ[3] == 0)

## Enforce the minimum SF
@constraint(m, max_cost, sum(c[i] * d[i] for i = 1:n_d) <= c_max)

## Solve and and obtain results
optimize!(m)
if has_values(m)
    opt_y = value(y)
    opt_d = value.(d)
    opt_obj = objective_value(m)
end

## Estimate the value of SF
SF = 1 - sum(opt_y .>= 1e-8) / num_samples

## Print the results
print("------------------RESULTS------------------\n")
print("Optimal Objective:     ", opt_obj, "\n")
print("Optimal Cost:          ", sum(c[i] * opt_d[i] for i = 1:n_d), "\n")
print("Maximum Cost:          ", c_max, "\n")
print("Predicted SF:          ", 100 * SF, "%\n")
print("Optimal Design Values: ", opt_d, "\n\n")

# ### Maintenance Tests
# These are here to ensure this example stays up to date. 
using Test
@test termination_status(m) == MOI.OPTIMAL
@test opt_y isa Vector{<:Real}
