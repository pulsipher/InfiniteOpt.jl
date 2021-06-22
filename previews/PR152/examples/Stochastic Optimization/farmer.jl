# # Two-Stage Stochastic Program

# ## Introduction
# First let's consider a standard two-stage stochastic program. Such problems
# consider 1st stage variables ``x \in X \subseteq \mathbb{R}^{n_x}`` which denote
# upfront (here-and-now) decisions made before any realization of the random
# parameters ``\xi \in \mathbb{R}^{n_\xi}`` is observed, and 2nd stage variables
# ``y(\xi) \in \mathbb{R}^{n_y}`` which denote recourse (wait-and-see) decisions
# that are made in response to realizations of ``\xi``. Moreover, the objective
# seeks to optimize 1st stage costs ``f_1(x)`` and second stage costs
# ``f_2(x, y(\xi))`` which are evaluated over the uncertain domain via a risk
# measure ``R_\xi[\cdot]`` (e.g., the expectation ``\mathbb{E}_\xi[\cdot]``).
# Putting this together, we obtain the two-stage stochastic program:
# ```math
# \begin{aligned}
#     &&\min_{x, y(\xi)} &&& f_1(x) + R_\xi[f_2(x, y(\xi))] \\
#     &&\text{s.t.} &&&  g_i(x, y(\xi)) = 0, && i \in I\\
#     &&&&& h_j(x, y(\xi)) \leq 0, && j \in J\\
#     &&&&&  x \in X\\
# \end{aligned}
# ```
# where ``g_i(x, y(\xi)), \ i \in I,`` denote 2nd stage equality constraints,
# ``h_j(x, y(\xi)), \ j \in J,`` are 2nd stage inequality constraints, and ``X``
# denotes the set of feasible 1st stage decisions.

# ## Formulation

# For an example, we consider the classic farmer problem. Here the farmer must
# allocate farmland ``x_c`` for each crop ``c \in C`` with random yields per acre
# ``\xi_c`` such that he minimizes expenses (i.e., maximizes profit) while fulfilling
# contractual demand ``d_c``. If needed he can purchase crops from other farmers
# to satisfy his contracts. He can also sell extra crop yield that exceeds his
# contractual obligations. Thus, here we have 1st stage variables ``x_c`` and
# 2nd stage variables of crops sold ``w_c(\xi)`` and crops purchased ``y_c(\xi)``.
# Putting this together using the expectation ``\mathbb{E}_\xi[\cdot]`` as our
# risk measure we obtain:
# ```math
# \begin{aligned}
#     &&\underset{x, y(\xi), w(\xi)}{\text{min}} &&& \sum_{c \in C} \alpha_c x_c + \mathbb{E}_{\xi}\left[\sum_{c \in C}\beta_c y_c(\xi) - \lambda_c w_c(\xi)\right] \\
#     &&\text{s.t.} &&&  \sum_{c \in C} x_c \leq \bar{x}\\
#     &&&&& \xi_c x_c + y_c(\xi) - w_c(\xi) \geq d_c, && c \in C \\
#     &&&&& 0 \leq x_c \leq \bar{x}, && c \in C \\
#     &&&&& 0 \leq y_c(\xi) \leq \bar{y}_c, && c \in C \\
#     &&&&& 0 \leq w_c(\xi) \leq \bar{w}_c, && c \in C \\
#     &&&&& \xi_c \in \Xi_c, && c \in C
# \end{aligned}
# ```
# where ``\alpha_c`` are production costs, ``\beta_c`` are the purchase prices,
# ``\lambda_c`` are the selling prices, ``\bar{x}`` is the total acreage,
# ``\bar{y}_c`` are purchases limits, ``\bar{w}_c`` are selling limits, and
# ``\Xi_c`` are the underlying distributions.

# ## Problem Setup

# First let's import the necessary packages:
using InfiniteOpt, Distributions, Ipopt

# Next let's specify the problem data:
num_scenarios = 10 # small amount for example
C = 1:3
α = [150, 230, 260] # land cost
β = [238, 210, 0]   # purchasing cost
λ = [170, 150, 36]  # selling price
d = [200, 240, 0]   # contract demand
xbar = 500          # total land
wbar3 = 6000        # no upper bound on the other crops
ybar3 = 0           # no upper bound on the other crops
Ξ = [Uniform(0, 5), Uniform(0, 5), Uniform(10, 30)]; # the distributions

# ## Problem Definition

# Let's start by setting up the infinite model that uses Ipopt as the optimizer 
# that will ultimately be used to solve the transcribed variant:
model = InfiniteModel(Ipopt.Optimizer)
set_optimizer_attribute(model, "print_level", 0);

# Now let's define the infinite parameters using [`@infinite_parameter`](@ref):
@infinite_parameter(model, ξ[c in C] ~ Ξ[c], num_supports = num_scenarios)

# Now let's define all of the decision variables using `@variables`:
@variables(model, begin 
    ## 1st stage variables
    0 <= x[C] <= xbar
    ## 2nd stage variables
    0 <= y[C], Infinite(ξ)
    0 <= w[C], Infinite(ξ)
end)

# Next, the objective is defined using `@objective` and [`𝔼`](@ref):
@objective(model, Min, sum(α[c] * x[c] for c in C) +
                       𝔼(sum(β[c] * y[c] - λ[c] * w[c] for c in C), ξ))

# Finally, all we need to do is define the constraints using `@constraints`:
@constraints(model, begin
    ## capacity constraint
    sum(x[c] for c in C) <= xbar
    ## balances
    [c in C], ξ[c] * x[c] + y[c] - w[c] >= d[c]
    ## crop limits
    w[3] <= wbar3
    y[3] <= ybar3
end)

# ## Problem Solution

# With the model defined, let's optimize and get the results
optimize!(model)
x_opt = value.(x)
profit = -objective_value(model)

println("Land Allocations: ", [round(x_opt[k], digits = 2) for k in keys(x_opt)])
println("Expected Profit: \$", round(profit, digits = 2))

# We did it! 

# ## CVaR Objective

# An interesting modification to the above problem would be to use a ``CVaR`` 
# risk measure instead of an expectation. This also can be readily achieved via 
# `InfiniteOpt`. The ``CVaR`` measure is defined:
# ```math
# CVaR_\epsilon(X) = \underset{t \in \mathbb{R}}{\text{inf}}\left\{t + \frac{1}{1-\epsilon} \mathbb{E}[\text{max}(0, X - t)] \right\}
# ```
# where ``\epsilon`` is the confidence level. Inserting this into the formulation,
# we now obtain:
# ```math
# \begin{aligned}
#     &&\underset{x, y(\xi), w(\xi), t, q(\xi)}{\text{min}} &&& \sum_{c \in C} \alpha_c x_c + t + \frac{1}{1-\epsilon} \mathbb{E}_{\xi}[q(\xi)] \\
#     &&\text{s.t.} &&& \sum_{c \in C} x_c \leq \bar{x}\\
#     &&&&& \xi_c x_c + y_c(\xi) - w_c(\xi) \geq d_c, && c \in C \\
#     &&&&& 0 \leq x_c \leq \bar{x}, && c \in C \\
#     &&&&& 0 \leq y_c(\xi) \leq \bar{y}_c, && c \in C \\
#     &&&&& 0 \leq w_c(\xi) \leq \bar{w}_c, && c \in C \\
#     &&&&& \xi_c \in \Xi_c, && c \in C \\
#     &&&&& q(\xi) \geq \sum_{c \in C}\beta_c y_c(\xi) - \lambda_c w_c(\xi) - t \\
#     &&&&& q(\xi) \geq 0
# \end{aligned}
# ```
# where ``q(\xi)`` is introduced to handle the max operator. Let's update and
# resolve our `InfiniteOpt` model using ``\epsilon = 0.95``:

# Define the additional variables:
@variables(model, begin 
    t
    q >= 0, Infinite(ξ)
end)

# Redefine the objective:
@objective(model, Min, sum(α[c] * x[c] for c in C) + t + 1 / (1 - 0.95) * 𝔼(q, ξ))

# Add the max constraint:
@constraint(model, q >= sum(β[c] * y[c] - λ[c] * w[c] for c in C) - t)

# Optimize and get the results
optimize!(model)
x_opt = value.(x)
y_opt = value.(y)
w_opt = value.(w)
profit = -sum(α[c] * x_opt[c] for c in C) - 1 / num_scenarios *
            sum(β[c] * y_opt[c][k] - λ[c] * w_opt[c][k] for c in C, k in 1:num_scenarios)

println("Land Allocations: ", [round(x_opt[k], digits = 2) for k in keys(x_opt)])
println("Expected Profit: \$", round(profit, digits = 2))

# That's it!

# ### Maintenance Tests
# These are here to ensure this example stays up to date. 
using Test
@test termination_status(model) == MOI.LOCALLY_SOLVED
@test x_opt isa JuMPC.DenseAxisArray{<:Real}
@test profit isa Real
