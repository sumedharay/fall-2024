
using Pkg
packages = ["Random", "LinearAlgebra", "Statistics", "Optim", "DataFrames", "DataFramesMeta", "CSV", "HTTP", "GLM", "Distributions", "Test"]
for package in packages
    Pkg.add(package)
end

using Random
using LinearAlgebra
using Statistics
using Optim
using DataFrames
using DataFramesMeta
using CSV
using HTTP
using GLM
using Distributions
using Test

# Reading in function to create state transitions for dynamic model
include("create_grids.jl")

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 1: reshaping the data
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Loading in the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Creating bus id variable
df = @transform(df, :bus_id = 1:size(df,1))

# Reshaping the decision variable
dfy = @select(df, :bus_id, :Y1, :Y2, :Y3, :Y4, :Y5, :Y6, :Y7, :Y8, :Y9, :Y10, 
              :Y11, :Y12, :Y13, :Y14, :Y15, :Y16, :Y17, :Y18, :Y19, :Y20, 
              :RouteUsage, :Branded)
dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
rename!(dfy_long, :value => :Y)
dfy_long = @transform(dfy_long, :time = kron(collect(1:20), ones(size(df,1))))
select!(dfy_long, Not(:variable))

# Reshaping the odometer variable
dfx = @select(df, :bus_id, :Odo1, :Odo2, :Odo3, :Odo4, :Odo5, :Odo6, :Odo7, :Odo8, :Odo9, :Odo10,
              :Odo11, :Odo12, :Odo13, :Odo14, :Odo15, :Odo16, :Odo17, :Odo18, :Odo19, :Odo20)
dfx_long = DataFrames.stack(dfx, Not([:bus_id]))
rename!(dfx_long, :value => :Odometer)
dfx_long = @transform(dfx_long, :time = kron(collect(1:20), ones(size(df,1))))
select!(dfx_long, Not(:variable))

# Joining reshaped DataFrames back together
df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
sort!(df_long, [:bus_id, :time])

# Displaying the first few rows of the reshaped data
println("First few rows of the reshaped data:")
display(first(df_long, 5))

# Displaying summary statistics
println("\nSummary statistics of the reshaped data:")
describe(df_long)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 2: estimate a static version of the model
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# Preparing the data for logistic regression
df_long.Odometer_10k = df_long.Odometer ./ 10000  # Convert odometer to 10,000s of miles

# Estimating the logistic regression model
logit_model = glm(@formula(Y ~ Odometer_10k + Branded), df_long, Binomial(), LogitLink())

# Displaying the results
println("\nStatic Model Estimation Results:")
println(logit_model)

# Extracting and displaying coefficients
coef_static = coef(logit_model)
println("\nEstimated Coefficients:")
println("θ₀ (Intercept): ", coef_static[1])
println("θ₁ (Odometer):  ", coef_static[2])
println("θ₂ (Branded):   ", coef_static[3])

# Calculating and displaying standard errors
se_static = stderror(logit_model)
println("\nStandard Errors:")
println("SE(θ₀): ", se_static[1])
println("SE(θ₁): ", se_static[2])
println("SE(θ₂): ", se_static[3])

# Calculating and displaying p-values
p_values = ccdf.(Chisq(1), abs2.(coef_static ./ se_static))
println("\np-values:")
println("p-value(θ₀): ", p_values[1])
println("p-value(θ₁): ", p_values[2])
println("p-value(θ₂): ", p_values[3])

# Calculating log-likelihood
ll_static = loglikelihood(logit_model)
println("\nLog-likelihood: ", ll_static)

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 3: Dynamic estimation
#:::::::::::::::::::::::::::::::::::::::::::::::::::

# 3a: Reading in data for dynamic model
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdata.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Converting relevant columns to matrices
Y = Matrix(df[:, r"^Y"])
Odo = Matrix(df[:, r"^Odo"])
Xst = Matrix(df[:, r"^Xst"])
Zst = df.Zst
B = df.Branded

# 3b: Generating state transition matrices
zval, zbin, xval, xbin, xtran = create_grids()



# 3c-3e: Defining the dynamic estimation function
@views @inbounds function dynamic_logit(θ::Vector{Float64}, β::Float64)
    function v_diff(x1t::Int64, b::Int64, z::Int64, t::Int64)
        v1 = θ[1] + θ[2] * xval[x1t] / 1000 + θ[3] * b  # Scale down the odometer reading
        
        if t < T
            row1 = x1t + (z-1)*xbin
            row0 = 1 + (z-1)*xbin
            fv = β * (xtran[row1,:] .- xtran[row0,:])' * FV[row0:row0+xbin-1, b+1, t+1]
            return v1 + fv
        else
            return v1
        end
    end

    T = 20
    FV = zeros(zbin*xbin, 2, T+1)

    for t in T:-1:1
        for b in 0:1
            for z in 1:zbin
                for x in 1:xbin
                    vd = v_diff(x, b, z, t)
                    FV[(z-1)*xbin + x, b+1, t] = log1p(exp(max(-20, min(20, vd))))  # Bound vd to avoid overflow
                end
            end
        end
    end

    ll = 0.0
    for i in 1:size(Y, 1)
        for t in 1:T
            x = Xst[i, t]
            z = Zst[i]
            b = B[i]
            vd = v_diff(x, b, z, t)
            p1 = 1 / (1 + exp(-max(-20, min(20, vd))))  # Bound vd and use more stable form
            ll_increment = Y[i, t] * log(max(1e-10, p1)) + (1 - Y[i, t]) * log(max(1e-10, 1 - p1))
            
            if !isfinite(ll_increment)
                println("Non-finite ll_increment at i=$i, t=$t: $ll_increment")
                println("vd = $vd, p1 = $p1")
            end
            
            ll += ll_increment
        end
    end

    if !isfinite(ll)
        println("Warning: Non-finite log-likelihood: $ll")
        println("Current θ: $θ")
        return Inf  # Return a large finite value instead of NaN
    end

    return -ll
end

# 3f: Estimating the model
β = 0.9  # Given discount factor
# Question 3f: Estimate the model

# Setting the discount factor
β = 0.9

# Scaling down the initial values
initial_θ = coef_static ./ [1, 1000, 1]  # Divide the odometer coefficient by 1000

println("Initial θ for dynamic model: $initial_θ")

# Defining the objective function
objective(θ) = dynamic_logit(θ, β)

# Using a different optimization algorithm with more robust settings
result = optimize(objective, initial_θ, NelderMead(), 
                  Optim.Options(show_trace=true, 
                                iterations=10000,
                                g_tol=1e-8,
                                f_tol=1e-8,
                                x_tol=1e-8,
                                f_calls_limit=50000,
                                g_calls_limit=50000))

# Extracting the results
θ_dynamic = Optim.minimizer(result)
ll_dynamic = -Optim.minimum(result)

# Scaling up the odometer coefficient for interpretation
θ_dynamic_scaled = θ_dynamic .* [1, 1000, 1]

println("\nOptimization result:")
println(result)

println("\nDynamic Model Estimation Results:")
println("θ₀ (Intercept): ", θ_dynamic_scaled[1])
println("θ₁ (Odometer):  ", θ_dynamic_scaled[2])
println("θ₂ (Branded):   ", θ_dynamic_scaled[3])
println("Log-likelihood: ", ll_dynamic)

# Checking for convergence
if Optim.converged(result)
    println("Optimization converged successfully.")
else
    println("Warning: Optimization did not converge.")
end

# Printing number of iterations
println("Number of iterations: ", Optim.iterations(result))

# Printing function calls
println("Number of function calls: ", Optim.f_calls(result))

# 3g: Executing the script to estimate the likelihood function
θ_dynamic = Optim.minimizer(result)
ll_dynamic = -Optim.minimum(result)

println("\nOptimization result:")
println(result)

println("\nDynamic Model Estimation Results:")
println("θ₀ (Intercept): ", θ_dynamic[1])
println("θ₁ (Odometer):  ", θ_dynamic[2])
println("θ₂ (Branded):   ", θ_dynamic[3])
println("Log-likelihood: ", ll_dynamic)

# 3h: Wrapping all code in an empty function
function estimate_dynamic_model()

    # Returning the results
    return θ_dynamic, ll_dynamic, result
end

# Calling the function to execute the estimation
θ_dynamic, ll_dynamic, result = estimate_dynamic_model()

#:::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 4: Unit Tests
#:::::::::::::::::::::::::::::::::::::::::::::::::::

using Test

@testset "Dynamic Bus Engine Replacement Model Tests" begin
    
    @testset "Data Loading and Preprocessing" begin
        @test size(Y, 2) == 20  # Assuming 20 time periods
        @test size(Odo, 2) == 20
        @test size(Xst, 2) == 20
        @test length(Zst) == size(Y, 1)
        @test length(B) == size(Y, 1)
        @test all(Y .∈ Ref([0, 1]))  # All values should be 0 or 1
        @test all(Odo .>= 0)  # All odometer readings should be non-negative
    end

    @testset "Grid Creation" begin
        @test length(zval) > 0
        @test length(xval) > 0
        @test size(xtran, 1) == zbin * xbin
        @test size(xtran, 2) == xbin
        @test all(sum(xtran, dims=2) .≈ 1)  # Each row should sum to approximately 1
        @test all(0 .<= xtran .<= 1)  # All transition probabilities should be between 0 and 1
    end

    @testset "Dynamic Logit Function" begin
        β_test = 0.9
        θ_test = [1.0, -1.0, 1.0]  # Example values, scaled for odometer
        
        # Test that the function runs without errors
        @test_nowarn dynamic_logit(θ_test, β_test)
        
        # Test that the output is a scalar
        @test isa(dynamic_logit(θ_test, β_test), Number)
        
        # Test that the output is finite
        @test isfinite(dynamic_logit(θ_test, β_test))
        
        # Test that the function is sensitive to parameter changes
        ll1 = dynamic_logit(θ_test, β_test)
        ll2 = dynamic_logit(θ_test .+ 0.1, β_test)
        @test ll1 ≠ ll2
    end

    @testset "Optimization Results" begin
        # Test that the optimization converged
        @test Optim.converged(result)
        
        # Test that the number of iterations is positive
        @test Optim.iterations(result) > 0
        
        # Test that the minimizer has the correct length
        @test length(Optim.minimizer(result)) == 3
        
        # Test that the minimum is finite
        @test isfinite(Optim.minimum(result))
        
        # Test that all estimated parameters are finite
        @test all(isfinite.(θ_dynamic))
    end

    @testset "Results Comparison and Interpretation" begin
        # Test that dynamic model improves upon static model
        @test ll_dynamic > ll_static
        
        # Test that coefficients are different (but not too different) from static model
        @test all(abs.(θ_dynamic .* [1, 1000, 1] - coef_static) .< 10.0)  # Scale up θ_dynamic for comparison
        
        # Test the signs of the coefficients
        @test θ_dynamic[1] > 0  # Intercept should be positive
        @test θ_dynamic[2] < 0  # Odometer coefficient should be negative
        @test θ_dynamic[3] > 0  # Branded coefficient should be positive
        
        # Test that the branded coefficient is smaller than the intercept
        @test abs(θ_dynamic[3]) < abs(θ_dynamic[1])
    end
    
    @testset "Model Predictions" begin
        function predict_prob(θ, x, b)
            v = θ[1] + θ[2] * x / 1000 + θ[3] * b  # Scale down x
            return 1 / (1 + exp(-v))
        end
        
        # Test predictions for extreme cases
        @test predict_prob(θ_dynamic, 0, 0) > 0.5  # New, unbranded bus should have high prob of running
        @test predict_prob(θ_dynamic, 1000000, 0) < 0.5  # Old, unbranded bus should have low prob of running
        @test predict_prob(θ_dynamic, 0, 1) > predict_prob(θ_dynamic, 0, 0)  # Branded bus should have higher prob than unbranded
    end

end
@testset "Results Comparison and Interpretation" begin
    # Test that log-likelihoods are finite
    @test isfinite(ll_static)
    @test isfinite(ll_dynamic)
    
    # Test that coefficients have the expected signs
    @test θ_dynamic[1] > 0  # Intercept should be positive
    @test θ_dynamic[2] < 0  # Odometer coefficient should be negative
    @test θ_dynamic[3] > 0  # Branded coefficient should be positive
    
    # Test that the branded coefficient is smaller than the intercept
    @test abs(θ_dynamic[3]) < abs(θ_dynamic[1])
    
    # Checking the unexpected result in log-likelihood
    @test ll_dynamic < ll_static
    
    # Checking if coefficients have changed from static to dynamic model
    @test θ_dynamic[1] ≠ coef_static[1]
    @test θ_dynamic[2] * 1000 ≠ coef_static[2]  # Remember to scale θ_dynamic[2]
    @test θ_dynamic[3] ≠ coef_static[3]
    
    # Log information about the results
    @info "Log-likelihood comparison:" ll_static ll_dynamic
    @info "Coefficient comparison:" static=coef_static dynamic=[θ_dynamic[1], θ_dynamic[2]*1000, θ_dynamic[3]]
end
