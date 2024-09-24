using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Random
using Base.Threads

# Question 1: Multinomial logit

# Loading the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS4-mixture/nlsw88t.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Preparing the data
X = Matrix([df.age df.white df.collgrad])
Z = Matrix(hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8))
y = df.occ_code

# Defining the number of choices and covariates
J = 8  # number of choices
K = size(X, 2)  # number of individual-specific covariates
L = 1  # number of alternative-specific covariates

# Initial values from PS3
β_init = [0.05570760821873361 0.04500069760812089 0.09264599987673823 0.02390335345913949 0.036087237758010406 0.08531087899609793 0.08662012953143235;
          0.08342800912507368 0.7365784224226264 -0.08417566822303942 0.7230680315358062 -0.6437626443990146 -1.1714282628621513 -0.797875842371163;
         -2.3448861127173504 -3.153242881656483 -4.27327816625058 -3.7493920045282323 -4.2796830878646634 -6.6786762695957 -4.969129759355472]
γ_init = -0.09419373857636062

# Function to compute choice probabilities
function compute_probabilities(β, γ, X, Z)
    N = size(X, 1)
    probs = zeros(eltype(β), N, J)
    
    Threads.@threads for i in 1:N
        denom = one(eltype(β))
        for k in 1:J-1
            exponent = sum(X[i,m] * β[m,k] for m in 1:K) + γ * (Z[i,k] - Z[i,J])
            denom += exp(exponent)
        end
        for j in 1:J-1
            exponent = sum(X[i,m] * β[m,j] for m in 1:K) + γ * (Z[i,j] - Z[i,J])
            probs[i,j] = exp(exponent) / denom
        end
        probs[i,J] = 1 / denom
    end
    return probs
end

# Log-likelihood function
function log_likelihood(θ)
    β = reshape(θ[1:K*(J-1)], K, J-1)
    γ = θ[end]
    probs = compute_probabilities(β, γ, X, Z)
    ll = sum(log(probs[i, y[i]]) for i in 1:size(X,1))
    return -ll  # Optim minimizes, so we return negative log-likelihood
end

# Initial values
θ_init = vcat(vec(β_init), γ_init)

# Optimizing using automatic differentiation
result = optimize(log_likelihood, θ_init, LBFGS(); autodiff = :forward)

# Extracting the results
θ_hat = Optim.minimizer(result)
β_hat = reshape(θ_hat[1:K*(J-1)], K, J-1)
γ_hat = θ_hat[end]

# Computing Fisher Information Matrix
function score(θ)
    β = reshape(θ[1:K*(J-1)], K, J-1)
    γ = θ[end]
    probs = compute_probabilities(β, γ, X, Z)
    N = size(X, 1)
    score = zeros(length(θ))
    for i in 1:N
        for j in 1:J-1
            if y[i] == j
                score[((j-1)*K+1):(j*K)] .+= X[i,:] .- probs[i,j] .* X[i,:]
                score[end] += Z[i,j] - Z[i,J] - probs[i,j] * (Z[i,j] - Z[i,J])
            else
                score[((j-1)*K+1):(j*K)] .-= probs[i,j] .* X[i,:]
                score[end] -= probs[i,j] * (Z[i,j] - Z[i,J])
            end
        end
        if y[i] == J
            score[end] -= Z[i,J] - sum(probs[i,j] * Z[i,j] for j in 1:J-1)
        end
    end
    return score
end

FIM = zeros(length(θ_hat), length(θ_hat))
for i in 1:size(X, 1)
    s = score(θ_hat)
    FIM .+= s * s'
end
FIM ./= size(X, 1)

# Computing standard errors
se = sqrt.(diag(inv(FIM)))

# Printing results
println("Estimated β:")
display(β_hat)
println("\nEstimated γ: ", γ_hat)
println("\nStandard Errors for β:")
display(reshape(se[1:K*(J-1)], K, J-1))
println("\nStandard Error for γ: ", se[end])

# Printing convergence information
println("\nConvergence: ", Optim.converged(result))
println("Number of iterations: ", Optim.iterations(result))
println("Minimum function value: ", Optim.minimum(result))

# Printing data information
println("\nData Information:")
println("Number of observations: ", size(X, 1))
println("Number of choices: ", J)
println("Number of individual-specific covariates: ", K)
println("Number of alternative-specific covariates: ", size(Z, 2))

# Question 2 

# Extracting the estimated γ
γ_hat = θ_hat[end]

# Printing the estimated γ and its standard error
println("Estimated γ: ", γ_hat)
println("Standard Error of γ: ", se[end])

# Interpretation
println("\nInterpretation:")
println("γ represents the effect of log wage on the odds of choosing each occupation relative to the base category.")
println("A positive γ indicates that higher wages increase the odds of choosing each occupation.")

# Comparing with Problem Set 3 
γ_ps3 = -0.0942
println("\nComparison to Problem Set 3:")
println("γ from PS3: ", γ_ps3)
println("γ from current model: ", γ_hat)

if abs(γ_hat) > abs(γ_ps3)
    println("The current γ estimate has a larger magnitude, suggesting a stronger wage effect.")
else
    println("The current γ estimate has a smaller magnitude, suggesting a weaker wage effect.")
end

println("The current estimate may make more sense than the one from PS03")


# Question 3: Mixed logit with Gauss-Legendre quadrature

function mixed_logit_q3(X, Z, y, β_init, μγ_init, σγ_init)
    # Including the lgwt function (assume it's in the working directory)
    include("lgwt.jl")

    # Generating quadrature points and weights
    num_quad_points = 7
    nodes, weights = lgwt(num_quad_points, -4, 4)

    function compute_mixed_probabilities(β, μγ, σγ, X, Z)
        N, J = size(Z)
        probs = zeros(N)
        for i in 1:N
            prob_i = 0.0
            for (node, weight) in zip(nodes, weights)
                γ = μγ + σγ * node
                denom = 1.0
                num = (y[i] == J) ? 1.0 : 0.0
                for j in 1:J-1
                    v = exp(X[i,:]'β[:,j] + γ * (Z[i,j] - Z[i,J]))
                    denom += v
                    if y[i] == j
                        num = v
                    end
                end
                prob_i += weight * (num / denom)
            end
            probs[i] = prob_i
        end
        return probs
    end

    function log_likelihood(θ)
        K, J = size(β_init)
        β = reshape(θ[1:K*(J-1)], K, J-1)
        μγ, σγ = θ[end-1:end]
        probs = compute_mixed_probabilities(β, μγ, σγ, X, Z)
        return -sum(log.(probs))
    end

    θ_init = [vec(β_init); μγ_init; σγ_init]
    result = optimize(log_likelihood, θ_init, LBFGS(); autodiff = :forward)
    
    return result
end

# Question 4: Modified likelihood function

function mixed_logit_q4(X, Z, y, β_init, μγ_init, σγ_init)
    # Assuming lgwt function is included as in Question 3
    num_quad_points = 7
    nodes, weights = lgwt(num_quad_points, -4, 4)

    function log_likelihood(θ)
        K, J = size(β_init)
        β = reshape(θ[1:K*(J-1)], K, J-1)
        μγ, σγ = θ[end-1:end]
        N = size(X, 1)
        ll = 0.0
        for i in 1:N
            prob_i = 0.0
            for (node, weight) in zip(nodes, weights)
                γ = μγ + σγ * node
                denom = 1.0
                num = (y[i] == J) ? 1.0 : 0.0
                for j in 1:J-1
                    v = exp(X[i,:]'β[:,j] + γ * (Z[i,j] - Z[i,J]))
                    denom += v
                    if y[i] == j
                        num = v
                    end
                end
                prob_i += weight * (num / denom) * pdf(Normal(μγ, σγ), γ)
            end
            ll += log(prob_i)
        end
        return -ll
    end

    θ_init = [vec(β_init); μγ_init; σγ_init]
    result = optimize(log_likelihood, θ_init, LBFGS(); autodiff = :forward)
    
    return result
end

# Question 5: Mixed logit with Monte Carlo integration

function mixed_logit_q5(X, Z, y, β_init, μγ_init, σγ_init)
    function log_likelihood(θ)
        K, J = size(β_init)
        β = reshape(θ[1:K*(J-1)], K, J-1)
        μγ, σγ = θ[end-1:end]
        N = size(X, 1)
        D = 1000  # Number of Monte Carlo draws
        ll = 0.0
        for i in 1:N
            prob_i = 0.0
            for _ in 1:D
                γ = rand(Normal(μγ, σγ))
                denom = 1.0
                num = (y[i] == J) ? 1.0 : 0.0
                for j in 1:J-1
                    v = exp(X[i,:]'β[:,j] + γ * (Z[i,j] - Z[i,J]))
                    denom += v
                    if y[i] == j
                        num = v
                    end
                end
                prob_i += num / denom
            end
            ll += log(prob_i / D)
        end
        return -ll
    end

    θ_init = [vec(β_init); μγ_init; σγ_init]
    result = optimize(log_likelihood, θ_init, LBFGS(); autodiff = :forward)
    
    return result
end

# Question 6: Wrapper function for all estimations

function estimate_all_models(X, Z, y)
    # Assuming β_init, μγ_init, and σγ_init are defined somewhere

    println("Estimating Multinomial Logit (Question 1):")
    result_q1 = multinomial_logit_q1(X, Z, y)
    println(result_q1)

    println("\nEstimating Mixed Logit with Quadrature (Question 3):")
    result_q3 = mixed_logit_q3(X, Z, y, β_init, μγ_init, σγ_init)
    println(result_q3)

    println("\nEstimating Mixed Logit with Modified Likelihood (Question 4):")
    result_q4 = mixed_logit_q4(X, Z, y, β_init, μγ_init, σγ_init)
    println(result_q4)

    println("\nEstimating Mixed Logit with Monte Carlo (Question 5):")
    result_q5 = mixed_logit_q5(X, Z, y, β_init, μγ_init, σγ_init)
    println(result_q5)
end

# Calling the wrapper function
estimate_all_models(X, Z, y)

# Question 7: Unit tests

using Test

@testset "Multinomial Logit Tests" begin
    # Testing multinomial_logit_q1 function
    @test typeof(multinomial_logit_q1(rand(100,3), rand(100,8), rand(1:8, 100))) == Optim.OptimizationResults

    # Testing compute_probabilities function
    β_test = rand(3, 7)
    γ_test = rand()
    X_test = rand(10, 3)
    Z_test = rand(10, 8)
    probs = compute_probabilities(β_test, γ_test, X_test, Z_test)
    @test size(probs) == (10, 8)
    @test all(sum(probs, dims=2) .≈ 1)

    
end

@testset "Mixed Logit Tests" begin
    # Testing mixed_logit_q3 function
    @test typeof(mixed_logit_q3(rand(100,3), rand(100,8), rand(1:8, 100), rand(3,7), 0.0, 1.0)) == Optim.OptimizationResults

    # Testing mixed_logit_q4 function
    @test typeof(mixed_logit_q4(rand(100,3), rand(100,8), rand(1:8, 100), rand(3,7), 0.0, 1.0)) == Optim.OptimizationResults

    # Testing mixed_logit_q5 function
    @test typeof(mixed_logit_q5(rand(100,3), rand(100,8), rand(1:8, 100), rand(3,7), 0.0, 1.0)) == Optim.OptimizationResults


end
