using Pkg

Pkg.add("Optim")
Pkg.add("HTTP")
Pkg.add("GLM")

using LinearAlgebra
using Random
using Statistics
using DataFrames
using CSV
using FreqTables
using Optim
using HTTP
using GLM

#.............................................
# Problem 1
#.............................................

using Optim

f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
negf(x) = x[1]^4+10x[1]^3+2x[1]^2+3x[1]+2
startval = rand(1) # random number as starting value
result = optimize(negf, startval, LBFGS())

println("argmin (minimizer) is ",Optim.minimizer(result)[1])
println("min is ",Optim.minimum(result))

#.............................................
# Problem 2
#.............................................

using DataFrames
using CSV
using HTTP
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)

# Set up  X and Y
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

# Define OLS objective function
function ols(beta, X, y)
    ssr = (y.-X*beta)'*(y.-X*beta)
    return ssr
end

# Estimate using Optim
beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS(), Optim.Options(g_tol=1e-6, iterations=100_000, show_trace=true))
println(beta_hat_ols.minimizer)

# Estimate the analytical solution
using GLM
bols = inv(X'*X)*X'*y
println("Optim estimates:")
println(beta_hat_ols.minimizer)

# Estimate using GLM
df.white = df.race.==1
bols_lm = lm(@formula(married ~ age + white + collgrad), df)
println("\nGLM estimates:")
println(coef(bols_lm))

# Verify that all methods give the same estimates
println("\nAll estimates are similar:")
println(isapprox(beta_hat_ols.minimizer, bols, atol=1e-5) && 
        isapprox(bols, coef(bols_lm), atol=1e-5))

#.............................................
# Problem 3
#.............................................

        function logit_ll(beta, X, y)
            Xb = X * beta
            p = 1 ./ (1 .+ exp.(-Xb))
            ll = sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
            return -ll  # Return negative because Optim minimizes
        end

# We have to ensure X and y are defined as before
X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.married.==1

# Initial values
initial_beta = zeros(size(X, 2))

# Optimize
result_logit = optimize(b -> logit_ll(b, X, y), initial_beta, LBFGS(),
                        Optim.Options(g_tol=1e-6, iterations=100_000))

# Print results
println("Logit coefficients:")
println(Optim.minimizer(result_logit))

# Fit logit model using GLM
logit_model = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())

println("\nGLM Logit coefficients:")
println(coef(logit_model))

println("\nOptim and GLM estimates are similar:")
println(isapprox(Optim.minimizer(result_logit), coef(logit_model), atol=1e-5))

#.............................................
# Problem 4
#.............................................

using GLM
# Ensure 'white' variable is defined
df.white = df.race .== 1

# Fit logit model using GLM
logit_model_glm = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())

# Print the coefficients
println("GLM Logit coefficients:")
println(coef(logit_model_glm))

println("\nComparing Optim and GLM estimates:")
optim_coefs = Optim.minimizer(result_logit)  # Make sure this is defined from question 3
glm_coefs = coef(logit_model_glm)

for i in 1:length(optim_coefs)
    println("Coefficient $i:")
    println("  Optim: $(optim_coefs[i])")
    println("  GLM:   $(glm_coefs[i])")
    println("  Difference: $(abs(optim_coefs[i] - glm_coefs[i]))")
end

# Check if they're approximately equal
println("\nOptim and GLM estimates are approximately equal:")
println(isapprox(optim_coefs, glm_coefs, atol=1e-5))

# Print summary of the GLM model
println("\nGLM Model Summary:")
println(logit_model_glm)

#.............................................
# Problem 5
#.............................................

freqtable(df, :occupation) # note small number of obs in some occupations
df = dropmissing(df, :occupation)
df[df.occupation.==10,:occupation] .= 9
df[df.occupation.==11,:occupation] .= 9
df[df.occupation.==12,:occupation] .= 9
df[df.occupation.==13,:occupation] .= 9
freqtable(df, :occupation) # problem solved

X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
y = df.occupation

function mlogit(alpha, X, d)

    # your turn

    return loglike
end

#..............................................
# Problem 6
#..............................................
using DataFrames, CSV, HTTP, Optim, GLM, LinearAlgebra, FreqTables, Random

function econometrics_ps2()
    # Data loading and preprocessing
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
    df = CSV.read(HTTP.get(url).body, DataFrame)
    
    println("Problem 1: Basic optimization")
    f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
    negf(x) = -f(x)
    result = optimize(negf, [rand()], LBFGS())
    println("Optimizer: ", Optim.minimizer(result)[1])
    println("Maximum value: ", -Optim.minimum(result))

    println("\nProblem 2: OLS Estimation")
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.married.==1
    
    function ols(beta, X, y)
        ssr = (y.-X*beta)'*(y.-X*beta)
        return ssr
    end
    
    beta_hat_ols = optimize(b -> ols(b, X, y), rand(size(X,2)), LBFGS())
    println("OLS estimates (Optim):")
    println(Optim.minimizer(beta_hat_ols))
    
    bols = inv(X'*X)*X'*y
    println("OLS estimates (Analytical):")
    println(bols)
    
    df.white = df.race.==1
    bols_lm = lm(@formula(married ~ age + white + collgrad), df)
    println("OLS estimates (GLM):")
    println(coef(bols_lm))

    println("\nProblem 3 & 4: Logit Estimation")
    function logit_ll(beta, X, y)
        Xb = X * beta
        p = 1 ./ (1 .+ exp.(-Xb))
        ll = sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
        return -ll
    end
    
    result_logit = optimize(b -> logit_ll(b, X, y), zeros(size(X, 2)), LBFGS())
    println("Logit estimates (Optim):")
    println(Optim.minimizer(result_logit))
    
    logit_model = glm(@formula(married ~ age + white + collgrad), df, Binomial(), LogitLink())
    println("Logit estimates (GLM):")
    println(coef(logit_model))

    println("\nProblem 5: Multinomial Logit Estimation")
    df = dropmissing(df, :occupation)
    for i in 8:13
        df[df.occupation.==i, :occupation] .= 7
    end
    
    X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
    y = df.occupation
    
    function mlogit_ll(alpha, X, y)
        n, k = size(X)
        J = length(unique(y))
        alpha_matrix = reshape(alpha, k, J-1)
        
        ll = 0.0
        for i in 1:n
            denom = 1 + sum(exp.(X[i,:]' * alpha_matrix))
            if y[i] == J
                p = 1 / denom
            else
                p = exp(X[i,:]' * alpha_matrix[:,y[i]]) / denom
            end
            ll += log(p)
        end
        
        return -ll
    end
    
    k = size(X, 2)
    J = length(unique(y))
    initial_alpha = rand(k * (J-1))
    
    result_mlogit = optimize(a -> mlogit_ll(a, X, y), initial_alpha, LBFGS(),
                             Optim.Options(g_tol=1e-5, iterations=100_000))
    
    println("Multinomial Logit coefficients:")
    println(reshape(Optim.minimizer(result_mlogit), size(X,2), J-1))
end

# Call the main function
econometrics_ps2()


#..............................................
# Problem 7
#..............................................
function run_tests()
    @testset "Problem Set 2 Tests" begin
        # Test for Problem 1: Basic optimization
        @testset "Basic Optimization" begin
            f(x) = -x[1]^4-10x[1]^3-2x[1]^2-3x[1]-2
            negf(x) = -f(x)
            result = optimize(negf, [0.0], LBFGS())
            @test isapprox(Optim.minimizer(result)[1], -7.38, atol=1e-2)
            @test isapprox(-Optim.minimum(result), 964.3, atol=1e-1)
        end

        # Test for Problem 2: OLS Estimation
        @testset "OLS Estimation" begin
            X = [ones(5) [1,2,3,4,5]]
            y = [2.0, 4.0, 5.0, 4.0, 5.0]
            
            function ols(beta, X, y)
                ssr = (y.-X*beta)'*(y.-X*beta)
                return ssr
            end
            
            beta_hat_ols = optimize(b -> ols(b, X, y), rand(2), LBFGS())
            @test isapprox(Optim.minimizer(beta_hat_ols), [2.2, 0.6], atol=1e-1)
        end

        # Test for Problem 3 & 4: Logit Estimation
        @testset "Logit Estimation" begin
            function logit_ll(beta, X, y)
                Xb = X * beta
                p = 1 ./ (1 .+ exp.(-Xb))
                ll = sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
                return -ll
            end
            
            X = [ones(5) [1,2,3,4,5]]
            y = [0.0, 0.0, 1.0, 1.0, 1.0]
            
            result_logit = optimize(b -> logit_ll(b, X, y), zeros(2), LBFGS())
            @test length(Optim.minimizer(result_logit)) == 2
        end

        # Test for Problem 5: Multinomial Logit Estimation
        @testset "Multinomial Logit Estimation" begin
            function mlogit_ll(alpha, X, y)
                n, k = size(X)
                J = length(unique(y))
                alpha_matrix = reshape(alpha, k, J-1)
                
                ll = 0.0
                for i in 1:n
                    denom = 1 + sum(exp.(X[i,:]' * alpha_matrix))
                    if y[i] == J
                        p = 1 / denom
                    else
                        p = exp(X[i,:]' * alpha_matrix[:,y[i]]) / denom
                    end
                    ll += log(p)
                end
                
                return -ll
            end
            
            X = [ones(5) [1,2,3,4,5]]
            y = [1, 2, 3, 2, 1]
            
            k = size(X, 2)
            J = length(unique(y))
            initial_alpha = rand(k * (J-1))
            
            result_mlogit = optimize(a -> mlogit_ll(a, X, y), initial_alpha, LBFGS(),
                                     Optim.Options(g_tol=1e-5, iterations=100_000))
            
            @test size(reshape(Optim.minimizer(result_mlogit), size(X,2), J-1)) == (2, 2)
        end
    end
end