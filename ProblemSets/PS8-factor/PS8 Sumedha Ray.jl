# PS8 Question 1: Basic Wage Regression
# ECON 6343: Econometrics III

# Load required packages
using DataFrames
using CSV
using GLM

function run_basic_analysis()
    println("Starting analysis...")
    
    # Use the full path to the file
    file_path = joinpath(pwd(), "ProblemSets", "PS8-factor", "nlsy.csv")
    
    # Load the data
    println("Loading data from: ", file_path)
    df = CSV.read(file_path, DataFrame)
    println("Successfully loaded data with $(size(df,1)) rows and $(size(df,2)) columns")
    
    # Run the baseline regression with correct variable names
    println("\nQuestion 1: Basic Wage Regression")
    println("Formula: logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr")
    
    # Estimate the model
    model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
    
    # Print the results
    println("\nRegression Results:")
    println("===================")
    println(model)
    
    return df, model
end

# Run the analysis
println("Starting PS8 Question 1 analysis...")
df, model = run_basic_analysis()


#Question 2
#:::::::::::::::::::::::::::::::::::::::

using DataFrames, CSV, Statistics, LinearAlgebra, Printf

"""
    compute_asvab_correlation(df::DataFrame)

Compute and format the correlation matrix for the six ASVAB variables.
Returns both the correlation matrix and a formatted string for display.
"""
function compute_asvab_correlation(df::DataFrame)
    # Select only the ASVAB variables
    asvab_vars = ["asvabAR", "asvabWK", "asvabPC", "asvabNO", "asvabCS", "asvabMK"]
    asvab_data = Matrix(df[:, asvab_vars])
    
    # Compute correlation matrix
    cor_matrix = cor(asvab_data)
    
    # Create formatted output
    println("\nCorrelation Matrix for ASVAB Variables:")
    println("======================================")
    
    # Print header
    print("         ")
    for var in asvab_vars
        print(lpad(var[6:end], 8), " ")  # Remove "asvab" prefix for cleaner output
    end
    println()
    
    # Print correlation matrix with row labels
    for (i, var) in enumerate(asvab_vars)
        print(lpad(var[6:end], 8), " ")  # Row label
        for j in 1:6
            print(lpad(round(cor_matrix[i,j], digits=3), 8), " ")
        end
        println()
    end
    
    return cor_matrix
end

# Run the analysis using the previously loaded dataframe
println("Question 2: Computing ASVAB Variable Correlations")
cor_matrix = compute_asvab_correlation(df)


#Question 3
#:::::::::::::::::::::::::::::::::::::::

using DataFrames, CSV, GLM, Statistics

"""
    run_expanded_regression(df::DataFrame)

Run the wage regression including all ASVAB variables.
"""
function run_expanded_regression(df::DataFrame)
    println("\nQuestion 3: Expanded Regression with ASVAB Variables")
    
    # Create and estimate the expanded model
    formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + 
                      asvabAR + asvabWK + asvabPC + asvabNO + asvabCS + asvabMK)
    
    model = lm(formula, df)
    
    # Print results
    println("\nRegression Results (Including ASVAB variables):")
    println("============================================")
    println(model)
    
    # Compute VIF for each ASVAB variable to quantify multicollinearity
    asvab_vars = ["asvabAR", "asvabWK", "asvabPC", "asvabNO", "asvabCS", "asvabMK"]
    vifs = Dict{String, Float64}()
    
    for var in asvab_vars
        # Create formula for auxiliary regression
        others = filter(x -> x != var, asvab_vars)
        aux_formula = Term(Symbol(var)) ~ sum(Term.(Symbol.(others)))
        
        # Run auxiliary regression
        aux_model = lm(aux_formula, df)
        
        # Calculate VIF
        r2 = r²(aux_model)
        vifs[var] = 1 / (1 - r2)
    end
    
    # Print VIF results
    println("\nVariance Inflation Factors for ASVAB variables:")
    println("=============================================")
    for (var, vif) in vifs
        println(rpad(var, 8), ": ", round(vif, digits=2))
    end
    
    return model, vifs
end

# Run the expanded regression
expanded_model, vifs = run_expanded_regression(df)

#:::::::::::::::::::::::::::::::::::::::
#Question 4
#:::::::::::::::::::::::::::::::::::::::
using Pkg
Pkg.add("MultivariateStats")

using DataFrames, CSV, GLM, Statistics, MultivariateStats, LinearAlgebra

function run_pca_regression(df::DataFrame)
    println("\nQuestion 4: Regression with First Principal Component")
    
    # Extract ASVAB variables and convert to matrix
    asvab_vars = ["asvabAR", "asvabWK", "asvabPC", "asvabNO", "asvabCS", "asvabMK"]
    asvab_matrix = Matrix(df[:, asvab_vars])
    
    # Transpose matrix for PCA
    asvab_matrix_t = transpose(asvab_matrix)
    
    # Fit PCA model
    println("\nFitting PCA model...")
    M = fit(PCA, asvab_matrix_t; maxoutdim=1)
    
    # Transform data to get first principal component
    pc1 = MultivariateStats.transform(M, asvab_matrix_t)
    
    # Add first principal component to dataframe
    df.pc1 = vec(transpose(pc1))
    
    # Calculate and print variance explained
    println("\nPCA Summary Statistics:")
    println("=====================")
    var_explained = principalratio(M)
    println("Variance explained by first principal component: ", 
            round(var_explained * 100, digits=2), "%")
    
    # Print loadings
    println("\nPCA Loadings (correlation between PC1 and original variables):")
    println("====================================================")
    loadings = projection(M)
    for (var, loading) in zip(asvab_vars, loadings)
        println(rpad(var, 8), ": ", round(loading, digits=4))
    end
    
    # Run regression with first principal component
    println("\nRegression Results:")
    println("==================")
    formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + 
                      grad4yr + pc1)
    model = lm(formula, df)
    println(model)
    
    return M, model
end

# Run PCA regression
pca_model, reg_model = run_pca_regression(df)

#:::::::::::::::::::::::::::::::::::::::
#Question 5
#:::::::::::::::::::::::::::::::::::::::

using DataFrames, CSV, GLM, Statistics, MultivariateStats, LinearAlgebra

"""
    run_fa_regression(df::DataFrame)

Perform Factor Analysis on ASVAB variables and run regression with first factor.
"""
function run_fa_regression(df::DataFrame)
    println("\nQuestion 5: Regression with Factor Analysis")
    
    # Extract ASVAB variables and convert to matrix
    asvab_vars = ["asvabAR", "asvabWK", "asvabPC", "asvabNO", "asvabCS", "asvabMK"]
    asvab_matrix = Matrix(df[:, asvab_vars])
    
    # Transpose matrix for FA (variables should be in columns)
    asvab_matrix_t = transpose(asvab_matrix)
    
    # Fit Factor Analysis model
    println("\nFitting Factor Analysis model...")
    M = fit(FactorAnalysis, asvab_matrix_t; maxoutdim=1)
    
    # Transform data to get factor scores
    fa1 = MultivariateStats.transform(M, asvab_matrix_t)
    
    # Add factor scores to dataframe
    df.fa1 = vec(transpose(fa1))
    
    # Print factor loadings
    println("\nFactor Loadings (correlation between Factor 1 and original variables):")
    println("=======================================================")
    loadings = projection(M)
    for (var, loading) in zip(asvab_vars, loadings)
        println(rpad(var, 8), ": ", round(loading, digits=4))
    end
    
    # Print variance explained
    println("\nFactor Analysis Summary:")
    println("=====================")
    var_exp = sum(loadings .^ 2) / length(loadings)  # Average of squared loadings
    println("Proportion of variance explained: ", round(var_exp * 100, digits=2), "%")
    
    # Run regression with factor score
    println("\nRegression Results:")
    println("==================")
    formula = @formula(logwage ~ black + hispanic + female + schoolt + gradHS + 
                      grad4yr + fa1)
    model = lm(formula, df)
    println(model)
    
    return M, model
end

# Run Factor Analysis regression
fa_model, reg_model = run_fa_regression(df)



#:::::::::::::::::::::::::::::::::::::::
#Question 6
#:::::::::::::::::::::::::::::::::::::::

function run_measurement_system()
    println("\nQuestion 6: Maximum Likelihood Estimation of Measurement System")
    println("============================================================")
    
    # Load the data using the correct file path
    println("Loading data...")
    file_path = joinpath(pwd(), "ProblemSets", "PS8-factor", "nlsy.csv")
    println("Loading data from: ", file_path)
    df = CSV.read(file_path, DataFrame)
    
    # Rest of the code remains the same
    # Prepare data matrices
    X_m = [ones(size(df,1)) df.black df.hispanic df.female]
    X = [ones(size(df,1)) df.black df.hispanic df.female df.schoolt df.gradHS df.grad4yr]
    asvab_vars = ["asvabAR", "asvabWK", "asvabPC", "asvabNO", "asvabCS", "asvabMK"]
    M = Matrix(df[:, asvab_vars])
    y = df.logwage
    
    # Set dimensions
    N = size(X, 1)
    J = length(asvab_vars)
    K_m = size(X_m, 2)
    K = size(X, 2)
    
    # Initialize parameters
    function init_params()
        # For each ASVAB equation: α₀, α₁, α₂, α₃, γ
        α_params = zeros(J * K_m)
        γ_params = ones(J) * 0.5
        # For wage equation: β, δ
        β_params = zeros(K)
        δ_param = 1.0
        # Standard deviations
        σ_j = ones(J)
        σ_w = 1.0
        
        return vcat(α_params, γ_params, β_params, δ_param, σ_j, σ_w)
    end
    
    # Likelihood function
    function loglikelihood(θ)
        # Unpack parameters
        idx = 1
        α = reshape(θ[idx:idx+J*K_m-1], K_m, J); idx += J*K_m
        γ = θ[idx:idx+J-1]; idx += J
        β = θ[idx:idx+K-1]; idx += K
        δ = θ[idx]; idx += 1
        σ_j = exp.(θ[idx:idx+J-1]); idx += J
        σ_w = exp(θ[idx])
        
        # Set up quadrature
        n_quad = 7
        nodes, weights = lgwt(n_quad, -4, 4)
        
        # Initialize log-likelihood
        ll = 0.0
        
        # Loop over observations
        for i in 1:N
            # Initialize likelihood for observation i
            li = 0.0
            
            for (node, weight) in zip(nodes, weights)
                # Measurement equations
                l_m = 1.0
                for j in 1:J
                    μ_ij = dot(X_m[i,:], α[:,j]) + γ[j]*node
                    l_m *= pdf(Normal(μ_ij, σ_j[j]), M[i,j])
                end
                
                # Wage equation
                μ_i = dot(X[i,:], β) + δ*node
                l_w = pdf(Normal(μ_i, σ_w), y[i])
                
                # Combine and weight
                li += weight * l_m * l_w * pdf(Normal(0,1), node)
            end
            
            ll += log(li + 1e-10)  # Added small constant to prevent log(0)
        end
        
        return -ll  # Return negative for minimization
    end
    
    # Optimize
    println("Starting optimization...")
    θ_init = init_params()
    
    try
        res = optimize(loglikelihood, θ_init, LBFGS(), 
                      Optim.Options(show_trace=true, store_trace=true,
                                  iterations=1000, g_tol=1e-5))
        
        # Extract and display results
        θ_hat = Optim.minimizer(res)
        
        # Print results
        println("\nResults:")
        println("--------")
        println("Convergence: ", Optim.converged(res))
        println("Final log-likelihood: ", -Optim.minimum(res))
        
        # Unpack and print parameter estimates
        idx = 1
        α = reshape(θ_hat[idx:idx+J*K_m-1], K_m, J); idx += J*K_m
        γ = θ_hat[idx:idx+J-1]; idx += J
        β = θ_hat[idx:idx+K-1]; idx += K
        δ = θ_hat[idx]; idx += 1
        σ_j = exp.(θ_hat[idx:idx+J-1]); idx += J
        σ_w = exp(θ_hat[idx])
        
        println("\nMeasurement Equations Parameters (α):")
        println("Format: [Intercept, Black, Hispanic, Female]")
        for j in 1:J
            println("ASVAB $j: ", round.(α[:,j], digits=4))
        end
        
        println("\nFactor Loadings (γ):")
        for j in 1:J
            println("ASVAB $j: ", round(γ[j], digits=4))
        end
        
        println("\nWage Equation Parameters (β):")
        println("Format: [Intercept, Black, Hispanic, Female, School, GradHS, Grad4yr]")
        println(round.(β, digits=4))
        println("Factor Loading (δ): ", round(δ, digits=4))
        
        println("\nStandard Deviations:")
        println("ASVAB equations (σ_j): ", round.(σ_j, digits=4))
        println("Wage equation (σ_w): ", round(σ_w, digits=4))
        
        return θ_hat, res
        
    catch e
        println("Error during optimization: ", e)
        return nothing, nothing
    end
end

# Execute the estimation
θ_hat, res = run_measurement_system()

#:::::::::::::::::::::::::::::::::::::::
#Question 7
#:::::::::::::::::::::::::::::::::::::::

using Test
using DataFrames
using CSV
using LinearAlgebra
using Statistics
using MultivariateStats
using GLM

@testset "PS8 Tests" begin
    # Create more realistic test data
    function create_test_data()
        N = 100  # Larger sample size
        Random.seed!(123)  # For reproducibility
        
        return DataFrame(
            logwage = 2.0 .+ 0.3 .* randn(N),
            black = rand(0:1, N),
            hispanic = rand(0:1, N),
            female = rand(0:1, N),
            schoolt = 12 .+ rand(0:8, N),
            gradHS = rand(0:1, N),
            grad4yr = rand(0:1, N),
            # Create ASVAB scores with realistic correlations
            asvabAR = randn(N),
            asvabWK = randn(N),
            asvabPC = randn(N),
            asvabNO = randn(N),
            asvabCS = randn(N),
            asvabMK = randn(N)
        )
    end

    @testset "Question 1: Basic Wage Regression" begin
        df = create_test_data()
        model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr), df)
        
        @test model isa StatsModels.TableRegressionModel
        @test coef(model) isa Vector{Float64}
        @test !any(isnan.(coef(model)))
        @test !any(isinf.(coef(model)))
    end

    @testset "Question 2: ASVAB Correlation" begin
        df = create_test_data()
        asvab_vars = ["asvabAR", "asvabWK", "asvabPC", "asvabNO", "asvabCS", "asvabMK"]
        cor_matrix = cor(Matrix(df[:, asvab_vars]))
        
        @test size(cor_matrix) == (6, 6)
        @test issymmetric(cor_matrix)
        @test all(diag(cor_matrix) .≈ 1.0)
        @test all(-1 .<= cor_matrix .<= 1)
    end

    @testset "Question 3: Expanded Regression" begin
        df = create_test_data()
        model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + 
                          grad4yr + asvabAR + asvabWK + asvabPC + asvabNO + asvabCS + asvabMK), df)
        
        @test !any(isnan.(coef(model)))
        @test !any(isinf.(coef(model)))
    end

    @testset "Question 4: PCA Regression" begin
        df = create_test_data()
        asvab_vars = ["asvabAR", "asvabWK", "asvabPC", "asvabNO", "asvabCS", "asvabMK"]
        asvab_matrix = Matrix(df[:, asvab_vars])
        
        M = fit(PCA, transpose(asvab_matrix); maxoutdim=1)
        df.pc1 = vec(transpose(MultivariateStats.transform(M, transpose(asvab_matrix))))
        
        model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + pc1), df)
        
        @test !any(isnan.(coef(model)))
        @test !any(isinf.(coef(model)))
    end

    @testset "Question 5: Factor Analysis Regression" begin
        df = create_test_data()
        asvab_vars = ["asvabAR", "asvabWK", "asvabPC", "asvabNO", "asvabCS", "asvabMK"]
        asvab_matrix = Matrix(df[:, asvab_vars])
        
        M = fit(FactorAnalysis, transpose(asvab_matrix); maxoutdim=1)
        df.fa1 = vec(transpose(MultivariateStats.transform(M, transpose(asvab_matrix))))
        
        model = lm(@formula(logwage ~ black + hispanic + female + schoolt + gradHS + grad4yr + fa1), df)
        
        @test !any(isnan.(coef(model)))
        @test !any(isinf.(coef(model)))
    end

    @testset "Question 6: Measurement System" begin
        df = create_test_data()
        J = 6  # Number of ASVAB variables
        K = 7  # Number of wage equation variables
        
        @testset "initialize_parameters" begin
            θ_init = initialize_parameters(J, K)
            expected_length = 4*J + J + K + 1 + J + 1
            @test length(θ_init) == expected_length
            @test all(isfinite.(θ_init))
        end
        
        @testset "compute_log_likelihood" begin
            X_m = [ones(size(df,1)) df.black df.hispanic df.female]
            X = [ones(size(df,1)) df.black df.hispanic df.female df.schoolt df.gradHS df.grad4yr]
            M = Matrix(df[:, r"asvab"])
            y = df.logwage
            
            θ_test = initialize_parameters(J, K)
            ll = compute_log_likelihood(θ_test, X_m, X, M, y)
            
            @test isfinite(ll)
            @test ll isa Real
        end
    end
end

println("All tests completed.")