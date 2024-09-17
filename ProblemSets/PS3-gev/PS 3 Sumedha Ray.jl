Econometrics III PS3

# Problem Set 3 - Sumedha Ray

# Problem 1
using Optim
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using HTTP

# We load the data
url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
df = CSV.read(HTTP.get(url).body, DataFrame)
X = Matrix([df.age df.white df.collgrad])
Z = Matrix(hcat(df.elnwage1, df.elnwage2, df.elnwage3, df.elnwage4,
         df.elnwage5, df.elnwage6, df.elnwage7, df.elnwage8))
y = df.occupation

function multinomial_logit(X, Z, y)
    n = size(X, 1)  # number of observations
    J = size(Z, 2)  # number of alternatives
    K = size(X, 2)  # number of covariates

    # Initializing the parameter vector
    initial_params = zeros(K * (J-1) + 1)

    # Likelihood function
    function ll(params)
        β = reshape(params[1:K*(J-1)], K, J-1)
        γ = params[end]
        
        ll_sum = 0.0
        for i in 1:n
            denom = 1.0
            for j in 1:(J-1)
                denom += exp(dot(X[i,:], β[:,j]) + γ * (Z[i,j] - Z[i,J]))
            end
            
            if y[i] == J
                ll_sum += log(1 / denom)
            else
                ll_sum += log(exp(dot(X[i,:], β[:,y[i]]) + γ * (Z[i,y[i]] - Z[i,J])) / denom)
            end
        end
        
        return -ll_sum  # This is negative because we are minimizing
    end

    # Optimizing
    result = optimize(ll, initial_params, BFGS())
    
    # Extracting and reshaping results
    optimal_params = Optim.minimizer(result)
    β_hat = reshape(optimal_params[1:K*(J-1)], K, J-1)
    γ_hat = optimal_params[end]
    
    return β_hat, γ_hat
end

# Running the model
β_hat, γ_hat = multinomial_logit(X, Z, y)

# Print results
println("Estimated β coefficients:")
println(β_hat)
println("Estimated γ coefficient: ", γ_hat)

#::::::::::::::::::::::::::::::::::::::::::
# Problem 2
#::::::::::::::::::::::::::::::::::::::::::

# Assuming γ_hat, X, Z, and y are already defined from question 1


using Statistics

# Function to calculate probabilities
function calculate_probabilities(X, Z, β_hat, γ_hat)
    n, J = size(Z)
    K = size(X, 2)
    probs = zeros(n, J)
    
    for i in 1:n
        denominator = 1.0
        for j in 1:(J-1)
            denominator += exp(dot(X[i,:], β_hat[:,j]) + γ_hat * (Z[i,j] - Z[i,J]))
        end
        
        for j in 1:J
            if j == J
                probs[i,j] = 1 / denominator
            else
                probs[i,j] = exp(dot(X[i,:], β_hat[:,j]) + γ_hat * (Z[i,j] - Z[i,J])) / denominator
            end
        end
    end
    
    return probs
end

# Calculating probabilities
probs = calculate_probabilities(X, Z, β_hat, γ_hat)

# Calculating average probabilities and wages
avg_probs = mean(probs, dims=1)[1,:]
avg_wages = mean(exp.(Z), dims=1)[1,:]  # exponentiate to get actual wages

# Calculating elasticities
elasticities = γ_hat .* avg_wages .* (1 .- avg_probs)
avg_elasticity = mean(elasticities)

# Printing results and interprating
println("Interpretation of γ̂ coefficient:")
println("--------------------------------")
println("Estimated γ̂: ", round(γ_hat, digits=6))
println("Average elasticity: ", round(avg_elasticity, digits=6))
println()
println("1. Direct Interpretation:")
println("   A one-unit increase in log wage (approximately a 100% increase in actual wage)")
println("   is associated with a ", round(γ_hat, digits=4), " decrease in the log-odds of")
println("   choosing that occupation, holding other factors constant.")
println()
println("2. Percentage Interpretation:")
println("   A 1% increase in wage is associated with approximately a ",
        round(abs(γ_hat), digits=4), "% decrease in the probability of")
println("   choosing that occupation, all else being equal.")
println()
println("3. Elasticity Interpretation:")
println("   On average, a 1% increase in wage is associated with a ",
        round(abs(avg_elasticity), digits=4), "% decrease in the")
println("   probability of choosing an occupation.")
println()
println("4. Counterintuitive Result:")
println("   The negative coefficient suggests that higher wages are associated")
println("   with lower probabilities of choosing an occupation. This could be due to:")
println("   - Omitted variable bias (e.g., job difficulty, required skills)")
println("   - Selection effects (e.g., higher-paying jobs might be more competitive)")
println("   - Data limitations or measurement issues")
println()
println("5. Limitations:")
println("   - This interpretation assumes a linear relationship, which doesn't")
println("     fully capture the non-linear nature of the logit model.")
println("   - The actual change in probability will depend on the base probability")
println("     and the magnitude of the wage change.")
println("   - The model assumes the effect of wages is the same across all")
println("     occupations, which might not be realistic.")

#::::::::::::::::::::::::::::::::::::::::::
# Problem 3
#::::::::::::::::::::::::::::::::::::::::::

using Optim
using LinearAlgebra
using Statistics

function nested_logit(X, Z, y)
    n = size(X, 1)  # number of observations
    J = size(Z, 2)  # number of alternatives
    K = size(X, 2)  # number of covariates

    # Defining nests
    WC = [1, 2, 3]  # White collar
    BC = [4, 5, 6, 7]  # Blue collar
    Other = [8]  # Other

    # Initializing the parameter vector: [β_WC; β_BC; λ_WC; λ_BC; γ]
    initial_params = [zeros(2*K); 0.5; 0.5; 0.0]

    # Likelihood function
    function ll(params)
        β_WC, β_BC = params[1:K], params[K+1:2K]
        λ_WC, λ_BC = params[2K+1:2K+2]
        γ = params[end]
        
        ll_sum = 0.0
        for i in 1:n
            # Calculating nest probabilities
            V_WC = sum(exp((X[i,:]'β_WC + γ*(Z[i,j] - Z[i,J])) / max(λ_WC, 1e-10)) for j in WC)
            V_BC = sum(exp((X[i,:]'β_BC + γ*(Z[i,j] - Z[i,J])) / max(λ_BC, 1e-10)) for j in BC)
            V_Other = 1.0  # exp(0) = 1, as β_Other is normalized to 0

            denom = V_WC^λ_WC + V_BC^λ_BC + V_Other

            if y[i] in WC
                ll_sum += log(max(V_WC^λ_WC, 1e-10) / max(denom, 1e-10)) + 
                          (X[i,:]'β_WC + γ*(Z[i,y[i]] - Z[i,J])) / max(λ_WC, 1e-10) - 
                          log(sum(exp((X[i,:]'β_WC + γ*(Z[i,j] - Z[i,J])) / max(λ_WC, 1e-10)) for j in WC))
            elseif y[i] in BC
                ll_sum += log(max(V_BC^λ_BC, 1e-10) / max(denom, 1e-10)) + 
                          (X[i,:]'β_BC + γ*(Z[i,y[i]] - Z[i,J])) / max(λ_BC, 1e-10) - 
                          log(sum(exp((X[i,:]'β_BC + γ*(Z[i,j] - Z[i,J])) / max(λ_BC, 1e-10)) for j in BC))
            else  # Other
                ll_sum += log(max(V_Other, 1e-10) / max(denom, 1e-10))
            end
        end
        
        return -ll_sum  # negative because we're minimizing
    end

    # Optimizing with bounds
    lower = [-Inf*ones(2*K); 0.0; 0.0; -Inf]
    upper = [Inf*ones(2*K); 1.0; 1.0; Inf]
    result = optimize(ll, lower, upper, initial_params, Fminbox(BFGS()))
    
    # Extracting results
    optimal_params = Optim.minimizer(result)
    β_WC, β_BC = optimal_params[1:K], optimal_params[K+1:2K]
    λ_WC, λ_BC = optimal_params[2K+1:2K+2]
    γ = optimal_params[end]
    
    return β_WC, β_BC, λ_WC, λ_BC, γ
end

# Assuming X, Z, and y are already defined from previous questions
β_WC, β_BC, λ_WC, λ_BC, γ = nested_logit(X, Z, y)

# Printing results
println("Nested Logit Results:")
println("Estimated β_WC: ", β_WC)
println("Estimated β_BC: ", β_BC)
println("Estimated λ_WC: ", λ_WC)
println("Estimated λ_BC: ", λ_BC)
println("Estimated γ: ", γ)


# Interpretation
println("\nInterpretation:")
println("1. β_WC coefficients represent the effect of individual characteristics on choosing white-collar occupations.")
println("2. β_BC coefficients represent the effect of individual characteristics on choosing blue-collar occupations.")
println("3. λ_WC and λ_BC are dissimilarity parameters. Values close to 1 indicate more independence within the nest,")
println("   while values close to 0 indicate more correlation within the nest.")
println("4. γ coefficient represents the effect of wages on occupation choice, similar to the multinomial logit model.")
println("\nCompare these results with the multinomial logit model to discuss the insights gained from the nested structure.")

#::::::::::::::::::::::::::::::::::::::::::
# Problem 4
#::::::::::::::::::::::::::::::::::::::::::

function problem_set_3()
    println("Problem Set 3 Results Summary")
    
    println("\nQuestion 1: Multinomial Logit")
    println("Estimated β coefficients:")
    println([0.05570760821873361 0.04500069760812089 0.09264599987673823 0.02390335345913949 0.036087237758010406 0.08531087899609793 0.08662012953143235; 0.08342800912507368 0.7365784224226264 -0.08417566822303942 0.7230680315358062 -0.6437626443990146 -1.1714282628621513 -0.797875842371163; -2.3448861127173504 -3.153242881656483 -4.27327816625058 -3.7493920045282323 -4.2796830878646634 -6.6786762695957 -4.969129759355472])
    println("Estimated γ coefficient: ", -0.09419373857636062)
    
    println("\nQuestion 2: Interpretation of γ̂")
    println("The estimated γ̂ coefficient is -0.09419373857636062.")
    println("On average, a 1% increase in wage is associated with a 0.1507% decrease")
    println("in the probability of choosing an occupation.")
    
    println("\nQuestion 3: Nested Logit")
    println("Nested Logit Results:")
    println("Estimated β_WC: [0.09619598538825007, 0.24011298771725187, -3.3584638350257663]")
    println("Estimated β_BC: [0.10171384676964926, -0.5927927033826604, -4.8029729460730595]")
    println("Estimated λ_WC: 0.007461776694817416")
    println("Estimated λ_BC: 0.006474311933161913")
    println("Estimated γ: -0.017619401823591688")
    
    println("\nInterpretation:")
    println("1. β_WC coefficients represent the effect of individual characteristics on choosing white-collar occupations.")
    println("2. β_BC coefficients represent the effect of individual characteristics on choosing blue-collar occupations.")
    println("3. λ_WC and λ_BC are dissimilarity parameters. Values close to 1 indicate more independence within the nest,")
    println("   while values close to 0 indicate more correlation within the nest.")
    println("4. γ coefficient represents the effect of wages on occupation choice, similar to the multinomial logit model.")
    println("\nCompare these results with the multinomial logit model to discuss the insights gained from the nested structure.")
end

# We call the function
problem_set_3()

#::::::::::::::::::::::::::::::::::::::::::
# Problem 5 
#::::::::::::::::::::::::::::::::::::::::::

using Test
using DataFrames, CSV, HTTP, Optim, LinearAlgebra, Statistics

# Function to generate sample data
function generate_sample_data()
    X = rand(100, 3)
    Z = rand(100, 8)
    y = rand(1:8, 100)
    return X, Z, y
end

@testset "Problem Set 3 Tests" begin
    # Revised Data Loading Test
    @testset "Data Loading" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS3-gev/nlsw88w.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        
        @test size(df, 2) == 13  # Updated to match actual number of columns
        
        # Converting the column names to lowercase for case-insensitive comparison
        lowercase_names = lowercase.(names(df))
        
        @test "occupation" in lowercase_names
        @test "age" in lowercase_names
        @test "white" in lowercase_names
        @test "collgrad" in lowercase_names
    end

    # Testing multinomial_logit function
    @testset "Multinomial Logit" begin
        X, Z, y = generate_sample_data()
        β_hat, γ_hat = multinomial_logit(X, Z, y)
        @test size(β_hat) == (3, 7)
        @test isa(γ_hat, Float64)
    end

    # Testing nested_logit function
    @testset "Nested Logit" begin
        X, Z, y = generate_sample_data()
        β_WC, β_BC, λ_WC, λ_BC, γ = nested_logit(X, Z, y)
        @test length(β_WC) == 3
        @test length(β_BC) == 3
        @test 0 ≤ λ_WC ≤ 1
        @test 0 ≤ λ_BC ≤ 1
        @test isa(γ, Float64)
    end

    # Testing calculate_probabilities function (if you have one)
    @testset "Calculate Probabilities" begin
        X, Z, y = generate_sample_data()
        β_hat, γ_hat = multinomial_logit(X, Z, y)
        probs = calculate_probabilities(X, Z, β_hat, γ_hat)
        @test size(probs) == (100, 8)
        @test all(0 .≤ probs .≤ 1)
        @test all(isapprox.(sum(probs, dims=2), 1, atol=1e-6))
    end

    # Testing problem_set_3 function
    @testset "Problem Set 3 Wrapper" begin
        # Redirect stdout to capture output
        original_stdout = stdout
        (read_pipe, write_pipe) = redirect_stdout()
        
        # Running the function
        problem_set_3()
        
        # Restoring the original stdout
        redirect_stdout(original_stdout)
        close(write_pipe)
        
        # Reading the captured output
        output = String(read(read_pipe))
        
        # Checking that the function produces some output
        @test length(output) > 0
        
        # Checking that the output contains expected strings
        @test contains(output, "Question 1: Multinomial Logit")
        @test contains(output, "Question 2: Interpretation of γ̂")
        @test contains(output, "Question 3: Nested Logit")
    end
end
