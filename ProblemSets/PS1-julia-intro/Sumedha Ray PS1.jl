import Pkg; Pkg.add("CSV")
import Pkg; Pkg.add("DataFrames")
import Pkg; Pkg.add("FreqTables")
import Pkg; Pkg.add("Distributions")
import Pkg; Pkg.add("JLD")

using Random, LinearAlgebra, Statistics, CSV, DataFrames, FreqTables, Distributions, JLD

function q1()
    Random.seed!(1234)

    # (a) Create matrices
    A = rand(Uniform(-5, 10), 10, 7)
    B = rand(Normal(-2, 15), 10, 7)
    C = [A[1:5, 1:5] B[1:5, 6:7]]
    D = copy(A)
    D[D .> 0] .= 0

    # (b) Number of elements in A
    println("Number of elements in A: ", length(A))

    # (c) Number of unique elements in D
    println("Number of unique elements in D: ", length(unique(D)))

    # (d) Create matrix E (vec operator on B)
    E = reshape(B, :, 1)
    # Alternative: E = vec(B)

    # (e) Create 3D array F
    F = cat(A, B, dims=3)

    # (f) Use permutedims on F
    F = permutedims(F, (3, 1, 2))

    # (g) Kronecker product
    G = kron(B, C)
    

    # (h) Save matrices to JLD file
    save("matrixpractice.jld", "A", A, "B", B, "C", C, "D", D, "E", E, "F", F, "G", G)

    # (i) Save subset of matrices
    save("firstmatrix.jld", "A", A, "B", B, "C", C, "D", D)

    # (j) Export C as CSV
    CSV.write("Cmatrix.csv", DataFrame(C, :auto))

    # (k) Export D as tab-delimited file
    CSV.write("Dmatrix.dat", DataFrame(D, :auto), delim='\t')

    return A, B, C, D
end

# Call the function
A, B, C, D = q1()

using Distributions, LinearAlgebra

function q2(A, B, C)
    # (a) Element-wise product of A and B
    AB = A .* B
    AB2 = A .* B  

    # (b) Create Cprime
    Cprime = [x for x in C if -5 <= x <= 5]
    Cprime2 = filter(x -> -5 <= x <= 5, vec(C))

    # (c) Create 3D array X
    N, K, T = 15169, 6, 5
    X = Array{Float64}(undef, N, K, T)
    
    for t in 1:T
        X[:, 1, t] .= 1  # Intercept
        X[:, 2, t] = rand(Bernoulli(0.75 * (6 - t) / 5), N)
        X[:, 3, t] = rand(Normal(15 + t - 1, 5 * (t - 1)), N)
        X[:, 4, t] = rand(Normal(π * (6 - t) / 3, 1/ℯ), N)
        X[:, 5, t] = rand(Binomial(20, 0.6), N)
        X[:, 6, t] = rand(Binomial(20, 0.5), N)
    end

    # (d) Create β matrix
    β = Array{Float64}(undef, K, T)
    for t in 1:T
        β[1, t] = 1 + 0.25*(t-1)
        β[2, t] = log(t)
        β[3, t] = -sqrt(t)
        β[4, t] = exp(t) - exp(t+1)
        β[5, t] = t
        β[6, t] = t/3
    end

    # (e) Create Y matrix
    ε = rand(Normal(0, 0.36), N, T)
    Y = zeros(N, T)
    for t in 1:T
        Y[:, t] = X[:, :, t] * β[:, t] + ε[:, t]
    end

end

using CSV, DataFrames, FreqTables, Statistics

function q3()
    # (a) Import and process the data
    nlsw88 = CSV.read("nlsw88.csv", DataFrame)
    
    # Convert missing values (assuming they are represented as "")
    for col in names(nlsw88)
        nlsw88[!, col] = replace(nlsw88[!, col], "" => missing)
    end
    
    # Save processed data
    CSV.write("nlsw88_processed.csv", nlsw88)

    # (b) Percentage of never married and college graduates
    pct_never_married = mean(nlsw88.never_married .== 1) * 100
    pct_college_grads = mean(nlsw88.grade .>= 16) * 100
    
    println("Percentage never married: $(round(pct_never_married, digits=2))%")
    println("Percentage college graduates: $(round(pct_college_grads, digits=2))%")

    # (c) Percentage in each race category
    race_dist = freqtable(nlsw88.race)
    race_pct = race_dist ./ sum(race_dist) .* 100
    
    println("Race distribution:")
    for (race, pct) in zip(levels(nlsw88.race), race_pct)
        println("$race: $(round(pct, digits=2))%")
    end

    # (d) Summary statistics
    summarystats = describe(nlsw88)
    println(summarystats)
    
    missing_grade = sum(ismissing.(nlsw88.grade))
    println("Number of missing grade observations: $missing_grade")

    # (e) Joint distribution of industry and occupation
    industry_occupation = freqtable(nlsw88.industry, nlsw88.occupation)
    println("Joint distribution of industry and occupation:")
    println(industry_occupation)

    # (f) Mean wage over industry and occupation
    wage_summary = combine(groupby(nlsw88, [:industry, :occupation]), :wage => mean => :mean_wage)
    println("Mean wage by industry and occupation:")
    println(wage_summary)
end

# Call the function
q3()

using Pkg
Pkg.add("FileIO")
using CSV, DataFrames, FreqTables, Statistics
using FileIO  

function q3()
    println("Please select the nlsw88.csv file.")
    file_path = open_dialog("Select nlsw88.csv", multiple=false)
    
    if isempty(file_path)
        error("No file was selected. Please run the function again and select the nlsw88.csv file.")
    end
    
    nlsw88 = CSV.read(file_path, DataFrame)
    
    function q3()
        # (a) Import and process the data
        nlsw88 = CSV.read("nlsw88.csv", DataFrame)
        
        # Convert missing values (assuming they are represented as "")
        for col in names(nlsw88)
            nlsw88[!, col] = replace(nlsw88[!, col], "" => missing)
        end
        
        # Save processed data
        CSV.write("nlsw88_processed.csv", nlsw88)
    
        # (b) Percentage of never married and college graduates
        pct_never_married = mean(nlsw88.never_married .== 1) * 100
        pct_college_grads = mean(nlsw88.grade .>= 16) * 100
        
        println("Percentage never married: $(round(pct_never_married, digits=2))%")
        println("Percentage college graduates: $(round(pct_college_grads, digits=2))%")
    
        # (c) Percentage in each race category
        race_dist = freqtable(nlsw88.race)
        race_pct = race_dist ./ sum(race_dist) .* 100
        
        println("Race distribution:")
        for (race, pct) in zip(levels(nlsw88.race), race_pct)
            println("$race: $(round(pct, digits=2))%")
        end
    
        # (d) Summary statistics
        summarystats = describe(nlsw88)
        println(summarystats)
        
        missing_grade = sum(ismissing.(nlsw88.grade))
        println("Number of missing grade observations: $missing_grade")
    
        # (e) Joint distribution of industry and occupation
        industry_occupation = freqtable(nlsw88.industry, nlsw88.occupation)
        println("Joint distribution of industry and occupation:")
        println(industry_occupation)
    
        # (f) Mean wage over industry and occupation
        wage_summary = combine(groupby(nlsw88, [:industry, :occupation]), :wage => mean => :mean_wage)
        println("Mean wage by industry and occupation:")
        println(wage_summary)
    end
    
    # We call the function
    q3()
    
end

using LinearAlgebra, JLD, DataFrames, CSV

function matrixops(A, B)
    # Error checking
    if size(A) != size(B)
        error("inputs must have the same size")
    end

    # (i) Element-wise product
    element_product = A .* B

    # (ii) Matrix product A'B
    matrix_product = A' * B

    # (iii) Sum of all elements of A + B
    sum_elements = sum(A + B)

    return element_product, matrix_product, sum_elements
end

function q4()
    # (a) Load matrices from firstmatrix.jld
    data = load("firstmatrix.jld")
    A, B = data["A"], data["B"]

    # (b-d) Evaluate matrixops with A and B
    result_AB = matrixops(A, B)
    println("Result of matrixops(A, B):")
    println("Element-wise product shape: ", size(result_AB[1]))
    println("Matrix product shape: ", size(result_AB[2]))
    println("Sum of elements: ", result_AB[3])

    # (f) Evaluate matrixops with C and D
    C, D = data["C"], data["D"]
    try
        result_CD = matrixops(C, D)
    catch e
        println("Error when calling matrixops(C, D): ", e)
    end

    # (g) Evaluate matrixops with ttl_exp and wage from nlsw88
    df = CSV.read("nlsw88_processed.csv", DataFrame)
    
    # We convert columns to vectors, then reshape into 2D arrays
    ttl_exp = reshape(df.ttl_exp, :, 1)
    wage = reshape(df.wage, :, 1)

    try
        result_exp_wage = matrixops(ttl_exp, wage)
        println("\nResult of matrixops(ttl_exp, wage):")
        println("Element-wise product shape: ", size(result_exp_wage[1]))
        println("Matrix product shape: ", size(result_exp_wage[2]))
        println("Sum of elements: ", result_exp_wage[3])
    catch e
        println("Error when calling matrixops(ttl_exp, wage): ", e)
    end
end

# We call the function
q4()

using Test
using Random
using Distributions
using LinearAlgebra
using DataFrames
using CSV

# Helper function to create test matrices
function create_test_matrices()
    Random.seed!(1234)
    A = rand(Uniform(-5, 10), 5, 5)
    B = rand(Normal(-2, 15), 5, 5)
    C = rand(5, 5)
    return A, B, C
end

using Test
using Random
using Distributions
using LinearAlgebra
using DataFrames
using CSV
using JLD

function run_test(test_name, test_function)
    println("Starting test: $test_name")
    try
        test_function()
        println("Test $test_name passed successfully")
    catch e
        println("Error in test $test_name:")
        showerror(stdout, e, catch_backtrace())
        println()
    end
    println("Finished test: $test_name\n")
end

# Individual test functions
function test_q1()
    A, B, C, D = q1()
    @test size(A) == (10, 7)
    @test size(B) == (10, 7)
    @test size(C) == (5, 7)
    @test size(D) == (10, 7)
    @test all(D .<= 0)
    @test isfile("matrixpractice.jld")
    @test isfile("firstmatrix.jld")
    @test isfile("Cmatrix.csv")
    @test isfile("Dmatrix.dat")
end

function test_q2()
    A, B, C = q1()[1:3]  # Get A, B, C from q1
    q2(A, B, C)
    @test true  
end

function test_q3()
    q3()
    @test isfile("nlsw88_processed.csv")
    df = CSV.read("nlsw88_processed.csv", DataFrame)
    @test nrow(df) > 0
    @test "industry" in names(df)
    @test "occupation" in names(df)
    @test "wage" in names(df)
end

function test_matrixops()
    A, B = q1()[1:2]  # Get A and B from q1
    element_product, matrix_product, sum_elements = matrixops(A, B)
    @test size(element_product) == size(A)
    @test size(matrix_product) == (size(A, 2), size(B, 2))
    @test sum_elements ≈ sum(A + B)
    @test_throws ErrorException matrixops(A, rand(4, 4))
end

function test_q4()
    q4()
    @test true 
end


run_test("q1", test_q1)
run_test("q2", test_q2)
run_test("q3", test_q3)
run_test("matrixops", test_matrixops)
run_test("q4", test_q4)