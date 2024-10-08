Dr. Ransom PS6

using Random, LinearAlgebra, Statistics, Optim, DataFrames, DataFramesMeta, CSV, HTTP, GLM, Test

#:::::::::::::::::::::::::::::::::::::::::
# Question 01
#:::::::::::::::::::::::::::::::::::::::::

# Data loading and preprocessing
function load_and_preprocess_data(url)
    df = CSV.read(HTTP.get(url).body, DataFrame)
    df = @transform(df, :bus_id = 1:nrow(df))
    return df
end

function reshape_data(df)
    dfy = @select(df, :bus_id, Between(:Y1, :Y20), :RouteUsage, :Branded)
    dfy_long = DataFrames.stack(dfy, Not([:bus_id, :RouteUsage, :Branded]))
    rename!(dfy_long, :value => :Y)
    dfy_long = @transform(dfy_long, :time = kron(1:20, fill(1, nrow(df))))
    select!(dfy_long, Not(:variable))

    dfx = @select(df, :bus_id, Between(:Odo1, :Odo20))
    dfx_long = DataFrames.stack(dfx, Not(:bus_id))
    rename!(dfx_long, :value => :Odometer)
    dfx_long = @transform(dfx_long, :time = kron(1:20, fill(1, nrow(df))))
    select!(dfx_long, Not(:variable))

    df_long = leftjoin(dfy_long, dfx_long, on = [:bus_id, :time])
    sort!(df_long, [:bus_id, :time])
    
    return df_long
end

# function for analysis
function run_analysis()
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df = load_and_preprocess_data(url)
    df_long = reshape_data(df)
    
    println("Dimensions of df_long: ", size(df_long))
    println("First few rows of df_long:")
    println(first(df_long, 5))
    
    return df_long
end

# Run the analysis
df_long = run_analysis()

#:::::::::::::::::::::::::::::::::::::::::
# Question 02
#:::::::::::::::::::::::::::::::::::::::::

# Function for flexible logit model
function estimate_flexible_logit(df_long)
    formula = @formula(Y ~ (Odometer + Odometer^2 + RouteUsage + RouteUsage^2 + Branded + time + time^2)^7)
    
    # Estimating the model
    model = glm(formula, df_long, Binomial(), LogitLink())
    
    return model
end

# Estimating the flexible logit model
flexible_logit_model = estimate_flexible_logit(df_long)

# Printing summary
println("Flexible Logit Model Summary:")
println(flexible_logit_model)

# for coefficients
println("\nModel Coefficients:")
println(coef(flexible_logit_model))

# for standard errors
println("\nStandard Errors:")
println(stderror(flexible_logit_model))

#:::::::::::::::::::::::::::::::::::::::::
# Question 03
#:::::::::::::::::::::::::::::::::::::::::

# Part (a): Constructing state transition matrices
function construct_state_transitions()
    zval, zbin, xval, xbin, xtran = create_grids()
    return zval, zbin, xval, xbin, xtran
end

# Part (b): Computing future value terms
function compute_future_values(flexible_logit_model, data_parms, zval, zbin, xval, xbin, xtran)
    state_df = DataFrame(
        Odometer = kron(ones(zbin), xval),
        RouteUsage = kron(zval, ones(xbin)),
        Branded = zeros(zbin * xbin),
        time = zeros(zbin * xbin)
    )
    
    FV = zeros(zbin * xbin, 2, data_parms.T + 1)
    
    for t in 2:data_parms.T
        for b in 0:1
            state_df.time .= t
            state_df.Branded .= b
            
            p0 = 1 .- predict(flexible_logit_model, state_df)
            FV[:, b+1, t] = -data_parms.β * log.(p0)
        end
    end
    
    FVT1 = zeros(data_parms.N, data_parms.T)
    for i in 1:data_parms.N
        for t in 1:data_parms.T
            row0 = (data_parms.Zstate[i] - 1) * xbin + 1
            row1 = data_parms.Xstate[i, t] + (data_parms.Zstate[i] - 1) * xbin
            FVT1[i, t] = (xtran[row1,:] .- xtran[row0,:])' * FV[row0:row0+xbin-1, data_parms.B[i]+1, t+1]
        end
    end
    
    return FVT1[:]
end

# Part (c): Estimating structural parameters
function estimate_structural_parameters(df_bus_long, fvt1)
    df_bus_long.fv = fvt1
    
    model = glm(@formula(Y ~ Odometer + Branded + offset(fv)), df_bus_long, Binomial(), LogitLink())
    
    return model
end

# Part (d): Custom binary logit estimation (optional)
function custom_binary_logit(X, y, fv)
    function logit_likelihood(β)
        Xβ = X * β .+ fv
        p = 1 ./ (1 .+ exp.(-Xβ))
        return -sum(y .* log.(p) .+ (1 .- y) .* log.(1 .- p))
    end
    
    initial_β = zeros(size(X, 2))
    result = optimize(logit_likelihood, initial_β, LBFGS())
    
    return Optim.minimizer(result)
end

# Part (e): Wrapper function
function allwrap()
    # Loading and preprocessing data
    url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
    df_bus = load_and_preprocess_data(url)
    df_bus_long = reshape_data(df_bus)
    
    # Estimating flexible logit model
    flexible_logit_model = estimate_flexible_logit(df_bus_long)
    
    # Preparing dynamic data
    Y = Matrix(df_bus[:,[:Y1,:Y2,:Y3,:Y4,:Y5,:Y6,:Y7,:Y8,:Y9,:Y10,:Y11,:Y12,:Y13,:Y14,:Y15,:Y16,:Y17,:Y18,:Y19,:Y20]])
    X = Matrix(df_bus[:,[:Odo1,:Odo2,:Odo3,:Odo4,:Odo5,:Odo6,:Odo7,:Odo8,:Odo9,:Odo10,:Odo11,:Odo12,:Odo13,:Odo14,:Odo15,:Odo16,:Odo17,:Odo18,:Odo19,:Odo20]])
    Z = Vector(df_bus[:,:RouteUsage])
    B = Vector(df_bus[:,:Branded])
    N = size(Y,1)
    T = size(Y,2)
    Xstate = Matrix(df_bus[:,[:Xst1,:Xst2,:Xst3,:Xst4,:Xst5,:Xst6,:Xst7,:Xst8,:Xst9,:Xst10,:Xst11,:Xst12,:Xst13,:Xst14,:Xst15,:Xst16,:Xst17,:Xst18,:Xst19,:Xst20]])
    Zstate = Vector(df_bus[:,:Zst])
    
    data_parms = (β = 0.9, Y = Y, B = B, N = N, T = T, X = X, Z = Z, Zstate = Zstate, Xstate = Xstate)
    
    # Part (a)
    zval, zbin, xval, xbin, xtran = construct_state_transitions()
    
    # Part (b)
    fvt1 = compute_future_values(flexible_logit_model, data_parms, zval, zbin, xval, xbin, xtran)
    
    # Part (c)
    structural_model = estimate_structural_parameters(df_bus_long, fvt1)
    
    println("Structural Model Summary:")
    println(structural_model)

    return structural_model
end

# Running the analysis
@time structural_model = allwrap()

#:::::::::::::::::::::::::::::::::::::::::
# Question 04
#:::::::::::::::::::::::::::::::::::::::::

# Unit tests
@testset "CCP Estimation Tests" begin
    # Test load_and_preprocess_data
    @testset "load_and_preprocess_data" begin
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2024/master/ProblemSets/PS5-ddc/busdataBeta0.csv"
        df_bus = load_and_preprocess_data(url)
        @test df_bus isa DataFrame
        @test size(df_bus, 2) > 0  # Check if dataframe has columns
        @test :bus_id in names(df_bus)
        # Check if bus_id is a sequence, allowing for different integer types and potential reordering
        @test Set(df_bus.bus_id) == Set(1:nrow(df_bus))
    end

    # Test reshape_data
    @testset "reshape_data" begin
        # Create a more comprehensive test dataframe
        test_df = DataFrame(
            bus_id = [1, 2],
            Y1 = [0, 1], Y2 = [1, 0], Y3 = [0, 1], Y4 = [1, 0],
            Odo1 = [100, 200], Odo2 = [150, 250], Odo3 = [200, 300], Odo4 = [250, 350],
            RouteUsage = [1, 2], Branded = [0, 1]
        )
        try
            df_long = reshape_data(test_df)
            @test df_long isa DataFrame
            @test size(df_long, 1) == 8  # 2 buses * 4 time periods
            @test :Y in names(df_long)
            @test :Odometer in names(df_long)
            @test :time in names(df_long)
        catch e
            @warn "reshape_data test failed" exception=(e, catch_backtrace())
            @test false
        end
    end

    # Test construct_state_transitions
    @testset "construct_state_transitions" begin
        result = construct_state_transitions()
        @test length(result) == 5
        zval, zbin, xval, xbin, xtran = result
        @test zval isa Vector
        @test xval isa Vector
        @test zbin isa Int
        @test xbin isa Int
        @test xtran isa Matrix
    end

    # Test compute_future_values
    @testset "compute_future_values" begin
        # Create mock data and model for testing
        mock_model = x -> DataFrame(pred = fill(0.5, size(x, 1)))  # Mock predict function returning a DataFrame
        mock_data = (β = 0.9, N = 10, T = 20, Zstate = ones(Int, 10), Xstate = ones(Int, 10, 20), B = zeros(Int, 10))
        zval, zbin, xval, xbin, xtran = construct_state_transitions()
        
        try
            fvt1 = compute_future_values(mock_model, mock_data, zval, zbin, xval, xbin, xtran)
            @test fvt1 isa Vector
            @test length(fvt1) == mock_data.N * mock_data.T
        catch e
            @warn "compute_future_values test failed" exception=(e, catch_backtrace())
            @test false
        end
    end

    # Test estimate_structural_parameters
    @testset "estimate_structural_parameters" begin
        # Create mock data for testing
        mock_df = DataFrame(Y = rand(0:1, 100), Odometer = rand(100), Branded = rand(0:1, 100))
        mock_fvt1 = rand(100)
        
        try
            model = estimate_structural_parameters(mock_df, mock_fvt1)
            @test model isa GLM.GeneralizedLinearModel
            @test coef(model) isa Vector
            @test length(coef(model)) == 3  # Intercept, Odometer, Branded
        catch e
            @warn "estimate_structural_parameters test failed" exception=(e, catch_backtrace())
            @test false
        end
    end

    # Test custom_binary_logit
    @testset "custom_binary_logit" begin
        X = rand(100, 2)
        y = rand(0:1, 100)
        fv = rand(100)
        
        β_hat = custom_binary_logit(X, y, fv)
        @test β_hat isa Vector
        @test length(β_hat) == size(X, 2)
    end
end

# Run the tests
Test.run()

# I tried running the code but it was taking too long to run. It is still showing "evaluating" in my laptop. I am not sure if the code is correct or not. I am so sorry. I hope this code is close to what you are looking for.
