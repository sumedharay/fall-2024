Ransom PS7

using Random, LinearAlgebra, Distributions, Statistics, Optim, DataFrames, CSV, HTTP, GLM, FreqTables

function simulate_mlogit_data(N, K, J, beta)
    X = randn(N, K)
    X = hcat(ones(N), X)
    
    U = X * beta
    P = exp.(U) ./ sum(exp.(U), dims=2)
    
    Y = [argmax(rand(Multinomial(1, P[i,:]))) for i in 1:N]
    
    return X, Y
end
function problem_set_7()

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 1: GMM estimation of linear regression
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    function gmm_linear_regression()
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
        y = df.married.==1

        function moment_condition(beta, X, y)
            return X' * (y - X * beta)
        end

        function gmm_objective(beta, X, y, W)
            g = moment_condition(beta, X, y)
            return g' * W * g
        end

        W = I(size(X, 2))  # Identity matrix as weighting matrix
        beta_initial = rand(size(X, 2))
        
        result = optimize(b -> gmm_objective(b, X, y, W), beta_initial, BFGS())
        beta_gmm = Optim.minimizer(result)

        # Compare with OLS
        beta_ols = inv(X'X) * X'y

        println("GMM estimates: ", beta_gmm)
        println("OLS estimates: ", beta_ols)
    end

    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 2: Multinomial logit estimation
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    function multinomial_logit_estimation()
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        df = dropmissing(df, :occupation)
        df[df.occupation.>7, :occupation] .= 7
        X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
        y = df.occupation

        function mlogit(alpha, X, y)
            K = size(X,2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N,J)
            for j=1:J
                bigY[:,j] = y.==j
            end
            bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
            
            num = zeros(N,J)
            dem = zeros(N)
            for j=1:J
                num[:,j] = exp.(X*bigAlpha[:,j])
                dem .+= num[:,j]
            end
            
            P = num./repeat(dem,1,J)
            
            loglike = -sum(bigY.*log.(P))
            
            return loglike
        end

        # MLE estimation
        alpha_start = rand(6*size(X,2))
        alpha_hat_mle = optimize(a -> mlogit(a, X, y), alpha_start, BFGS()).minimizer

        # GMM estimation
        function gmm_mlogit(alpha, X, y)
            K = size(X,2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N,J)
            for j=1:J
                bigY[:,j] = y.==j
            end
            bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
            
            num = zeros(N,J)
            dem = zeros(N)
            for j=1:J
                num[:,j] = exp.(X*bigAlpha[:,j])
                dem .+= num[:,j]
            end
            
            P = num./repeat(dem,1,J)
            
            g = vec(bigY - P)
            
            return g
        end

        function gmm_objective(alpha, X, y, W)
            g = gmm_mlogit(alpha, X, y)
            return g' * W * g
        end

        W = I(size(X,1)*7)  # Identity matrix as weighting matrix

        # GMM with MLE starting values
        alpha_hat_gmm_mle = optimize(a -> gmm_objective(a, X, y, W), alpha_hat_mle, BFGS()).minimizer

        # GMM with random starting values
        alpha_hat_gmm_random = optimize(a -> gmm_objective(a, X, y, W), rand(6*size(X,2)), BFGS()).minimizer

        println("MLE estimates: ", alpha_hat_mle)
        println("GMM estimates (MLE start): ", alpha_hat_gmm_mle)
        println("GMM estimates (random start): ", alpha_hat_gmm_random)
    end
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 3: Simulate multinomial logit data
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    function simulate_mlogit_data(N, K, J, beta)
        X = randn(N, K)
        X = hcat(ones(N), X)
        
        U = X * beta
        P = exp.(U) ./ sum(exp.(U), dims=2)
        
        Y = [argmax(rand(Multinomial(1, P[i,:]))) for i in 1:N]
        
        return X, Y
    end
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 4: SMM example from slide #21
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    function smm_example()
        function simul(theta, n)
            # Ensure standard deviation is positive
            sigma = abs(theta[2])
            return rand(Normal(theta[1], sigma), n)
        end
    
        function auxstat(x)
            return [mean(x), var(x)]
        end
    
        function smm_objective(theta, x, S, W)
            n = length(x)
            m_data = auxstat(x)
            m_sim = zeros(length(m_data))
            
            for s in 1:S
                x_sim = simul(theta, n)
                m_sim += auxstat(x_sim)
            end
            m_sim /= S
            
            g = m_data - m_sim
            return g' * W * g
        end
    
        n = 1000
        S = 100
        true_theta = [1.2, 0.6]
        x = rand(Normal(true_theta[1], true_theta[2]), n)
        W = I(2)
        theta_initial = [1.0, 1.0]
    
        # Using Nelder-Mead method
        result = optimize(theta -> smm_objective(theta, x, S, W), theta_initial, NelderMead(), 
                          Optim.Options(iterations=10000, g_tol=1e-8, show_trace=true))
        
        if Optim.converged(result)
            theta_hat = Optim.minimizer(result)
            println("SMM estimates: ", theta_hat)
            println("True values: ", true_theta)
            println("Optimization result: ", Optim.summary(result))
        else
            println("Optimization did not converge. Consider trying different initial values or increasing the number of iterations.")
            println("Last iteration values: ", Optim.minimizer(result))
            println("Optimization result: ", Optim.summary(result))
        end
    end
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::
    # Question 5: SMM estimation of multinomial logit
    #::::::::::::::::::::::::::::::::::::::::::::::::::::::::

    function smm_mlogit()
        url = "https://raw.githubusercontent.com/OU-PhD-Econometrics/fall-2022/master/ProblemSets/PS1-julia-intro/nlsw88.csv"
        df = CSV.read(HTTP.get(url).body, DataFrame)
        df = dropmissing(df, :occupation)
        df[df.occupation.>7, :occupation] .= 7
        X = [ones(size(df,1),1) df.age df.race.==1 df.collgrad.==1]
        y = df.occupation

        function simul(theta, X)
            N, K = size(X)
            J = 7
            beta = reshape(theta, K, J-1)
            beta = [beta zeros(K)]
            
            U = X * beta
            P = exp.(U .- maximum(U, dims=2)) ./ sum(exp.(U .- maximum(U, dims=2)), dims=2)
            
            Y = zeros(Int, N)
            for i in 1:N
                if any(isnan.(P[i,:]))
                    Y[i] = rand(1:J)
                else
                    Y[i] = rand(Categorical(P[i,:]))
                end
            end
            
            return Y
        end

        function auxstat(y)
            return [mean(y .== j) for j in 1:7]
        end

        function smm_objective(theta, X, y, S, W)
            m_data = auxstat(y)
            m_sim = zeros(length(m_data))
            
            for s in 1:S
                y_sim = simul(theta, X)
                m_sim += auxstat(y_sim)
            end
            m_sim /= S
            
            g = m_data - m_sim
            return g' * W * g
        end

        n = size(X, 1)
        S = 100
        W = I(7)
        theta_initial = rand(24) .* 0.1  # Use smaller initial values

        result = optimize(theta -> smm_objective(theta, X, y, S, W), theta_initial, BFGS(), 
                          Optim.Options(iterations=1000, g_tol=1e-6, show_trace=true))
        theta_hat = Optim.minimizer(result)
        println("SMM estimates: ", theta_hat)
        println("Optimization result: ", Optim.summary(result))
    end

    # Running all questions
    println("Question 1: GMM Linear Regression")
    gmm_linear_regression()

    println("\nQuestion 2: Multinomial Logit Estimation")
    multinomial_logit_estimation()

    println("\nQuestion 3: Simulate Multinomial Logit Data")
    N, K, J = 1000, 4, 3
    beta = randn(K+1, J)
    X, Y = simulate_mlogit_data(N, K, J, beta)
    println("Data simulated. X shape: ", size(X), ", Y shape: ", size(Y))

    println("\nQuestion 4: SMM Example")
    smm_example()

    println("\nQuestion 5: SMM Multinomial Logit")
    smm_mlogit()
end

# Running all questions
problem_set_7()

#::::::::::::::::::::::::::::::::::::::::::::::::::::::::
# Question 7: Unit tests
#::::::::::::::::::::::::::::::::::::::::::::::::::::::::

using Test

@testset "Problem Set 7 Tests" begin
    @testset "Simulate Multinomial Logit Data" begin
        N, K, J = 1000, 4, 3
        beta = randn(K+1, J)
        X, Y = simulate_mlogit_data(N, K, J, beta)
        
        @test size(X) == (N, K+1)
        @test size(Y) == (N,)
        @test all(Y .<= J)
        @test all(Y .>= 1)
    end

    @testset "GMM Linear Regression" begin
        # Generating some test data
        N = 1000
        X = [ones(N) randn(N, 2)]
        beta_true = [1.0, 2.0, -1.5]
        y = X * beta_true + randn(N) * 0.1
        
        function moment_condition(beta, X, y)
            return X' * (y - X * beta)
        end

        function gmm_objective(beta, X, y, W)
            g = moment_condition(beta, X, y)
            return g' * W * g
        end

        W = I(size(X, 2))
        beta_initial = rand(size(X, 2))
        result = optimize(b -> gmm_objective(b, X, y, W), beta_initial, BFGS())
        beta_gmm = Optim.minimizer(result)
        
        @test isapprox(beta_gmm, beta_true, rtol=0.1)
    end

    @testset "Multinomial Logit Estimation" begin
        N, K, J = 1000, 4, 3
        beta_true = randn(K+1, J)
        X, Y = simulate_mlogit_data(N, K, J, beta_true)
        
        function mlogit(alpha, X, y)
            K = size(X,2)
            J = length(unique(y))
            N = length(y)
            bigY = zeros(N,J)
            for j=1:J
                bigY[:,j] = y.==j
            end
            bigAlpha = [reshape(alpha,K,J-1) zeros(K)]
            
            num = zeros(N,J)
            dem = zeros(N)
            for j=1:J
                num[:,j] = exp.(X*bigAlpha[:,j])
                dem .+= num[:,j]
            end
            
            P = num./repeat(dem,1,J)
            
            loglike = -sum(bigY.*log.(P))
            
            return loglike
        end
        
        alpha_start = rand((K+1)*(J-1))
        result = optimize(a -> mlogit(a, X, Y), alpha_start, BFGS())
        alpha_hat = Optim.minimizer(result)
        
        @test length(alpha_hat) == (K+1)*(J-1)
        @test mlogit(alpha_hat, X, Y) < mlogit(alpha_start, X, Y)
    end

    @testset "SMM Example" begin
        function simul(theta, n)
            return rand(Normal(theta[1], abs(theta[2])), n)
        end

        function auxstat(x)
            return [mean(x), var(x)]
        end

        function smm_objective(theta, x, S, W)
            n = length(x)
            m_data = auxstat(x)
            m_sim = zeros(length(m_data))
            
            for s in 1:S
                x_sim = simul(theta, n)
                m_sim += auxstat(x_sim)
            end
            m_sim /= S
            
            g = m_data - m_sim
            return g' * W * g
        end

        n = 1000
        S = 100
        true_theta = [1.2, 0.6]
        x = rand(Normal(true_theta[1], true_theta[2]), n)
        W = I(2)
        theta_initial = [1.0, 1.0]

        result = optimize(theta -> smm_objective(theta, x, S, W), theta_initial, NelderMead(), 
                          Optim.Options(iterations=1000, g_tol=1e-6))
        
        theta_hat = Optim.minimizer(result)
        
        @test isapprox(theta_hat, true_theta, rtol=0.1)
        @test Optim.converged(result)
    end
end
