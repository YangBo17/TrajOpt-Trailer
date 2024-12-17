# source code: https://shuvomoy.github.io/blogs/posts/Solving_semidefinite_programming_problems_in_Julia/
using COPT, Mosek, MosekTools 
using SCS, COSMO, ProxSDP, CSDP, SDPA
using BenchmarkTools

function random_mat_create(n)
    # this function creates a symmetric n×n matrix
    A = randn(n,n)
    A = A'*A
    A = (A+A')/2
    return A
end

n = 10
m = 20
# set of all data matrices A_i
# the data matrix A = [A1 A2 A3 ....]
A = zeros(n, m*n);
b = zeros(m);
# just ensuring our problem is feasible
X_test = rand(n,n)
X_test = X_test'*X_test
X_test = (X_test+X_test')/2
for i in 1:m
    A[:, (i-1)*n+1:i*n] .= random_mat_create(n);
    b[i] = tr(A[:, (i-1)*n+1:i*n]*X_test);
end
C = abs.(random_mat_create(n));

function solve_SDP(A, b, C; solver_name=:COSMO)
    # Create variable
    if solver_name == :COPT
        model = Model(COPT.ConeOptimizer)
    elseif solver_name == :Mosek
        model = Model(Mosek.Optimizer)
    elseif solver_name == :COSMO
        model = Model(COSMO.Optimizer)
    elseif solver_name == :SCS
        model = Model(SCS.Optimizer)
    elseif solver_name == :SDPA
        model = Model(SDPA.Optimizer)
    elseif solver_name == :ProxSDP
        model = Model(ProxSDP.Optimizer)
    elseif solver_name == :CSDP
        model = Model(CSDP.Optimizer)
    end

    set_silent(model)

    @variable(model, X[1:n, 1:n], PSD)

    @objective(model, Min, tr(C * X));
    for j in 1:m
        A_j = A[:, (j - 1) * n + 1:j * n]
        @constraint(model, tr(A_j * X) == b[j])
    end

    optimize!(model)

    status = JuMP.termination_status(model)
    X_sol = JuMP.value.(X)
    obj_value = JuMP.objective_value(model)

    return status, X_sol, obj_value
end

##
println("COPT")
@btime solve_SDP($A, $b, $C; solver_name=:COPT)


println("Mosek")
@btime solve_SDP($A, $b, $C; solver_name=:Mosek)

println("COSMO")
@btime solve_SDP($A, $b, $C; solver_name=:COSMO)

println("SCS")
@btime solve_SDP($A, $b, $C; solver_name=:SCS)

println("SDPA")
@btime solve_SDP($A, $b, $C; solver_name=:SDPA)

println("CSDP")
@btime solve_SDP($A, $b, $C; solver_name=:CSDP);

println("ProxSDP")
@btime solve_SDP($A, $b, $C; solver_name=:ProxSDP)


##
println("COPT")
@time status,_,_ = solve_SDP(A, b, C; solver_name=:COPT)
@show status

println("Mosek")
@time status,_,_ = solve_SDP(A, b, C; solver_name=:Mosek)
@show status

println("COSMO")
@time status,_,_ = solve_SDP(A, b, C; solver_name=:COSMO)
@show status

println("SCS")
@time status,_,_ = solve_SDP(A, b, C; solver_name=:SCS)
@show status

println("SDPA")
@time status,_,_ = solve_SDP(A, b, C; solver_name=:SDPA)
@show status

println("ProxSDP")
@time status,_,_ = solve_SDP(A, b, C; solver_name=:ProxSDP)
@show status

println("CSDP")
@time status,_,_ = solve_SDP(A, b, C; solver_name=:CSDP)
@show status

##
function solve_SDP2(A, b, C; solver_name=:COSMO)
    # Create variable
    if solver_name == :COPT
        model = Model(COPT.ConeOptimizer)
    elseif solver_name == :Mosek
        model = Model(Mosek.Optimizer)
    elseif solver_name == :COSMO
        model = Model(COSMO.Optimizer)
    elseif solver_name == :SCS
        model = Model(SCS.Optimizer)
    elseif solver_name == :SDPA
        model = Model(SDPA.Optimizer)
    elseif solver_name == :ProxSDP
        model = Model(ProxSDP.Optimizer)
    elseif solver_name == :CSDP
        model = Model(CSDP.Optimizer)
    end

    @variable(model, X[1:n, 1:n], PSD)

    @objective(model, Min, tr(C * X));
    for j in 1:m
        A_j = A[:, (j - 1) * n + 1:j * n]
        @constraint(model, tr(A_j * X) == b[j])
    end

    optimize!(model)

    status = JuMP.termination_status(model)
    X_sol = JuMP.value.(X)
    obj_value = JuMP.objective_value(model)

    return status, X_sol, obj_value
end

status,_,_ = solve_SDP2(A, b, C; solver_name=:COPT)

##
status,_,_ = solve_SDP2(A, b, C; solver_name=:Mosek)