
##
using LinearAlgebra
using JuMP, COPT, Mosek, MosekTools
using OSQP
using GLPK
using Test
using Random 
Random.seed!(123)

n = 10
m = 5

c = rand(n)
A = rand(m, n)
x_feasible = rand(n)
x_feasible = abs.(x_feasible)
# x_feasible[1] = 0.1
b = A * x_feasible

solver = optimizer_with_attributes(GLPK.Optimizer)
model = Model(solver)
@variable(model, x[1:n])
@objective(model, Min, c' * x)
@constraint(model, c1, A * x == b)
@constraint(model, c2, x >= 0)
JuMP.optimize!(model)
@show status = JuMP.termination_status(model)
x_val = value.(x)
c1_dual = dual(c1)

##
function LP(A, b, c)
    solver = optimizer_with_attributes(GLPK.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @objective(model, Min, c' * x)
    @constraint(model, A * x == b)
    @constraint(model, x .>= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_primal = value.(x)
    y_dual = dual.(c1)
    return status, opt_val, x_primal, y_dual
end

function LP(A, b, c, dA)
    A_new = A + dA
    solver = optimizer_with_attributes(GLPK.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @objective(model, Min, c' * x)
    @constraint(model, A_new * x == b)
    @constraint(model, x .>= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model) 
    x_primal = value.(x)
    y_dual = dual.(c1)
    return status, opt_val, x_primal, y_dual
end

status, opt0_val, x0_primal, y0_dual = LP(A, b, c)
dz_dA_AG = zeros(m,n)
for i in 1:m, j in 1:n
    dz_dA_AG[i,j] = - y0_dual[i] * x0_primal[j]
end
@show dz_dA_AG

dz_dA_FD = zeros(m,n)
for i in 1:m, j in 1:n
    dA = zeros(size(A))
    δ = 1e-3
    dA[i,j] = δ
    status, opt_val, x_primal, y_dual = LP(A, b, c, dA)
    dz_dA_FD[i,j] = (opt_val - opt0_val) / δ
end
@show dz_dA_FD


@show norm(dz_dA_AG - dz_dA_FD)
# @test dz_dA_AG ≈ dz_dA_FD atol=1e-3
dz_dA_AG ./ dz_dA_FD
##
n = 10
m = 5

C = rand(n,n)
C = C * C'
Alist = []
blist = []
X_random = rand(n, n)
X_feasible = X_random' * X_random
for k in 1:m
    A = rand(n,n)
    A = A * A'
    b = tr(A' * X_feasible)
    # b = rand()
    push!(Alist, A)
    push!(blist, b)
end

solver = optimizer_with_attributes(COPT.ConeOptimizer)
model = Model(solver)
@variable(model, X[1:n,1:n], PSD)
@objective(model, Min, sum(C[i,j] * X[j, i] for i in 1:n, j in 1:n))
@constraints(model, begin
    c1[k=1:m], sum(Alist[k][i, j] * X[j, i] for i in 1:n, j in 1:n) == blist[k]
end)
@constraint(model, X in PSDCone())
JuMP.optimize!(model)
@show status = JuMP.termination_status(model)
opt_val = objective_value(model)
X_primal = value.(X)
Y_dual = dual.(c1)

##
function SDP(Alist, blist, C)
    solver1 = optimizer_with_attributes(COPT.ConeOptimizer)
    model1 = Model(solver1)
    @variable(model1, X[1:n,1:n], PSD)
    @objective(model1, Min, sum(C[i,j] * X[i, j] for i in 1:n, j in 1:n))
    @constraints(model1, begin
        c1[k=1:m], sum(Alist[k][i, j] * X[i, j] for i in 1:n, j in 1:n) == blist[k]
    end)
    @constraint(model1, X in PSDCone())
    JuMP.optimize!(model1)
    status1 = JuMP.termination_status(model1)
    opt1_val = objective_value(model1)
    X1_primal = value.(X)
    y1_dual = dual.(c1)

    solver2 = optimizer_with_attributes(COPT.ConeOptimizer)
    model2 = Model(solver2)
    @variable(model2, y[1:m])
    @variable(model2, S[1:n,1:n], PSD)
    @objective(model2, Max, sum(blist[k] * y[k] for k in 1:m))
    @constraint(model2, c1, sum(Alist[k] * y[k] for k in 1:m) + S .== C)
    JuMP.optimize!(model2)
    status2 = JuMP.termination_status(model2)
    opt2_val = objective_value(model2)
    y2_primal = value.(y)
    S2_primal = value.(S)
    X2_dual = -dual.(c1)
    return status1, opt1_val, X1_primal, y1_dual, status2, opt2_val, y2_primal, X2_dual, S2_primal
end

function SDP(Alist, blist, C, dAlist)
    Alist_new = deepcopy(Alist)
    for k in 1:m
        Alist_new[k] += dAlist[k]
    end
    solver = optimizer_with_attributes(COPT.ConeOptimizer)
    model = Model(solver)
    @variable(model, X[1:n,1:n], PSD)
    @objective(model, Min, sum(C[j,i] * X[i, j] for i in 1:n, j in 1:n))
    @constraints(model, begin
        c1[k=1:m], sum(Alist_new[k][j, i] * X[i, j] for i in 1:n, j in 1:n) == blist[k]
    end)
    @constraint(model, X in PSDCone())
    JuMP.optimize!(model)
    @show status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    X_primal = value.(X)
    y_dual = dual.(c1)

    return status, opt_val, X_primal, y_dual
end


status1, opt1_val, X1_primal, y1_dual, status2, opt2_val, y2_primal, X2_dual, S2_primal = SDP(Alist, blist, C)
# @test X1_primal ≈ X2_dual atol=1e-5
@test y1_dual ≈ y2_primal atol=1e-3

dz_dA_AG = [-y1_dual[k] * X1_primal for k in 1:m]

##
dz_dA_FD = [zeros(n,n) for k in 1:m]
for i in 1:n, j in 1:n, k = 1:m
    dAlist = [zeros(n,n) for k in 1:m]
    δ = 1e-3
    dAlist[k][i,j] = δ
    # dAlist[k][j,i] = δ
    status, opt_val, X_primal, y_dual = SDP(Alist, blist, C, dAlist)
    dz_dA_FD[k][i,j] = (opt_val - opt1_val) / δ
end

@show norm(dz_dA_AG - dz_dA_FD)
# @test dz_dA_AG[1] ≈ dz_dA_FD[1] atol=1e-5

dz_dA_FD_sym = []
for k in 1:m
    dz_dA_FD_sym_temp = (dz_dA_FD[k] + dz_dA_FD[k]') / 2
    push!(dz_dA_FD_sym, dz_dA_FD_sym_temp)
end

# println("dz_dA_AG[1]")
# @show dz_dA_AG[1]

# println("dz_dA_FD[1]")
# dz_dA_FD[1]

# println("dz_dA_FD_sym[1]")
# dz_dA_FD_sym[1]

println("dz_dA_AG[1] ./ dz_dA_FD_sym[1]")
dz_dA_AG[1] ./ dz_dA_FD_sym[1]

## QP
n = 10
m1 = 5
m2 = 5
Q = rand(n,n)
Q = Q' * Q
c = rand(n)
A = rand(m1,n)
b = rand(m1)
G = rand(m2,n)
h = rand(m2)

solver = optimizer_with_attributes(OSQP.Optimizer)
model = Model(solver)
@variable(model, x[1:n])
@objective(model, Min, 0.5 * x' * Q * x + c' * x)
@constraint(model, u, A * x == b)
@constraint(model, v, G * x <= h)
JuMP.optimize!(model)
@show status = JuMP.termination_status(model)
x_primal = value.(x)
u_dual = dual(u)
v_dual = dual(v)

solver = optimizer_with_attributes(OSQP.Optimizer)
model = Model(solver)
@variable(model, u[1:m1])
@variable(model, v[1:m2])
@objective(model, Max, -0.5(c + A' * u + G' * v)' * Q^-1 * (c + A' * u + G' * v) - u' * b - v' * h)
@constraint(model, cons1, v .>= 0)
JuMP.optimize!(model)
@show status = JuMP.termination_status(model)
u_primal = value.(u)
v_primal = value.(v)

##
function QP(Q,c,A,b,G,h)
    solver = optimizer_with_attributes(OSQP.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @objective(model, Min, 0.5 * x' * Q * x + c' * x)
    @constraint(model, u, A * x == b)
    @constraint(model, v, G * x <= h)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_primal = value.(x)
    u_dual = dual(u)
    v_dual = dual(v)
    return status, opt_val, x_primal, u_dual, v_dual
end

function QP_dA(Q,c,A,b,G,h,dA)
    A_new = A + dA
    solver = optimizer_with_attributes(OSQP.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @objective(model, Min, 0.5 * x' * Q * x + c' * x)
    @constraint(model, u, A_new * x == b)
    @constraint(model, v, G * x <= h)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_primal = value.(x)
    u_dual = dual(u)
    v_dual = dual(v)
    return status, opt_val, x_primal, u_dual, v_dual
end

function QP_dG(Q,c,A,b,G,h,dG)
    G_new = G + dG
    solver = optimizer_with_attributes(OSQP.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @objective(model, Min, 0.5 * x' * Q * x + c' * x)
    @constraint(model, u, A * x == b)
    @constraint(model, v, G_new * x <= h)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_primal = value.(x)
    u_dual = dual(u)
    v_dual = dual(v)
    return status, opt_val, x_primal, u_dual, v_dual
end

status, opt0_val, x_primal, u_dual, v_dual = QP(Q,c,A,b,G,h)
dz_dA_AG = zeros(m1,n)
for i in 1:m1, j in 1:n
    dz_dA_AG[i,j] = -u_dual[i] * x_primal[j]
end
@show dz_dA_AG

dz_dA_FD = zeros(m1,n)
for i in 1:m1, j in 1:n
    dA = zeros(size(A))
    δ = 1e-3
    dA[i,j] = δ
    status, opt_val, x_primal, _, _ = QP_dA(Q,c,A,b,G,h,dA)
    dz_dA_FD[i,j] = (opt_val - opt0_val) / δ
end
@show dz_dA_FD

println("dz_dA_AG ./ dz_dA_FD")
dz_dA_AG ./ dz_dA_FD
##
status, opt0_val, x_primal, u_dual, v_dual = QP(Q,c,A,b,G,h)
dz_dG_AG = zeros(m2,n)
for i in 1:m2, j in 1:n
    dz_dG_AG[i,j] = -v_dual[i] * x_primal[j]
end
@show dz_dG_AG

dz_dG_FD = zeros(m2,n)
for i in 1:m2, j in 1:n
    dG = zeros(size(G))
    δ = 1e-3
    dG[i,j] = δ
    status, opt_val, x_primal, _, _ = QP_dG(Q,c,A,b,G,h,dG)
    dz_dG_FD[i,j] = (opt_val - opt0_val) / δ
end
@show dz_dG_FD

println("dz_dG_AG ./ dz_dG_FD")
dz_dG_AG ./ dz_dG_FD

##
function QP_s(Q,c,A,b,G,h)
    solver = optimizer_with_attributes(OSQP.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @variable(model, s[1:m2])
    @objective(model, Min, 0.5 * x' * Q * x + c' * x)
    @constraint(model, u, A * x == b)
    @constraint(model, v, G * x + s == h)
    @constraint(model, λ, s >= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_primal = value.(x)
    u_dual = dual(u)
    v_dual = dual(v)
    λ_dual = dual(λ)
    # λ_dual = 0
    return status, opt_val, x_primal, u_dual, v_dual, λ_dual
end

function QP_s_dA(Q,c,A,b,G,h,dA)
    A_new = A + dA
    solver = optimizer_with_attributes(OSQP.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @variable(model, s[1:m2])
    @objective(model, Min, 0.5 * x' * Q * x + c' * x)
    @constraint(model, u, A_new * x == b)
    @constraint(model, v, G * x + s == h)
    @constraint(model, s >= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_primal = value.(x)
    u_dual = dual(u)
    v_dual = dual(v)
    return status, opt_val, x_primal, u_dual, v_dual
end

function QP_s_dG(Q,c,A,b,G,h,dG)
    G_new = G + dG
    solver = optimizer_with_attributes(OSQP.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @variable(model, s[1:m2])
    @objective(model, Min, 0.5 * x' * Q * x + c' * x)
    @constraint(model, u, A * x == b)
    @constraint(model, v, G_new * x + s == h)
    @constraint(model, s >= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_primal = value.(x)
    u_dual = dual(u)
    v_dual = dual(v)
    return status, opt_val, x_primal, u_dual, v_dual
end

status, opt0_val, x_primal, u_dual, v_dual, λ_dual = QP_s(Q,c,A,b,G,h)
dz_dA_AG = zeros(m1,n)
for i in 1:m1, j in 1:n
    dz_dA_AG[i,j] = -u_dual[i] * x_primal[j]
end
@show dz_dA_AG

dz_dA_FD = zeros(m1,n)
for i in 1:m1, j in 1:n
    dA = zeros(size(A))
    δ = 1e-6
    dA[i,j] = δ
    status, opt_val, x_primal, _, _ = QP_s_dA(Q,c,A,b,G,h,dA)
    dz_dA_FD[i,j] = (opt_val - opt0_val) / δ
end
@show dz_dA_FD

println("dz_dA_AG ./ dz_dA_FD")
dz_dA_AG ./ dz_dA_FD

##
status, opt0_val, x_primal, u_dual, v_dual, λ_dual = QP_s(Q,c,A,b,G,h)
dz_dG_AG = zeros(m2,n)
for i in 1:m2, j in 1:n
    dz_dG_AG[i,j] = -v_dual[i] * x_primal[j]
end
@show dz_dG_AG

dz_dG_FD = zeros(m2,n)
for i in 1:m2, j in 1:n
    dG = zeros(size(G))
    δ = 1e-5
    dG[i,j] = δ
    status, opt_val, x_primal, _, _ = QP_s_dG(Q,c,A,b,G,h,dG)
    dz_dG_FD[i,j] = (opt_val - opt0_val) / δ
end
@show dz_dG_FD

println("dz_dG_AG ./ dz_dG_FD")
mat = dz_dG_AG ./ dz_dG_FD

