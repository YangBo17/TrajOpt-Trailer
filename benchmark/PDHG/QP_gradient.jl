using LinearAlgebra
using JuMP, Mosek, MosekTools
using OSQP
using Test
using Random
# Random.seed!(0)

n = 2
m = 1
Q = rand(n,n)
Q = Q' * Q
c = rand(n)
G = rand(m,n)
h = rand(m)

solver = optimizer_with_attributes(OSQP.Optimizer)
model = Model(solver)
@variable(model, x[1:n])
@objective(model, Min, 0.5 * x' * Q * x + c' * x)
@constraint(model, v, G * x <= h)
JuMP.optimize!(model)
@show status = JuMP.termination_status(model)
x_primal = value.(x)
v_dual = dual(v)

solver = optimizer_with_attributes(OSQP.Optimizer)
model = Model(solver)
@variable(model, v[1:m])
@objective(model, Max, -0.5(c  + G' * v)' * Q^-1 * (c + G' * v) - v' * h)
@constraint(model, cons1, v .>= 0)
JuMP.optimize!(model)
@show status = JuMP.termination_status(model)
v_primal = value.(v)

#
function QP_s(Q,c,G,h)
    solver = optimizer_with_attributes(OSQP.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @variable(model, s[1:m])
    @objective(model, Min, 0.5 * x' * Q * x + c' * x)
    @constraint(model, v, G * x + s == h)
    @constraint(model, λ, s >= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_val = value.(x)
    s_val = value.(s)
    v_val = dual(v)
    λ_val = dual(λ)
    return status, opt_val, x_val, s_val, v_val, λ_val
end

function QP_s_dG(Q,c,G,h,dG)
    G_new = G + dG
    solver = optimizer_with_attributes(OSQP.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @variable(model, s[1:m])
    @objective(model, Min, 0.5 * x' * Q * x + c' * x)
    @constraint(model, v, G_new * x + s == h)
    @constraint(model, λ, s >= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_val = value.(x)
    s_val = value.(s)
    v_val = dual(v)
    λ_val = dual(λ)
    return status, opt_val, x_val, s_val, v_val, λ_val
end

status, opt_val0, x0, s0, v0, λ0 = QP_s(Q,c,G,h)
dz_dG_AG = zeros(m,n)
for i in 1:m, j in 1:n
    dz_dG_AG[i,j] = -v0[i] * x0[j]
end

dz_dG_FD = zeros(m,n)
for i in 1:m, j in 1:n
    dG = zeros(size(G))
    δ = 1e-2
    dG[i,j] = δ
    status, opt_val, x, s, v, λ = QP_s_dG(Q,c,G,h,dG)
    dz_dG_FD[i,j] = (opt_val - opt_val0) / δ
end


@show G * x0 + s0 - h
@show v0
@show λ0
@show dz_dG_AG
@show dz_dG_FD
@show dz_dG_AG ./ dz_dG_FD;

##
n = 2
m1 = 1
m2 = 1
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

#
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
    x_val = value.(x)
    s_val = value.(s)
    u_val = dual(u)
    v_val = dual(v)
    λ_val = dual(λ)
    return status, opt_val, x_val, s_val, u_val, v_val, λ_val
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
    @constraint(model, λ, s >= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_val = value.(x)
    s_val = value.(s)
    u_val = dual(u)
    v_val = dual(v)
    λ_val = dual(λ)
    return status, opt_val, x_val, s_val, u_val, v_val, λ_val
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
    @constraint(model, λ, s >= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_val = value.(x)
    s_val = value.(s)
    u_val = dual(u)
    v_val = dual(v)
    λ_val = dual(λ)
    return status, opt_val, x_val, s_val, u_val, v_val, λ_val
end

function QP_s_dQ(Q,c,A,b,G,h,dQ)
    Q_new = Q + dQ
    Q_new = (Q_new' + Q_new) / 2
    solver = optimizer_with_attributes(OSQP.Optimizer)
    model = Model(solver)
    @variable(model, x[1:n])
    @variable(model, s[1:m2])
    @objective(model, Min, 0.5 * x' * Q_new * x + c' * x)
    @constraint(model, u, A * x == b)
    @constraint(model, v, G * x + s == h)
    @constraint(model, λ, s >= 0)
    JuMP.optimize!(model)
    status = JuMP.termination_status(model)
    opt_val = objective_value(model)
    x_val = value.(x)
    s_val = value.(s)
    u_val = dual(u)
    v_val = dual(v)
    λ_val = dual(λ)
    return status, opt_val, x_val, s_val, u_val, v_val, λ_val
end

status, opt_val0, x0, s0, u0, v0, λ0 = QP_s(Q,c,A,b,G,h)
dz_dA_AG = zeros(m1,n)
for i in 1:m1, j in 1:n
    dz_dA_AG[i,j] = -u0[i] * x0[j]
end

dz_dA_FD = zeros(m1,n)
for i in 1:m1, j in 1:n
    dA = zeros(size(A))
    δ = 1e-2
    dA[i,j] = δ
    status, opt_val, x, s, v, λ = QP_s_dA(Q,c,A,b,G,h,dA)
    dz_dA_FD[i,j] = (opt_val - opt_val0) / δ
end

@show A * x0 - b
@show G * x0 + s0 - h
@show u0
@show v0
@show λ0
@show dz_dA_AG
@show dz_dA_FD
@show dz_dA_AG ./ dz_dA_FD;


##
status, opt_val0, x0, s0, u0, v0, λ0 = QP_s(Q,c,A,b,G,h)
dz_dQ_AG = zeros(m1,n)
for i in 1:m1, j in 1:n
    dz_dQ_AG[i,j] = -u0[i] * x0[j]
end

dz_dQ_FD = zeros(m1,n)
for i in 1:m1, j in 1:n
    dA = zeros(size(A))
    δ = 1e-2
    dA[i,j] = δ
    status, opt_val, x, s, v, λ = QP_s_dA(Q,c,A,b,G,h,dA)
    dz_dA_FD[i,j] = (opt_val - opt_val0) / δ
end

@show A * x0 - b
@show G * x0 + s0 - h
@show u0
@show v0
@show λ0
@show dz_dA_AG
@show dz_dA_FD
@show dz_dA_AG ./ dz_dA_FD;