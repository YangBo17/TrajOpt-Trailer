#=
SDP formulation of TrajOpt for car-like robot, using SDP solvers: COPT, Mosek, etc。
=#
using Revise
using DynamicPolynomials
using SumOfSquares
using JuMP
using COPT
using Mosek, MosekTools
using SCS, COSMO, ProxSDP, CSDP
using OSQP
using LinearAlgebra
using Plots
using CSV, DataFrames

#=
Curve driving, the environment is same as that of Car experiments 
=#
# env and car settings
include("utils/env_car2.jl") 

# trajopt

trajopt_set = TrajOpt_Set(d=7, M=2, N=sdp_N)
@polyvar t 
# solver = optimizer_with_attributes(Ours.Optimizer)
# solver = optimizer_with_attributes(COPT.ConeOptimizer)
# solver = optimizer_with_attributes(Mosek.Optimizer)
# solver = optimizer_with_attributes(SCS.Optimizer)
# solver = optimizer_with_attributes(COSMO.Optimizer)
# solver = optimizer_with_attributes(CSDP.Optimizer)
# solver = optimizer_with_attributes(ProxSDP.Optimizer)
solver = optimizer_with_attributes(OSQP.Optimizer)

iter = 10
function trajopt_space(Time::Vector{Float64}, trajopt_set::TrajOpt_Set, cons_init::Cons_Init, limits_list::Vector, obs_list::Vector; dt::Float64=0.02)
    model = Model(solver)
    d = trajopt_set.d 
    M = trajopt_set.M 
    N = trajopt_set.N
    @variable(model, coef[i=1:(d+1),j=1:M,k=1:N])
    poly_basis = monomials([t], 0:d)
    p = [coef[:,i,j]'*poly_basis for i in 1:M, j in 1:N]

    # objective
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    obj = []
    for i in 1:M
        for j in 1:N
            push!(obj, sum([(D1p[i,j](t=>tk))^2 * dt + (D1p[i,j](t=>tk))^2 * dt for tk in collect(0:dt:Time[j])]))
        end
    end
    @objective(model, Min, sum(obj))

    # equality constraint
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    @constraint(model, p[1,1](t=>0.0) == start[1])
    @constraint(model, p[2,1](t=>0.0) == start[2])
    @constraint(model, D1p[1,1](t=>0.0) == start[4]*cos(start[3]))
    @constraint(model, D1p[2,1](t=>0.0) == start[4]*sin(start[3]))
    # @constraint(model, D2p[1,1](t=>0.0) == 0.1)
    # @constraint(model, D2p[2,1](t=>0.0) == 0.0)
    @constraint(model, p[1,end](t=>Time[end]) == goal[1])
    @constraint(model, p[2,end](t=>Time[end]) == goal[2])
    @constraint(model, D1p[1,end](t=>Time[end]) == goal[4]*cos(goal[3]))
    @constraint(model, D1p[2,end](t=>Time[end]) == goal[4]*sin(goal[3]))
    # @constraint(model, D2p[1,end](t=>Time[end]) == cons_init.acc[2,1])
    # @constraint(model, D2p[2,end](t=>Time[end]) == cons_init.acc[2,2])

    # continuous
    for i in 1:N-1
        @constraint(model, p[1,i](t=>Time[i]) == p[1,i+1](t=>0.0))
        @constraint(model, p[2,i](t=>Time[i]) == p[2,i+1](t=>0.0))
        @constraint(model, D1p[1,i](t=>Time[i]) == D1p[1,i+1](t=>0.0))
        @constraint(model, D1p[2,i](t=>Time[i]) == D1p[2,i+1](t=>0.0))
        @constraint(model, D2p[1,i](t=>Time[i]) == D2p[1,i+1](t=>0.0))
        @constraint(model, D2p[2,i](t=>Time[i]) == D2p[2,i+1](t=>0.0))
    end

    # inequality constraint
    # space 
    for i in 1:N
        X = [p[1,i], p[2,i]]
        for j in 1:L1
            for tk in collect(0:dt:Time[i])
                X = [p[1,i](t=>tk), p[2,i](t=>tk)]
                @constraint(model, obs_list[i].B[j] - obs_list[i].A[j,:]'*X >= 0.0)
            end
        end
    end

    # dynamic limit
    for i in 1:N
        X = [D1p[1,i], D1p[2,i], D2p[1,i], D2p[2,i]]
        for j in 1:L2
            for tk in collect(0:dt:Time[i])
                X = [D1p[1,i](t=>tk), D1p[2,i](t=>tk), D2p[1,i](t=>tk), D2p[2,i](t=>tk)]
                @constraint(model, limits_list[i].B[j] - limits_list[i].A[j,:]'*X >= 0.0)
            end
        end
    end

    JuMP.optimize!(model)
    return value.(p), model
end

function trajopt_time(Time0::Vector{Float64}, trajopt_set::TrajOpt_Set, cons_init::Cons_Init, limits_list::Vector, obs_list::Vector)
    function get_gradient_fd(obj0::Float64, Time0::Vector{Float64}, h=1e-6)
        grad = zeros(size(Time0))
        origin_time = Time0
        for i in eachindex(Time0)
            try_time = copy(origin_time)
            try_time[i] += h
            _, mdl = trajopt_space(try_time, trajopt_set, cons_init, limits_list, obs_list)
            obj = objective_value(mdl)
            grad[i] = (obj - obj0) / h
        end
        grad += tweight 
        return grad
    end
    
    function get_gradient_mellinger(obj0::Float64, Time0::Vector{Float64}, h=1e-6)
        grad = zeros(size(Time0))
        origin_time = Time0
        m = length(Time0)
        for i in eachindex(Time0)
            gi = -1/(m-1) * ones(m)
            gi[i] = 1
            try_time = copy(origin_time)
            try_time += h * gi
            _, mdl = trajopt_space(try_time, trajopt_set, cons_init, limits_list, obs_list)
            obj = objective_value(mdl)
            grad[i] = (obj - obj0) / h
        end
        grad += tweight 
        return grad
    end
    
    function backtracking_line_search(;alpha0::Float64=0.175, h::Float64=1e-6, c::Float64=0.2, tau::Float64=0.2, max_iter::Int64=20, j_iter::Int64=5, grad_method::Symbol=:fd)
        ```
        Parameters
        ----------
        alpha0: initial step length, 0.175 and 0.375 are found to be very good
        h: step size for finding gradient using forward differentiation
        c: the objective decrease parameter 
        tau: the step length shrink parameter
        max_iter: maximum iteration for gradient descent
        j_iter: maximum iteration for finding alpha 
        abs_tol: absolute objective tolerance
        rel_tol: relative objective tolerance
    
        Returns
        -------
        ```
        Times = []
        Objs = []
        Objts = []
        ps = []
        for i in 1:max_iter
            if grad_method == :fd
                grad = get_gradient_fd(obj0, Time0, h)
            elseif grad_method == :mel 
                grad = get_gradient_mellinger(obj0, Time0, h)
            else
                println("No this grad method")
            end
    
            ∇F = grad / norm(grad)
            ΔT = - ∇F
            # use a maximum alpha that makes sure time are always positive
            alpha_max = maximum(- Time0 ./ ΔT) - 1e-6
            @show alpha_max
            if alpha_max > 0
                alpha = min(alpha_max, alpha0)
            else
                alpha = alpha0
            end
            # find alpha
            candid_time = Time0
            objf = obj0
            objft = obj0t
            p = 0
            for j in 1:j_iter
                candid_time = Time0 + alpha * ΔT
                p, mdl = trajopt_space(candid_time, trajopt_set, cons_init, limits_list, obs_list)
                objf = objective_value(mdl)
                objft = objf + sum(tweight.*candid_time)
                if objf < 0
                    alpha = tau * alpha
                    continue
                end
                if obj0t - objft >= - alpha * c * ( ∇F'*ΔT ) || obj0t - objft >= 0.1 * obj0 # either backtrack or decrease sufficiently
                    break
                else
                    alpha = tau * alpha
                end
            end
            obj0 = objf
            obj0t = objft
            Time0 = candid_time
            push!(Times, Time0)
            push!(Objs, obj0)
            push!(Objts, obj0t)
            push!(ps, p)
        end
        return Times, Objs, Objts, ps
    end

    grad_method = :mel
    _, mdl = trajopt_space(Time0, trajopt_set, cons_init, limits_list, obs_list)
    obj0 = objective_value(mdl)
    obj0t = obj0 + sum(tweight.*Time0)
    Times, Objs, Objts, ps = backtracking_line_search(max_iter=max_iter,grad_method=grad_method)
    return Times, Objs, Objts, ps
end

p0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list)
sol_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
obj_val = objective_value(model)
@show sol_time = round(sol_time, digits=3)
@show obj_val

function process_data(p::Matrix{Polynomial{true, Float64}}, Time::Vector{Float64}, car_body::Body)
    L, W = car_body.L, car_body.W
    data = []
    total_time = [0.0]
    dt = 0.01
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    L = length(p[1,:])
    for n in 1:L
        if n != L
            ts = 0:dt:Time[n]-dt |> collect
        else
            ts = 0:dt:Time[n] |> collect
        end
        push!(total_time, total_time[end]+Time[n])
        for i in eachindex(ts)
            ti = ts[i] + total_time[n]
            x = p[1,n](t=>ts[i])
            y = p[2,n](t=>ts[i])
            vx = D1p[1,n](t=>ts[i])
            vy = D1p[2,n](t=>ts[i])
            v = sqrt(vx^2+vy^2)
            ax = D2p[1,n](t=>ts[i])
            ay = D2p[2,n](t=>ts[i])
            a = sqrt(ax^2+ay^2)
            θ = atan(vy,vx)
            δ = atan(L*(ay*vx-ax*vy),v^3)
            κ = (ay*vx-vy*ax)/v^3
            
            datapoint = CarState(ti,x,y,θ,vx,vy,ax,ay,v,a,δ,κ)
            push!(data, datapoint)
        end
    end
    return data
end

data0 = process_data(p0, T0, car_body)
data = data0

t = [data[i].t for i in eachindex(data)];
x = [data[i].x for i in eachindex(data)];
y = [data[i].y for i in eachindex(data)];  
ψ = [data[i].θ for i in eachindex(data)];
v = [data[i].v for i in eachindex(data)];
δ = [data[i].δ for i in eachindex(data)];
a = [data[i].a for i in eachindex(data)];
fig_opt1 = plot(fig_env, x, y)

##
traj = Traj_car(t, x, y, ψ, v, δ, a)
using JLD2
using CSV, DataFrames
@save "benchmark/data/Ours2024S_car_traj.jld2" traj

method_name = "Ours24S"

df = CSV.read("benchmark/data/sol_time_car.csv", DataFrame)

row_index = findfirst(row -> row.Method == method_name, eachrow(df))
if row_index !== nothing
    # 如果存在，覆盖该行
    df[row_index, :SolTime] = sol_time
else
    # 如果不存在，添加新行
    new_row = DataFrame(Method = [method_name], SolTime = [sol_time])
    append!(df, new_row)
end

CSV.write("benchmark/data/sol_time_car.csv", df)