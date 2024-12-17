#=
SDP formulation of TrajOpt for car-like robot, using SDP solvers: COPT, Mosek, etc。
=#
using DynamicPolynomials
using SumOfSquares
using JuMP
using COPT
using Mosek, MosekTools
using SCS, COSMO, ProxSDP, CSDP, SDPA
using OSQP
using LinearAlgebra
using Plots
using CSV, DataFrames

#=
Curve driving, the environment is same as that of Car experiments 
=#
# env and car settings
include("utils/env_car.jl") 

# plot_car_trans!(fig_env, car_raw, cons_raw)

# trajopt

trajopt_set = TrajOpt_Set(d=7, M=2, N=seg_N)

L1 = maximum([length(obs_list[i].B) for i in eachindex(obs_list)])
L2 = maximum([length(limits_list[i].B) for i in eachindex(limits_list)])

@polyvar t  
# solver = optimizer_with_attributes(Ours.Optimizer)
solver = optimizer_with_attributes(OSQP.Optimizer)
solver = optimizer_with_attributes(COPT.Optimizer)
solver = optimizer_with_attributes(Mosek.Optimizer)
solver = optimizer_with_attributes(COSMO.Optimizer)
solver = optimizer_with_attributes(SCS.Optimizer)


iter = 10
function trajopt_space(T0::Vector{Float64}, trajopt_set::TrajOpt_Set, cons_init::Cons_Init, limits_list::Vector, obs_list::Vector; dt::Float64=0.02)
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
            push!(obj, sum([(D2p[i,j](t=>tk))^2 * dt + (D2p[i,j](t=>tk))^2 * dt for tk in collect(0:dt:T0[j])]))
        end
    end
    @objective(model, Min, sum(obj))

    # equality constraint
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    @constraint(model, p[1,1](t=>0.0) == start[1])
    @constraint(model, p[2,1](t=>0.0) == start[2])
    @constraint(model, p[1,end](t=>T0[end]) == goal[1])
    @constraint(model, p[2,end](t=>T0[end]) == goal[2])
    @constraint(model, D1p[1,1](t=>0.0) == start[4]*cos(start[3]))
    @constraint(model, D1p[2,1](t=>0.0) == start[4]*sin(start[3]))
    @constraint(model, D1p[1,end](t=>T0[end]) == goal[4]*cos(goal[3]))
    @constraint(model, D1p[2,end](t=>T0[end]) == goal[4]*sin(goal[3]))
    @constraint(model, D2p[1,end](t=>T0[end]) == cons_init.acc[end,1])
    @constraint(model, D2p[2,end](t=>T0[end]) == cons_init.acc[end,2])

    # continuous
    for i in 1:N-1
        @constraint(model, p[1,i](t=>T0[i]) == p[1,i+1](t=>0.0))
        @constraint(model, p[2,i](t=>T0[i]) == p[2,i+1](t=>0.0))
        @constraint(model, D1p[1,i](t=>T0[i]) == D1p[1,i+1](t=>0.0))
        @constraint(model, D1p[2,i](t=>T0[i]) == D1p[2,i+1](t=>0.0))
        @constraint(model, D2p[1,i](t=>T0[i]) == D2p[1,i+1](t=>0.0))
        @constraint(model, D2p[2,i](t=>T0[i]) == D2p[2,i+1](t=>0.0))
    end

    # inequality constraint
    # space 
    for i in 1:N
        X = [p[1,i], p[2,i]]
        for j in 1:L1
            for tk in collect(0:dt:T0[i])
                X = [p[1,i](t=>tk), p[2,i](t=>tk)]
                @constraint(model, obs_list[i].B[j] - obs_list[i].A[j,:]'*X >= 0.0)
            end
        end
    end

    # dynamic limit
    for i in 1:N
        X = [D1p[1,i], D1p[2,i], D2p[1,i], D2p[2,i]]
        for j in 1:L2
            for tk in collect(0:dt:T0[i])
                X = [D1p[1,i](t=>tk), D1p[2,i](t=>tk), D2p[1,i](t=>tk), D2p[2,i](t=>tk)]
                @constraint(model, limits_list[i].B[j] - limits_list[i].A[j,:]'*X >= 0.0)
            end
        end
    end

    JuMP.optimize!(model)
    return value.(p), model
end

function trajopt_time(T0::Vector{Float64}, trajopt_set::TrajOpt_Set, cons_init::Cons_Init, limits_list::Vector, obs_list::Vector)
    function get_gradient_fd(obj0::Float64, T0::Vector{Float64}, h=1e-6)
        grad = zeros(size(T0))
        origin_time = T0
        for i in eachindex(T0)
            try_time = copy(origin_time)
            try_time[i] += h
            _, mdl = trajopt_space(try_time, trajopt_set, cons_init, limits_list, obs_list)
            obj = objective_value(mdl)
            grad[i] = (obj - obj0) / h
        end
        grad += tweight 
        return grad
    end
    
    function get_gradient_mellinger(obj0::Float64, T0::Vector{Float64}, h=1e-6)
        grad = zeros(size(T0))
        origin_time = T0
        m = length(T0)
        for i in eachindex(T0)
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
                grad = get_gradient_fd(obj0, T0, h)
            elseif grad_method == :mel 
                grad = get_gradient_mellinger(obj0, T0, h)
            else
                println("No this grad method")
            end
    
            ∇F = grad / norm(grad)
            ΔT = - ∇F
            # use a maximum alpha that makes sure time are always positive
            alpha_max = maximum(- T0 ./ ΔT) - 1e-6
            @show alpha_max
            if alpha_max > 0
                alpha = min(alpha_max, alpha0)
            else
                alpha = alpha0
            end
            # find alpha
            candid_time = T0
            objf = obj0
            objft = obj0t
            p = 0
            for j in 1:j_iter
                candid_time = T0 + alpha * ΔT
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
            T0 = candid_time
            push!(Times, T0)
            push!(Objs, obj0)
            push!(Objts, obj0t)
            push!(ps, p)
        end
        return Times, Objs, Objts, ps
    end

    grad_method = :mel
    _, mdl = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list)
    obj0 = objective_value(mdl)
    obj0t = obj0 + sum(tweight.*T0)
    Times, Objs, Objts, ps = backtracking_line_search(max_iter=max_iter,grad_method=grad_method)
    return Times, Objs, Objts, ps
end

p0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list; dt=0.1)
solve_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
obj_val = objective_value(model)
@show status = JuMP.termination_status(model)
@show solve_time = round(solve_time, digits=3)
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
traj = Traj_car(t, x, y, ψ, v, δ, a)

fig_opt = plot(fig_env, x, y, label="")

function visual_traj(fig_env, traj::Traj_car; method::String, c_traj::Symbol, c_car::Symbol, num::Int64, legend_loc::Symbol)
    N = length(traj.δ)
    interval_frame = Int64(floor(N / (num+1)))
    index_frame = [1 + i * interval_frame for i in 0:num]
    push!(index_frame, N)
    fig_traj = plot(fig_env, traj.x, traj.y, legend = legend_loc, color = c_traj, label=method, linewidth=2, aspect_ratio=1)
    for i in index_frame
        car_s = [traj.x[i], traj.y[i], traj.ψ[i], traj.δ[i]]
        car_shapes = depict_car(Body(), car_s)
        render_car!(fig_traj, car_shapes; c=c_car)
    end
    return fig_traj
end

num = 3
fig_traj_opt = visual_traj(fig_opt, traj, method="Traj", c_traj=:darkorange, c_car=:darkorange, num=num, legend_loc=:topright)