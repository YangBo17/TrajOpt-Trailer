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
using LinearAlgebra
using Plots
using CSV, DataFrames
using TensorOperations

#=
Curve driving, the environment is same as that of Car experiments 
=#
# env and car settings
include("PDHG/get_PDHG_para_car.jl") 

# plot_car_trans!(fig_env, car_raw, cons_raw)

# trajopt

trajopt_set = TrajOpt_Set(d=d, M=m, N=seg_N)

L1 = maximum([length(obs_list[i].B) for i in eachindex(obs_list)])
L2 = maximum([length(limits_list[i].B) for i in eachindex(limits_list)])

@polyvar t  
# solver = optimizer_with_attributes(Ours.Optimizer)
solver = optimizer_with_attributes(COPT.ConeOptimizer)
# solver = optimizer_with_attributes(Mosek.Optimizer)
# solver = optimizer_with_attributes(Mosek.Optimizer,"MSK_IPAR_INTPNT_MAX_ITERATIONS" => 100,        "MSK_DPAR_INTPNT_TOL_REL_GAP" => 1e-12)
solver = optimizer_with_attributes(SCS.Optimizer)
solver = optimizer_with_attributes(COSMO.Optimizer)
# solver = optimizer_with_attributes(CSDP.Optimizer)
solver = optimizer_with_attributes(ProxSDP.Optimizer)

#
function trajopt_space(T::Vector{Float64}, trajopt_set::TrajOpt_Set, cons_init::Cons_Init, limits_list::Vector, obs_list::Vector)
    # set_optimizer_attribute(model, "TimeLimit", 0.3)
    # set_optimizer_attribute(model, "MSK_IPAR_INTPNT_MAX_ITERATIONS", iter)
    d = trajopt_set.d 
    M = trajopt_set.M 
    N = trajopt_set.N
    d2 = Int((d-1)/2+1)
    Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_PHLR(T)
    PL, PV = eigen(Popt)
    PLhalf=broadcast(sqrt,PL[size(Popt,1)-rank(Popt)+1:end])
    Pt=diagm(PLhalf)*PV'[size(Popt,1)-rank(Popt)+1:end,:]
    @show maximum(broadcast(abs,Pt'*Pt-Popt))

    model = Model(solver)
    # set_silent(model)
    @variable(model, c[i=1:(d+1),j=1:M,k=1:N])
    @variable(model, Xf[1:d2,1:d2,1:K*N])
    @variable(model, Xg[1:d2,1:d2,1:K*N])
    @constraint(model, [i=1:K*N], Xf[:,:,i] in PSDCone())
    @constraint(model, [i=1:K*N], Xg[:,:,i] in PSDCone())
    c_vec = reshape(c, (d+1)*M*N)
    X = cat(Xf, Xg, dims=3)
    sopt_large = Array{AffExpr}(undef, 2*(d+1), 2*K*N)
    for i = 1:size(FG, 3), j = 1:size(X, 3), a = 1:size(FG, 1), b = 1:size(FG, 2)
        sopt_large[i, j] = FG[a, b, i] * X[b, a, j]
    end
    sopt = sopt_large[1:d+1, 1:K*N] + sopt_large[d+2:end, K*N+1:end]
    sopt = reshape(sopt, K*N*(d+1))

    @show size(c_vec)
    @constraint(model, Hopt*c_vec .== ropt)
    @constraint(model, Lopt*c_vec .== (gopt + sopt))

    @variable(model, schur >= 0)
    obj_dim=size(Pt,1)
    I_objdn=Matrix(1.0I,obj_dim, obj_dim)
    @constraint(model, [I_objdn (Pt*c_vec); (Pt*c_vec)' schur] in PSDCone())

    @objective(model, Min, schur)

    JuMP.optimize!(model)
    return value.(c), model
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

T0 = ones(seg_N) * 1.0
c0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list)
solve_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
obj_val = objective_value(model)
@show solve_time = round(solve_time, digits=3)
@show obj_val

#

function calc_traj(T, cstar; dt=0.02)
    cstar = reshape(cstar, d+1, m, N)
    poly_basis = monomials([t], 0:d)
    p = [cstar[:,i,j]' * poly_basis for i in 1:m, j in 1:N]
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    data = []
    total_time = [0.0]
    for j in 1:N 
        if j != N 
            ts = 0:dt:T[j]-dt |> collect
        else
            ts = 0:dt:T[j] |> collect
        end
        push!(total_time, total_time[end] + T[j])
        for i in eachindex(ts)
            ti = ts[i] + total_time[j]
            x = p[1,j](t=>ts[i])
            y = p[2,j](t=>ts[i])
            vx = D1p[1,j](t=>ts[i])
            vy = D1p[2,j](t=>ts[i])
            v = sqrt(vx^2+vy^2)
            ax = D2p[1,j](t=>ts[i])
            ay = D2p[2,j](t=>ts[i])
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

data = calc_traj(T0, c0; dt=0.02)

@show 2*seg_N*(L1+L2)

x = [data[i].x for i in eachindex(data)]
y = [data[i].y for i in eachindex(data)]
fig_opt = plot(fig_env, x, y)
