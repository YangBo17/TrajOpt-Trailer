##
using Revise
using DynamicPolynomials
using SumOfSquares
using JuMP, COPT
using LinearAlgebra
using Plots
using Mosek, MosekTools
using SCS, CSDP

# load IRIS matrix
using CSV
using DataFrames

As1 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/As1.csv"), DataFrame))
As2 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/As2.csv"), DataFrame))
As3 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/As3.csv"), DataFrame))
As4 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/As4.csv"), DataFrame))

Bs1 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Bs1.csv"), DataFrame))
Bs2 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Bs2.csv"), DataFrame))
Bs3 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Bs3.csv"), DataFrame))
Bs4 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Bs4.csv"), DataFrame))

Cs1 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Cs1.csv"), DataFrame))
Cs2 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Cs2.csv"), DataFrame))
Cs3 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Cs3.csv"), DataFrame))
Cs4 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Cs4.csv"), DataFrame))

Ds1 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Ds1.csv"), DataFrame))
Ds2 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Ds2.csv"), DataFrame))
Ds3 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Ds3.csv"), DataFrame))
Ds4 = Matrix(CSV.read(joinpath(@__DIR__,"IRIS/backparking/Ds4.csv"), DataFrame))

# setting
# trailer
trailer_length = 0.35*1.4
trailer_width = 0.26*1.4
link_length = 0.856
car_length = trailer_length + link_length
# road
road_unit = 1.5
road_width = road_unit
garage_width = 0.6
garage_length = road_unit
# waypoints
waypoints = [-1.0*road_unit 0.65*road_unit; 0.7*road_unit 0.4*road_unit; 0.0 0.0; 0.0 -trailer_length; 0.0 -garage_length]'
start_vel = [1.0, 0.0]
goal_vel = [0.0, -0.2]
# plot env
function plot_env()
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    # start and goal
    fig_env = plot([waypoints[1,1]], [waypoints[2,1]], seriestype=:scatter, aspect_ratio=:equal, markershape=:star5, markersize=3, color=:red, label="")
    plot!(fig_env, [waypoints[1,end]], [waypoints[2,end]], seriestype=:scatter, markershape=:circle, markersize=3,aspect_ratio=:equal, color=:orange, label="")
    ylims!(fig_env, -2.0, +2.0)
    xlims!(fig_env, -2.0, +3.0)
    # obstacle
    plot!(fig_env, rectangle(2-garage_width/2,2,-2,-2), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(3-garage_width/2,2,garage_width/2,-2), opacity=0.5, color=:gray, label="")
    # plot!(fig_env, rectangle(garage_width,2.0-garage_length,-0.5,-2), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(5,2,-2,garage_length), opacity=0.5, color=:gray, label="")
    # middle points
    plot!(fig_env, [waypoints[1,2]], [waypoints[2,2]], seriestype=:scatter, aspect_ratio=:equal, markershape=:rect, markersize=2, color=:blue, label="")
    plot!(fig_env, [waypoints[1,3]], [waypoints[2,3]], seriestype=:scatter, aspect_ratio=:equal, markershape=:rect, markersize=2, color=:blue, label="")
    plot!(fig_env, [waypoints[1,4]], [waypoints[2,4]], seriestype=:scatter, aspect_ratio=:equal, markershape=:rect, markersize=2, color=:blue, label="")
    return fig_env
end

## trajectory optimization
d = 5 # poly degree
N = 4 # poly section
M = 2 # flat output
@polyvar t
solver = optimizer_with_attributes(COPT.ConeOptimizer)
# solver = optimizer_with_attributes(Mosek.Optimizer)
function trajopt_space(Time::Vector{Float64})
    model = SOSModel(solver)
    @variable(model, coef[i=1:(d+1),j=1:M,k=1:N])
    poly_basis = monomials([t], 0:d)
    p = [coef[:,i,j]'*poly_basis for i in 1:M, j in 1:N]

    # objective
    order = 4 # order of derivation
    hilbert = zeros(d+1-order, d+1-order, N) # hilbert matrix
    size_hilbert = d+1-order
    h = []
    
    new_coef = 1.0 * model[:coef]
    for i in 1:(d+1)
        if (d+1)-i-order >= 0
            new_coef[i,:,:] = new_coef[i,:,:] * factorial((d+1)-i,(d+1)-i-order)
        else
            new_coef[i,:,:] = new_coef[i,:,:] * 0
        end
    end

    for n in 1:N
        for i in 1:(2*size_hilbert-1)
            for j in 1:size_hilbert, k in 1:size_hilbert
                if j + k == i + 1
                    hilbert[j,k,n] = Time[n]^(2*size_hilbert-i)/(2*size_hilbert-i)                    
                end
            end
        end
        hilbert_L = cholesky(hilbert[:,:,n]).L 
        hc = Matrix(hilbert_L') * new_coef[1:d+1-order,:,n]
        for i in 1:M
            push!(h, hc[:,i])
        end
    end
    L = length(h[1])
    @variable(model, α[1:M*N])
    @objective(model, Min, sum(α))
    mat = Array{Any}(undef, L+1, L+1, M*N)
    for i in 1:M*N
        new_mat = α[i]*Matrix(I,L+1,L+1)
        new_mat[2:L+1,1] = h[i]
        new_mat[1,2:L+1] = h[i]'
        mat[:,:,i] = new_mat
        @constraint(model, new_mat in PSDCone())
    end

    # equality constraint
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    D3p = differentiate.(p,t,3)
    for i in 1:N
        @constraint(model, p[1,i](t=>0.0) == waypoints[1,i])
        @constraint(model, p[2,i](t=>0.0) == waypoints[2,i])
    end
    @constraint(model, p[1,end](t=>Time[end]) == waypoints[1,end])
    @constraint(model, p[2,end](t=>Time[end]) == waypoints[2,end])

    @constraint(model, D1p[1,1](t=>0.0) == start_vel[1])
    @constraint(model, D1p[2,1](t=>0.0) == start_vel[2])
    @constraint(model, D2p[1,1](t=>0.0) == 0.0)
    @constraint(model, D2p[2,1](t=>0.0) == 0.0)
    @constraint(model, D1p[1,end](t=>Time[end]) == goal_vel[1])
    @constraint(model, D1p[2,end](t=>Time[end]) == goal_vel[2])
    @constraint(model, D2p[1,end](t=>Time[end]) == 0.0)
    @constraint(model, D2p[2,end](t=>Time[end]) == 0.0)

    # continuous
    for i in 1:N-1
        if i == 1 
            @constraint(model, p[1,i](t=>Time[i]) == p[1,i+1](t=>0.0))
            @constraint(model, p[2,i](t=>Time[i]) == p[2,i+1](t=>0.0))
            @constraint(model, D1p[1,i](t=>Time[i]) == -D1p[1,i+1](t=>0.0))
            @constraint(model, D1p[2,i](t=>Time[i]) == -D1p[2,i+1](t=>0.0))
            @constraint(model, D2p[1,i](t=>Time[i]) == -D2p[1,i+1](t=>0.0))
            @constraint(model, D2p[2,i](t=>Time[i]) == -D2p[2,i+1](t=>0.0))
            @constraint(model, D3p[1,i](t=>Time[i]) == -D3p[1,i+1](t=>0.0))
            @constraint(model, D3p[2,i](t=>Time[i]) == -D3p[2,i+1](t=>0.0))
            # @constraint(model, D1p[1,i](t=>Time[i]) == 0.1)
            # @constraint(model, D1p[2,i](t=>Time[i]) == 0.0)
            # @constraint(model, D2p[1,i](t=>Time[i]) == 0.2)
            # @constraint(model, D2p[2,i](t=>Time[i]) == 0.0)
            # @constraint(model, D3p[1,i](t=>Time[i]) == 0.2)
            # @constraint(model, D3p[2,i](t=>Time[i]) == 0.0)
        else
            @constraint(model, p[1,i](t=>Time[i]) == p[1,i+1](t=>0.0))
            @constraint(model, p[2,i](t=>Time[i]) == p[2,i+1](t=>0.0))
            @constraint(model, D1p[1,i](t=>Time[i]) == D1p[1,i+1](t=>0.0))
            @constraint(model, D1p[2,i](t=>Time[i]) == D1p[2,i+1](t=>0.0))
            @constraint(model, D2p[1,i](t=>Time[i]) == D2p[1,i+1](t=>0.0))
            @constraint(model, D2p[2,i](t=>Time[i]) == D2p[2,i+1](t=>0.0))
            @constraint(model, D3p[1,i](t=>Time[i]) == D3p[1,i+1](t=>0.0))
            @constraint(model, D3p[2,i](t=>Time[i]) == D3p[2,i+1](t=>0.0))
        end
    end

    # inequality constraint
    if (d-1)%2 == 0
        Xf = monomials([t], 0:d-1)
        Xg = monomials([t], 0:d-3)
    else
        Xf = monomials([t], 0:d-2)
        Xg = monomials([t], 0:d-2)
    end
    @variable(model, f1[1:size(As1,1)], Poly(Xf))
    @variable(model, g1[1:size(As1,1)], Poly(Xg))
    @variable(model, f2[1:size(As2,1)], Poly(Xf))
    @variable(model, g2[1:size(As2,1)], Poly(Xg))
    @variable(model, f3[1:size(As3,1)], Poly(Xf))
    @variable(model, g3[1:size(As3,1)], Poly(Xg))
    @variable(model, f4[1:size(As4,1)], Poly(Xf))
    @variable(model, g4[1:size(As4,1)], Poly(Xg))

    # section 1
    X1 = [D1p[1,1], D1p[2,1], D2p[1,1], D2p[2,1], D3p[1,1], D3p[2,1]]
    for i in 1:size(As1,1)
        @constraint(model, f1[i] >= 0)
        @constraint(model, g1[i] >= 0)
        if (d-1)%2 == 0
            @constraint(model, Bs1[i] - As1[i,:]'*X1 == f1[i] - (t-0.0)*(t-Time[1])*g1[i])
        else
            @constraint(model, Bs1[i] - As1[i,:]'*X1 == (t-0.0)*f1[i] - (t-Time[1])*g1[i])
        end
    end

    # section 2
    X2 = [D1p[1,2], D1p[2,2], D2p[1,2], D2p[2,2], D3p[1,2], D3p[2,2]]
    for i in 1:size(As2,1)
        @constraint(model, f2[i] >= 0)
        @constraint(model, g2[i] >= 0)
        if (d-1)%2 == 0
            @constraint(model, Bs2[i] - As2[i,:]'*X2 == f2[i] - (t-0.0)*(t-Time[2])*g2[i])
        else
            @constraint(model, Bs2[i] - As2[i,:]'*X2 == (t-0.0)*f2[i] - (t-Time[2])*g2[i])
        end
    end

    # section 3
    X3 = [D1p[1,3], D1p[2,3], D2p[1,3], D2p[2,3], D3p[1,3], D3p[2,3]]
    for i in 1:size(As3,1)
        @constraint(model, f3[i] >= 0)
        @constraint(model, g3[i] >= 0)
        if (d-1)%2 == 0
            @constraint(model, Bs3[i] - As3[i,:]'*X3 == f3[i] - (t-0.0)*(t-Time[3])*g3[i])
        else
            @constraint(model, Bs3[i] - As3[i,:]'*X3 == (t-0.0)*f3[i] - (t-Time[3])*g3[i])
        end
    end
    
    # section 4
    X4 = [D1p[1,4], D1p[2,4], D2p[1,4], D2p[2,4], D3p[1,4], D3p[2,4]]
    for i in 1:size(As4,1)
        @constraint(model, f4[i] >= 0)
        @constraint(model, g4[i] >= 0)
        if (d-1)%2 == 0
            @constraint(model, Bs4[i] - As4[i,:]'*X4 == f4[i] - (t-0.0)*(t-Time[4])*g4[i])
        else
            @constraint(model, Bs4[i] - As4[i,:]'*X4 == (t-0.0)*f4[i] - (t-Time[4])*g4[i])
        end
    end

    JuMP.optimize!(model)
    return value.(p), model
end

function trajopt_time(Time0::Vector{Float64})
    function get_gradient_fd(obj0::Float64, Time0::Vector{Float64}, h=1e-6)
        grad = zeros(size(Time0))
        origin_time = Time0
        for i in eachindex(Time0)
            try_time = copy(origin_time)
            try_time[i] += h
            _, mdl = trajopt_space(try_time)
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
            _, mdl = trajopt_space(try_time)
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
                p, mdl = trajopt_space(candid_time)
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
    _, mdl = trajopt_space(Time0)
    obj0 = objective_value(mdl)
    obj0t = obj0 + sum(tweight.*Time0)
    Times, Objs, Objts, ps = backtracking_line_search(max_iter=max_iter,grad_method=grad_method)
    return Times, Objs, Objts, ps
end

includet("../visual/visual_trailer.jl")
trailer_body = TrailerBody()
d1 = trailer_body.link_length

v1 = 1.0
Time0 = [1.5, 1.5, 0.5, 1.3] .+ 0.3
p0, model = trajopt_space(Time0)
solve_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
obj_val = objective_value(model)
@show solve_time
@show obj_val

# tweight = 1.0*ones(4)
# max_iter = 10
# Times, Objs, Objts, ps = trajopt_time(Time0)

##

function process_data(p::Matrix{Polynomial{true, Float64}}, Time::Vector{Float64}, trailer_body::TrailerBody)
    d0, d1, w = trailer_body.trailer_length, trailer_body.link_length, trailer_body.trailer_width
    data = []
    total_time = [0.0]
    dt = 0.01
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    D3p = differentiate.(p,t,3)
    for n in 1:N
        if n != N
            ts = 0:dt:Time[n]-dt |> collect
        else
            ts = 0:dt:Time[n] |> collect
        end
        push!(total_time, total_time[end]+Time[n])
        for i in eachindex(ts)
            if n <= 1
                ti = ts[i] + total_time[n]
                x1 = p[1,n](t=>ts[i])
                y1 = p[2,n](t=>ts[i])
                vx1 = D1p[1,n](t=>ts[i])
                vy1 = D1p[2,n](t=>ts[i])
                v1 = sqrt(vx1^2+vy1^2)
                ax1 = D2p[1,n](t=>ts[i])
                ay1 = D2p[2,n](t=>ts[i])
                a1 = sqrt(ax1^2+ay1^2)
                jx1 = D3p[1,n](t=>ts[i])
                jy1 = D3p[2,n](t=>ts[i])
                j1 = sqrt(jx1^2+jy1^2)
                x0 = x1 + d1 * vx1 / v1
                y0 = y1 + d1 * vy1 / v1
                vx0 = vx1 - d1 * vy1 * (ay1*vx1-vy1*ax1) / (v1^3)
                vy0 = vy1 + d1 * vx1 * (ay1*vx1-vy1*ax1) / (v1^3)
                ax0 = ax1 - d1 / v1^6 * ( (ay1*(ay1*vx1-vy1*ax1)+vy1*(jy1*vx1-vy1*jx1))*v1^3 - vy1*(ay1*vx1-vy1*ax1)*3/2*v1*(2*vx1*ax1+2*vy1*ay1) )
                ay0 = ay1 + d1 / v1^6 * ( (ax1*(ay1*vx1-vy1*ax1)+vx1*(jy1*vx1-vy1*jx1))*v1^3 - vx1*(ay1*vx1-vy1*ax1)*3/2*v1*(2*vx1*ax1+2*vy1*ay1) )
                θ1 = atan(vy1, vx1)
                θ0 = atan(vy0, vx0)
                v = sqrt(vx0^2+vy0^2)
                a = sqrt(ax0^2+ay0^2)
                ϕ = atan(d0*(ay0*vx0-vy0*ax0), v^3)
                # κ = (ay1*vx1-vy1*ax1)/v1^3
                κ = (ay0*vx0-vy0*ax0)/v^3
            else
                ti = ts[i] + total_time[n]
                x1 = p[1,n](t=>ts[i])
                y1 = p[2,n](t=>ts[i])
                vx1 = D1p[1,n](t=>ts[i])
                vy1 = D1p[2,n](t=>ts[i])
                v1 = sqrt(vx1^2+vy1^2)
                ax1 = D2p[1,n](t=>ts[i])
                ay1 = D2p[2,n](t=>ts[i])
                a1 = sqrt(ax1^2+ay1^2)
                jx1 = D3p[1,n](t=>ts[i])
                jy1 = D3p[2,n](t=>ts[i])
                j1 = sqrt(jx1^2+jy1^2)
                x0 = x1 - d1 * vx1 / v1
                y0 = y1 - d1 * vy1 / v1
                vx0 = vx1 + d1 * vy1 * (ay1*vx1-vy1*ax1) / (v1^3)
                vy0 = vy1 - d1 * vx1 * (ay1*vx1-vy1*ax1) / (v1^3)
                ax0 = ax1 + d1 / v1^6 * ( (ay1*(ay1*vx1-vy1*ax1)+vy1*(jy1*vx1-vy1*jx1))*v1^3 - vy1*(ay1*vx1-vy1*ax1)*3/2*v1*(2*vx1*ax1+2*vy1*ay1) )
                ay0 = ay1 - d1 / v1^6 * ( (ax1*(ay1*vx1-vy1*ax1)+vx1*(jy1*vx1-vy1*jx1))*v1^3 - vx1*(ay1*vx1-vy1*ax1)*3/2*v1*(2*vx1*ax1+2*vy1*ay1))
                θ1 = atan(vy1, vx1)+π
                θ0 = atan(vy0, vx0)+π
                v = sqrt(vx0^2+vy0^2)
                a = sqrt(ax0^2+ay0^2)
                ϕ = -atan(d0*(ay0*vx0-vy0*ax0), v^3)
                # κ = (ay1*vx1-vy1*ax1)/v1^3
                κ = (ay0*vx0-vy0*ax0)/v^3
            end
            datapoint = TrailerState(ti, x0, y0, vx0, vy0, ax0, ay0, x1, y1, vx1, vy1, ax1, ay1, jx1, jy1, θ0, θ1, v,a, ϕ, κ)
            push!(data, datapoint)
        end
    end
    return data
end

data0 = process_data(p0, Time0, trailer_body)
# data1 = process_data(ps[end], Times[end], trailer_body)
data = data0

function plot_data1(env::Plots.Plot, data::Vector{Any}; color::Symbol=red, label::String)
    fig_traj = plot!(env, [data[i].x1 for i in eachindex(data)], [data[i].y1 for i in eachindex(data)], color=color, label=label, legend=:outertopright)
    return fig_traj
end

function plot_data2(env::Plots.Plot, data::Vector{Any}; color::Symbol=red, label::String)
    fig_traj = plot!(env, [data[i].x0 for i in eachindex(data)], [data[i].y0 for i in eachindex(data)], color=color, label=label, legend=:outertopright)
    return fig_traj
end


fig_env = plot_env()
# fig_traj = plot_data(fig_env, data0; color=:red, label="initial")
# fig_traj = plot_data(fig_traj, data1; color=:blue, label="final")
visual_traj(fig_env, data; num=0, c0=:orange, c1=:green, label="opt")
animate_traj(fig_env, data; c0=:orange, c1=:green, label="opt", fps=15, interval=5)

## 
plot_data(fig_env, data1; color=:red, label="opt")

##
data = data0
fig_v1 = plot([data[i].t for i in eachindex(data)], [data[i].vx1 for i in eachindex(data)], label="vx1",legend=:outertopright)
plot!(fig_v1, [data[i].t for i in eachindex(data)], [data[i].vy1 for i in eachindex(data)], label="vy1")
fig_a1 = plot([data[i].t for i in eachindex(data)], [data[i].ax1 for i in eachindex(data)], label="ax1",legend=:outertopright)
plot!(fig_a1, [data[i].t for i in eachindex(data)], [data[i].ay1 for i in eachindex(data)], label="ay1")
fig_j1 = plot([data[i].t for i in eachindex(data)], [data[i].ax1 for i in eachindex(data)], label="jx1",legend=:outertopright)
plot!(fig_j1, [data[i].t for i in eachindex(data)], [data[i].jy1 for i in eachindex(data)], label="jy1")
vline!(fig_v1, [Time0[1], sum(Time0[1:2]), sum(Time0[1:3]), sum(Time0[1:4])])
vline!(fig_a1, [Time0[1], sum(Time0[1:2]), sum(Time0[1:3]), sum(Time0[1:4])])
vline!(fig_j1, [Time0[1], sum(Time0[1:2]), sum(Time0[1:3]), sum(Time0[1:4])])
plot(fig_v1,fig_a1,fig_j1,layout=(3,1))
# fig_v = plot([data[i].t for i in eachindex(data)], [sqrt(data[i].vx1^2+data[i].vy1^2) for i in eachindex(data)], label="v1")
# fig_κ = plot([data[i].t for i in eachindex(data)], [data[i].v for i in eachindex(data)], label="v")
fig_κ = plot([data[i].t for i in eachindex(data)], [data[i].κ for i in eachindex(data)], label="κ")
vline!(fig_κ, [Time0[1], sum(Time0[1:2]), sum(Time0[1:3]), sum(Time0[1:4])],label="")
hline!(fig_κ, [2.0], color=:red,label="")
fig_v = plot([data[i].t for i in eachindex(data)], [data[i].v for i in eachindex(data)], label="v")
vline!(fig_v, [Time0[1], sum(Time0[1:2]), sum(Time0[1:3]), sum(Time0[1:4])],label="")
hline!(fig_v, [2.0], color=:red,label="")
fig_a = plot([data[i].t for i in eachindex(data)], [data[i].a for i in eachindex(data)], label="a")
vline!(fig_a, [Time0[1], sum(Time0[1:2]), sum(Time0[1:3]), sum(Time0[1:4])],label="")
hline!(fig_a, [2.0], color=:red,label="")
plot(fig_κ,fig_v,fig_a,layout=(3,1))
# ylims!(fig_κ, -10,10)
# fig_va0 = plot([data[i].t for i in eachindex(data)], [sqrt(data[i].vx0^2+data[i].vy0^2) for i in eachindex(data)], label="v0")
# plot!(fig_va0, [data[i].t for i in eachindex(data)], [sqrt(data[i].ax0^2+data[i].ay0^2) for i in eachindex(data)], label="a0")

##
vx1 = data[360].vx1
vy1 = data[360].vy1
ax1 = data[360].ax1
ay1 = data[360].ay1
jx1 = data[360].jx1
jy1 = data[360].jy1
vx0 = data[360].vx0
vy0 = data[360].vy0
ax0 = data[360].ax0
ay0 = data[360].ay0
κ0 = abs(ay0*vx0-ax0*vy0)/sqrt(vx0^2+vy0^2)^3

X = [vx1,vy1,ax1,ay1,jx1,jy1]
res2 = [As2[i,:]'*X-Bs2[i] for i in 1:size(As2,1)]
res3 = [As3[i,:]'*X-Bs3[i] for i in 1:size(As2,1)]
@show minimum(res2)
@show minimum(res3)
@show maximum(res2)
@show maximum(res3)

## 
fig_obj = plot([Objts[i] for i in 1:max_iter], label="obj")

##
function plot_time(Times::Vector{Any})
    fig_time = plot([Times[i][1] for i in 1:max_iter],legend=:outertopright)
    for n in 2:N
        plot!(fig_time, [Times[i][n] for i in 1:max_iter])
    end
    return fig_time
end
fig_time = plot_time(Times)

##
fig_env = plot_env()
fig_traj1 = plot_data1(fig_env, data0; color=:red, label="initial")
fig_traj1 = plot_data1(fig_traj1, data1; color=:blue, label="final")

##
fig_env = plot_env()
fig_traj2 = plot_data2(fig_env, data0; color=:red, label="initial")
fig_traj2 = plot_data2(fig_traj2, data1; color=:blue, label="final")

