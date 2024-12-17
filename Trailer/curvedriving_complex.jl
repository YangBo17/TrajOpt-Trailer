##
using Revise
using DynamicPolynomials
using SumOfSquares
using JuMP, COPT
using LinearAlgebra
using Plots

# setting
# trailer
trailer_length = 0.35
trailer_width = 0.35
link_length = 0.5
car_length = trailer_length + link_length
# road
road_unit = 1.0
# waypoints
waypoints = [0.0 0.0; 9*road_unit 11*road_unit]'
# plot env
function plot_env()
    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    # start and goal
    fig_env = plot([waypoints[1,1]], [waypoints[2,1]], seriestype=:scatter, aspect_ratio=:equal, markershape=:star5, markersize=3, color=:red, label="")
    plot!(fig_env, [waypoints[1,end]], [waypoints[2,end]], seriestype=:scatter, aspect_ratio=:equal, markershape=:circle, markersize=3, color=:orange, label="") 
    ylims!(fig_env, -2*road_unit, 12*road_unit)
    xlims!(fig_env, -2*road_unit, 12*road_unit)

    # obstacle
    obstacle_unit = road_unit * 1.0
    obstacle_x = 0:2*road_unit:4*2*road_unit |> collect
    obstacle_y = road_unit:2*road_unit:road_unit+4*2*road_unit |> collect
    for i in eachindex(obstacle_x), j in eachindex(obstacle_y)
        plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[i], obstacle_y[j]), opacity=0.5, color=:gray, label="")
    end
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[1], obstacle_y[1]+road_unit), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[1]+road_unit, obstacle_y[2]), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[2]+road_unit, obstacle_y[2]), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[2]+road_unit, obstacle_y[1]), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[3]+road_unit, obstacle_y[1]), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[4], obstacle_y[1]+road_unit), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[3], obstacle_y[2]+road_unit), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[4], obstacle_y[2]+road_unit), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[3], obstacle_y[3]+road_unit), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[3]+road_unit, obstacle_y[4]), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[4]+road_unit, obstacle_y[3]), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[4], obstacle_y[4]+road_unit), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[5], obstacle_y[3]+road_unit), opacity=0.5, color=:gray, label="")
    plot!(fig_env, rectangle(obstacle_unit,obstacle_unit, obstacle_x[5], obstacle_y[4]+road_unit), opacity=0.5, color=:gray, label="")

    return fig_env
end

## trajectory optimization
d = 5
N = 7
M = 2
@polyvar t 
solver = optimizer_with_attributes(COPT.ConeOptimizer)
Time0 = [1.0, 2.0, 3.0, 3.0, 1.0, 3.0, 1.0]
function trajopt_space(Time::Vector{Float64})
    model = SOSModel(solver)
    @variable(model, coef[i=1:(d+1),j=1:M,k=1:N])
    poly_basis = monomials([t], 0:d)
    p = [coef[:,i,j]'*poly_basis for i in 1:M, j in 1:N]

    # objective
    order = 3 # order of  derivation
    hilbert = zeros(d+1-order, d+1-order, N)
    # matrix
    size_hilbert = d+1-order
    h = []

    new_coef = 1.0 * model[:coef]
    for i in 1:(d+1)
        if (d+1)-i-order >= 0
            new_coef[i,:,:] = new_coef[i,:,:] * factorial((d+1)-i, (d+1)-i-order)
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
    @constraint(model, p[1,1](t=>0.0) == waypoints[1,1])
    @constraint(model, p[2,1](t=>0.0) == waypoints[2,1])
    @constraint(model, p[1,end](t=>Time[end]) == waypoints[1,end])
    @constraint(model, p[2,end](t=>Time[end]) == waypoints[2,end])
    @constraint(model, p[1,2](t=>0.0) == 1.5)
    @constraint(model, p[2,2](t=>0.0) == 1.0)
    @constraint(model, p[1,3](t=>0.0) == 2.0)
    @constraint(model, p[2,3](t=>0.0) == 2.5)
    @constraint(model, p[1,4](t=>0.0) == 5.5)
    @constraint(model, p[2,4](t=>0.0) == 3.0)
    @constraint(model, p[1,5](t=>0.0) == 6.0)
    @constraint(model, p[2,5](t=>0.0) == 6.5)
    @constraint(model, p[1,6](t=>0.0) == 7.5)
    @constraint(model, p[2,6](t=>0.0) == 7.0)
    @constraint(model, p[1,7](t=>0.0) == 7.5)
    @constraint(model, p[2,7](t=>0.0) == 10.0)
    # continuous
    for i in 1:N-1 
        @constraint(model, p[1,i](t=>Time[i]) == p[1,i+1](t=>0.0))
        @constraint(model, p[2,i](t=>Time[i]) == p[2,i+1](t=>0.0))
        @constraint(model, D1p[1,i](t=>Time[i]) == D1p[1,i+1](t=>0.0))
        @constraint(model, D1p[2,i](t=>Time[i]) == D1p[2,i+1](t=>0.0))
        @constraint(model, D2p[1,i](t=>Time[i]) == D2p[1,i+1](t=>0.0))
        @constraint(model, D2p[2,i](t=>Time[i]) == D2p[2,i+1](t=>0.0))
        @constraint(model, D3p[1,i](t=>Time[i]) == D3p[1,i+1](t=>0.0))
        @constraint(model, D3p[2,i](t=>Time[i]) == D3p[2,i+1](t=>0.0))
    end

    # inequality constraint
    # space
    if d%2 == 0
        Xf = monomials([t], 0:d)
        Xg = monomials([t], 0:d-2)
    else
        Xf = monomials([t], 0:d-1)
        Xg = monomials([t], 0:d-1)
    end
    @variable(model, f[1:4*N], Poly(Xf))
    @variable(model, g[1:4*N], Poly(Xg))
    for i in 1:4*N
        @constraint(model, f[i] >= 0)
        @constraint(model, g[i] >= 0)
    end

    if d%2 == 0
        @constraint(model, p[1,1] - 0*road_unit == f[1] - (t-0.0)*(t-Time[1])*g[1])
        @constraint(model, -p[1,1] + 2*road_unit == f[2] - (t-0.0)*(t-Time[1])*g[2])
        @constraint(model, p[2,1] + 0*road_unit == f[3] - (t-0.0)*(t-Time[1])*g[3])
        @constraint(model, -p[2,1] + 1*road_unit == f[4] - (t-0.0)*(t-Time[1])*g[4])

        @constraint(model, p[1,2] - 1*road_unit == f[5] - (t-0.0)*(t-Time[2])*g[5])
        @constraint(model, -p[1,2] + 2*road_unit == f[6] - (t-0.0)*(t-Time[2])*g[6])
        @constraint(model, p[2,2] - 1*road_unit == f[7] - (t-0.0)*(t-Time[2])*g[7])
        @constraint(model, -p[2,2] + 3*road_unit == f[8] - (t-0.0)*(t-Time[2])*g[8])

        @constraint(model, p[1,3] - 2*road_unit == f[9] - (t-0.0)*(t-Time[3])*g[9])
        @constraint(model, -p[1,3] + 6*road_unit == f[10] - (t-0.0)*(t-Time[3])*g[10])
        @constraint(model, p[2,3] - 2*road_unit == f[11] - (t-0.0)*(t-Time[3])*g[11])
        @constraint(model, -p[2,3] + 3*road_unit == f[12] - (t-0.0)*(t-Time[3])*g[12])

        @constraint(model, p[1,4] - 5*road_unit == f[13] - (t-0.0)*(t-Time[4])*g[13])
        @constraint(model, -p[1,4] + 6*road_unit == f[14] - (t-0.0)*(t-Time[4])*g[14])
        @constraint(model, p[2,4] - 3*road_unit == f[15] - (t-0.0)*(t-Time[4])*g[15])
        @constraint(model, -p[2,4] + 7*road_unit == f[16] - (t-0.0)*(t-Time[4])*g[16])

        @constraint(model, p[1,5] - 6*road_unit == f[17] - (t-0.0)*(t-Time[5])*g[17])
        @constraint(model, -p[1,5] + 8*road_unit == f[18] - (t-0.0)*(t-Time[5])*g[18])
        @constraint(model, p[2,5] - 6*road_unit == f[19] - (t-0.0)*(t-Time[5])*g[19])
        @constraint(model, -p[2,5] + 7*road_unit == f[20] - (t-0.0)*(t-Time[5])*g[20])

        @constraint(model, p[1,6] - 7*road_unit == f[21] - (t-0.0)*(t-Time[6])*g[21])
        @constraint(model, -p[1,6] + 8*road_unit == f[22] - (t-0.0)*(t-Time[6])*g[22])
        @constraint(model, p[2,6] - 7*road_unit == f[23] - (t-0.0)*(t-Time[6])*g[23])
        @constraint(model, -p[2,6] + 11*road_unit == f[24] - (t-0.0)*(t-Time[6])*g[24])

        @constraint(model, p[1,7] - 7*road_unit == f[25] - (t-0.0)*(t-Time[7])*g[25])
        @constraint(model, -p[1,7] + 9*road_unit == f[26] - (t-0.0)*(t-Time[7])*g[26])
        @constraint(model, p[2,7] - 10*road_unit == f[27] - (t-0.0)*(t-Time[7])*g[27])
        @constraint(model, -p[2,7] + 11*road_unit == f[28] - (t-0.0)*(t-Time[7])*g[28])
    else
        @constraint(model, p[1,1] + 0*road_unit == (t-0.0)*f[1] - (t-Time[1])*g[1])
        @constraint(model, -p[1,1] + 2*road_unit == (t-0.0)*f[2] - (t-Time[1])*g[2])
        @constraint(model, p[2,1] - 0*road_unit == (t-0.0)*f[3] - (t-Time[1])*g[3])
        @constraint(model, -p[2,1] + 1*road_unit == (t-0.0)*f[4] - (t-Time[1])*g[4])

        @constraint(model, p[1,2] - 1*road_unit == (t-0.0)*f[5] - (t-Time[2])*g[5])
        @constraint(model, -p[1,2] + 2*road_unit == (t-0.0)*f[6] - (t-Time[2])*g[6])
        @constraint(model, p[2,2] - 1*road_unit == (t-0.0)*f[7] - (t-Time[2])*g[7])
        @constraint(model, -p[2,2] + 3*road_unit == (t-0.0)*f[8] - (t-Time[2])*g[8])

        @constraint(model, p[1,3] - 2*road_unit == (t-0.0)*f[9] - (t-Time[3])*g[9])
        @constraint(model, -p[1,3] + 6*road_unit == (t-0.0)*f[10] - (t-Time[3])*g[10])
        @constraint(model, p[2,3] - 2*road_unit == (t-0.0)*f[11] - (t-Time[3])*g[11])
        @constraint(model, -p[2,3] + 3*road_unit == (t-0.0)*f[12] - (t-Time[3])*g[12])

        @constraint(model, p[1,4] - 5*road_unit == (t-0.0)*f[13] - (t-Time[4])*g[13])
        @constraint(model, -p[1,4] + 6*road_unit == (t-0.0)*f[14] - (t-Time[4])*g[14])
        @constraint(model, p[2,4] - 3*road_unit == (t-0.0)*f[15] - (t-Time[4])*g[15])
        @constraint(model, -p[2,4] + 7*road_unit == (t-0.0)*f[16] - (t-Time[4])*g[16])

        @constraint(model, p[1,5] - 6*road_unit == (t-0.0)*f[17] - (t-Time[5])*g[17])
        @constraint(model, -p[1,5] + 8*road_unit == (t-0.0)*f[18] - (t-Time[5])*g[18])
        @constraint(model, p[2,5] - 6*road_unit == (t-0.0)*f[19] - (t-Time[5])*g[19])
        @constraint(model, -p[2,5] + 7*road_unit == (t-0.0)*f[20] - (t-Time[5])*g[20])

        @constraint(model, p[1,6] - 7*road_unit == (t-0.0)*f[21] - (t-Time[6])*g[21])
        @constraint(model, -p[1,6] + 8*road_unit == (t-0.0)*f[22] - (t-Time[6])*g[22])
        @constraint(model, p[2,6] - 7*road_unit == (t-0.0)*f[23] - (t-Time[6])*g[23])
        @constraint(model, -p[2,6] + 11*road_unit == (t-0.0)*f[24] - (t-Time[6])*g[24])

        @constraint(model, p[1,7] - 7*road_unit == (t-0.0)*f[25] - (t-Time[7])*g[25])
        @constraint(model, -p[1,7] + 9*road_unit == (t-0.0)*f[26] - (t-Time[7])*g[26])
        @constraint(model, p[2,7] - 10*road_unit == (t-0.0)*f[27] - (t-Time[7])*g[27])
        @constraint(model, -p[2,7] + 11*road_unit == (t-0.0)*f[28] - (t-Time[7])*g[28])
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

Time0 = [2.0, 2.0, 2.0, 2.0, 2.0, 3.0, 2.0] * 2
p0, model = trajopt_space(Time0)
solve_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
obj_val = objective_value(model)
@show solve_time
@show obj_val

tweight = 1.0*ones(7)
max_iter = 50
Times, Objs, Objts, ps = trajopt_time(Time0)


##
includet("../visual/visual_trailer.jl")
trailer_body = TrailerBody()

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

            datapoint = TrailerState(ti, x0, y0, vx0, vy0, ax0, ay0, x1, y1, vx1, vy1, ax1, ay1, jx1, jy1, θ0, θ1, v,a, ϕ, κ)
            push!(data, datapoint)
        end
    end
    return data
end

data0 = process_data(p0, Time0, trailer_body)
data1 = process_data(ps[end], Times[end], trailer_body)

# function plot_data(env::Plots.Plot, data::Vector{Any}; color::Symbol=red, label::String)
#     fig_traj = plot!(env, [data[i].x1 for i in eachindex(data)], [data[i].y1 for i in eachindex(data)], color=color, label=label, legend=:outertopright)
#     return fig_traj
# end

fig_env = plot_env()
# fig_traj = plot_data(fig_env, data0; color=:red, label="initial")
# fig_traj = plot_data(fig_traj, data1; color=:blue, label="final")
visual_traj(fig_env, data1; num=7, c0=:orange, c1=:green, label="opt")
