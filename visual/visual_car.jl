using Revise
using Plots 

includet("../models/car.jl")

struct CarState
    t::Float64
    x::Float64
    y::Float64
    θ::Float64
    vx::Float64
    vy::Float64
    ax::Float64
    ay::Float64
    v::Float64
    a::Float64
    δ::Float64
    κ::Float64
end

function depict_car(body, carstate::Vector{Float64})
    L, W = body.L, body.W 
    x, y, θ, δ = carstate
    car_shapes = []
    # Body
    length_margin = 0.20 * L
    width_margin = 0.20 * W
    α = atan(length_margin, width_margin+W/2)
    β = atan(width_margin+W/2, L+length_margin)
    angle = [θ-π/2-α, θ-β, θ+β, θ+π/2+α, θ-π/2-α]
    dist = [sqrt(length_margin^2 + (width_margin + W/2)^2), sqrt((L+length_margin)^2 + (width_margin + W/2)^2), sqrt((L+length_margin)^2 + (width_margin + W/2)^2), sqrt(length_margin^2 + (width_margin + W/2)^2), sqrt(length_margin^2 + (width_margin + W/2)^2)]
    car = Shape(x .+ dist .* cos.(angle), y .+ dist .* sin.(angle))
    push!(car_shapes, car)
    # Tire 
    width_tire = 0.05 * W  # half
    length_tire = 0.1 * L # half
    β = atan(W/2, L)
    angle = [θ-π/2, θ-β, θ+β, θ+π/2]
    dist = [W/2, sqrt((W/2)^2+L^2), sqrt((W/2)^2+L^2), W/2]
    loca_tire_x = x .+ dist .* cos.(angle)
    loca_tire_y = y .+ dist .* sin.(angle)
    α = atan(length_tire, width_tire)
    β = atan(width_tire, length_tire)
    d_tire = sqrt(width_tire^2+length_tire^2)
    dist_tire = [d_tire, d_tire, d_tire, d_tire]
    for i in 1:4
        if i ∈ [2,3]
            angle = [θ-π/2-α+δ, θ-β+δ, θ+β+δ, θ+π/2+α+δ]
        else
            angle = [θ-π/2-α, θ-β, θ+β, θ+π/2+α]
        end
        tire = Shape(loca_tire_x[i] .+ dist_tire .* cos.(angle), loca_tire_y[i] .+ dist_tire .* sin.(angle))
        push!(car_shapes, tire)
    end
    return car_shapes
end

function depict_car(body::Body, carstate::CarState)
    L, W = body.L, body.W 
    x, y, θ = carstate.x, carstate.y, carstate.θ
    δ, v = carstate.δ, carstate.v
    car_shapes = []
    # Body
    length_margin = 0.20 * L
    width_margin = 0.20 * W
    α = atan(length_margin, width_margin+W/2)
    β = atan(width_margin+W/2, L+length_margin)
    angle = [θ-π/2-α, θ-β, θ+β, θ+π/2+α]
    dist = [sqrt(length_margin^2 + (width_margin + W/2)^2), sqrt((L+length_margin)^2 + (width_margin + W/2)^2), sqrt((L+length_margin)^2 + (width_margin + W/2)^2), sqrt(length_margin^2 + (width_margin + W/2)^2)]
    car = Shape(x .+ dist .* cos.(angle), y .+ dist .* sin.(angle))
    push!(car_shapes, car)
    # Tire 
    width_tire = 0.05 * W  # half
    length_tire = 0.1 * L # half
    β = atan(W/2, L)
    angle = [θ-π/2, θ-β, θ+β, θ+π/2]
    dist = [W/2, sqrt((W/2)^2+L^2), sqrt((W/2)^2+L^2), W/2]
    loca_tire_x = x .+ dist .* cos.(angle)
    loca_tire_y = y .+ dist .* sin.(angle)
    α = atan(length_tire, width_tire)
    β = atan(width_tire, length_tire)
    d_tire = sqrt(width_tire^2+length_tire^2)
    dist_tire = [d_tire, d_tire, d_tire, d_tire]
    for i in 1:4
        if i ∈ [2,3]
            angle = [θ-π/2-α+δ, θ-β+δ, θ+β+δ, θ+π/2+α+δ]
        else
            angle = [θ-π/2-α, θ-β, θ+β, θ+π/2+α]
        end
        tire = Shape(loca_tire_x[i] .+ dist_tire .* cos.(angle), loca_tire_y[i] .+ dist_tire .* sin.(angle))
        push!(car_shapes, tire)
    end
    return car_shapes
end

function render_car!(fig::Plots.Plot, car_shapes; c::Symbol)
    plot!(fig, car_shapes[1], c = c, label="")
    plot!(fig, car_shapes[2], c = :black, label="")
    plot!(fig, car_shapes[3], c = :black, label="")
    plot!(fig, car_shapes[4], c = :black, label="")
    plot!(fig, car_shapes[5], c = :black, label="")
    return fig
end

function moose_env(X::Matrix{Float64}, Y::Matrix{Float64}, scope::Vector{Float64}; L::Float64)
    M = size(X)[2]
    fig_env = plot(aspect_ratio = 1)
    for i in 1:M-1
        line1_x = collect(range(X[1,i], X[1,i+1], 100))
        line1_y = collect(range(Y[1,i]-L, Y[1,i+1]-L, 100))
        line2_x = collect(range(X[2,i], X[2,i+1], 100))
        line2_y = collect(range(Y[2,i]+L, Y[2,i+1]+L, 100))
        plot!(fig_env, line1_x, line1_y, c=:black, label="")
        plot!(fig_env, line2_x, line2_y, c=:black, label="")
    end
    xlims!(fig_env, scope[1], scope[2])
    ylims!(fig_env, scope[3], scope[4])
    for i in 1:300
        x = rand(collect(scope[1]:0.8*L:scope[2]))
        y = rand(collect(scope[3]:0.8*L:scope[4]))
        marker = rand([:circle, :rect, :diamond, :hexagon, :utriangle, :dtriangle, :pentagon, :octagon], 1)
        if x <= X[1,1]-L || x >= X[1,end]+L
            plot!(fig_env, [x], [y], seriestype = :scatter, m=marker, markersize=4, label="")
        else
            for j in 1:M-1
                if (x >= X[1, j] && x <= X[1,j+1])
                    line1 = [Y[1,j]-Y[1,j+1], X[1,j+1]-X[1,j], X[1,j]*(Y[1,j+1]-L)-X[1,j+1]*(Y[1,j]-L)]
                    line2 = [Y[2,j]-Y[2,j+1], X[2,j+1]-X[2,j], X[2,j]*(Y[2,j+1]+L)-X[2,j+1]*(Y[2,j]+L)]
                    check1 = line1[1]*x + line1[2]*y + line1[3] <= -0.4*L
                    check2 = line2[1]*x + line2[2]*y + line2[3] >= +0.4*L
                    if check1 || check2
                        plot!(fig_env, [x], [y], seriestype = :scatter, m=marker, markersize=4, label="")
                    end
                end
            end
        end
    end
    return fig_env
end

function visual_traj(data::Vector{Any}, num::Int64, c::Symbol, label::String)
    N = length(data)
    interval_frame = Int64(N / (num+1)) 
    index_frame = [1 + i * interval_frame for i in 0:num]
    push!(index_frame, N)
    fig_traj = plot([p.x for p in data], [p.y for p in data], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, c = c, label=label)
    for i in index_frame
        car_shapes = depict_trailer(Body(), data[i])
        render_car!(fig_traj, car_shapes; c=c)
    end
    return fig_traj
end

function visual_traj(fig::Plots.Plot, data::Vector{Any}; num::Int64, c::Symbol, label::String)
    N = length(data)
    interval_frame = Int64(N / (num+1)) 
    index_frame = [1 + i * interval_frame for i in 0:num]
    push!(index_frame, N)
    fig_traj = plot(fig, [p.x for p in data], [p.y for p in data], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, c = c, label=label)
    for i in index_frame
        car_shapes = depict_car(Body(), data[i])
        render_car!(fig_traj, car_shapes; c=c)
    end
    return fig_traj
end

function visual_data(data_opt::Vector{Any}, data_dyn::Vector{Any}, dt::Float64=0.01)
    N = length(data_opt)
    fig_x = plot([p.t for p in data_opt], [p.x for p in data_opt], label = "x-opt",legend = :outertopright)
    fig_y = plot([p.t for p in data_opt], [p.y for p in data_opt], label = "y-opt",legend = :outertopright)
    fig_v = plot([p.t for p in data_opt], [p.v for p in data_opt], label = "v-opt",legend = :outertopright)
    fig_θ = plot([p.t for p in data_opt], [p.θ for p in data_opt], label = "θ-opt",legend = :outertopright)
    fig_δ = plot([p.t for p in data_opt], [p.δ for p in data_opt], label = "δ-opt",legend = :outertopright)
    plot!(fig_x, [p.t for p in data_dyn], [p.x for p in data_dyn], label = "x-dyn",legend = :outertopright)
    plot!(fig_y, [p.t for p in data_dyn], [p.y for p in data_dyn], label = "y-dyn")
    plot!(fig_v, [p.t for p in data_dyn], [p.v for p in data_dyn], label = "v-dyn")
    plot!(fig_θ, [p.t for p in data_dyn], [p.θ for p in data_dyn], label = "θ-dyn")
    plot!(fig_δ, [p.t for p in data_dyn], [p.δ for p in data_dyn], label = "δ-dyn")
    fig_data = plot(fig_x, fig_y, fig_v, fig_θ, fig_δ, fig_at, layout=(3,2), size=(800,600))
    return fig_data
end

function animate_traj(fig::Plots.Plot, data::Vector{Any}; c::Symbol, label::String, fps::Int64, interval::Int64, L::Float64)
    N = length(data)
    Anim = Animation()
    index_list = collect(1:interval:N)
    push!(index_list, N)
    for i in index_list
        car_shapes = depict_car(Body(), data[i])

        fig_traj = plot(fig, [p.x for p in data[1:i]], [p.y for p in data[1:i]], aspect_ratio = 1, legend = :outertopright, c = c, label="",size=(500,500))
        # fig_traj = plot(fig, [p.x for p in data[1:i]], [p.y for p in data[1:i]], aspect_ratio = 1, title="", c = c, label="")
        render_car!(fig_traj, car_shapes; c=c)
        annotate!(fig_traj, 50, 40, (string("velocity = ", round(data[i].v,digits=3), "m/s"), c, 15))
        fig_traj_zoom = plot(fig, [p.x for p in data[1:i]], [p.y for p in data[1:i]], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, c = c, label=label)
        render_car!(fig_traj_zoom, car_shapes; c=c)
        xlims!(fig_traj_zoom, data[i].x-8*L, data[i].x+8*L)
        ylims!(fig_traj_zoom, data[i].y-5*L, data[i].y+5*L)
        annotate!(fig_traj_zoom, data[i].x, data[i].y + 3*L, (string("velocity = ", round(data[i].v,digits=3), "m/s"), c, 15))
        fig_traj_final = plot(fig_traj, fig_traj_zoom, layout=(2,1), size=(800,1200))

        frame(Anim, fig_traj_zoom)
    end
    return gif(Anim, fps=fps)
end

function animate_traj(fig::Plots.Plot, data_opt::Vector{Any}, data_dyn::Vector{Any}; c1::Symbol, c2::Symbol, fps::Int64, interval::Int64)
    N = length(data_opt)
    Anim = Animation()
    index_list = collect(1:interval:N)
    push!(index_list, N)
    for i in index_list
        car_shapes = depict_car(Body(), data_opt[i])
        fig_traj_opt = plot(fig, [p.x for p in data_opt[1:i]], [p.y for p in data_opt[1:i]], aspect_ratio = 1, title = "Trajectory-Optimization", legend = :outertopright, c = c1, label="opt")
        render_car!(fig_traj_opt, trailer_shapes; c=c1)

        fig_traj_opt_zoom = plot(fig, [p.x for p in data_opt[1:i]], [p.y for p in data_opt[1:i]], aspect_ratio = 1, title = "Trajectory-Optimization", legend = :outertopright, c = c1, label="opt")
        render_car!(fig_traj_opt_zoom, car_shapes; c=c1)
        xlims!(fig_traj_opt_zoom, data_opt[i].x-15, data_opt[i].x+15)
        annotate!(fig_traj_opt_zoom, data_opt[i].x, data[i].y + 10.0, (string("velocity = ", round(data_opt[i].v,digits=3), "m/s"), c1, 20))

        car_shapes = depict_car(Body(), data_dyn[i])
        fig_traj_dyn = plot(fig, [p.x for p in data_dyn[1:i]], [p.y for p in data_dyn[1:i]], aspect_ratio = 1, title = "Trajectory-Kinematics", legend = :outertopright, c = c2, label="dyn")
        render_car!(fig_traj_dyn, trailer_shapes; c=c2)
        fig_traj_dyn_zoom = plot(fig, [p.x for p in data_dyn[1:i]], [p.y for p in data_dyn[1:i]], aspect_ratio = 1, title = "Trajectory-Kinematics", legend = :outertopright, c = c2, label="dyn")
        render_car!(fig_traj_dyn_zoom, car_shapes; c=c2)
        xlims!(fig_traj_dyn_zoom, data_dyn[i].x-15, data_dyn[i].x+15)
        annotate!(fig_traj_dyn_zoom, data_dyn[i].x, data[i].y + 10.0, (string("velocity = ", round(data_dyn[i].v,digits=3), "m/s"), c2, 20))

        fig_traj = plot(fig_traj_opt, fig_traj_dyn, fig_traj_opt_zoom, fig_traj_dyn_zoom, layout=(4,1), size=(1200,1600))
        frame(Anim, fig_traj)
    end
    return gif(Anim, fps=fps)
end