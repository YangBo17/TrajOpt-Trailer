using Revise
using Plots 

includet("../models/trailer1.jl")

function depict_trailer(trailer_size::Body,allstate::AllState)
    d0, d1, w = trailer_size.d0, trailer_size.d1, trailer_size.w  
    x0, y0, θ0 = allstate.x0, allstate.y0, allstate.θ0 
    x1, y1, θ1 = allstate.x1, allstate.y1, allstate.θ1 
    v, ϕ = allstate.v, allstate.ϕ
    trailer_shapes = []
    # Body
    length_margin = 0.20 * d0
    width_margin = 0.20 * w
    α = atan(length_margin, width_margin+w/2)
    β = atan(width_margin+w/2, d0+length_margin)
    angle0 = [θ0-π/2-α, θ0-β, θ0+β, θ0+π/2+α]
    angle1 = [θ1-π/2-α, θ1-β, θ1+β, θ1+π/2+α]
    dist = [sqrt(length_margin^2 + (width_margin + w/2)^2), sqrt((d0+length_margin)^2 + (width_margin + w/2)^2), sqrt((d0+length_margin)^2 + (width_margin + w/2)^2), sqrt(length_margin^2 + (width_margin + w/2)^2)]
    trailer0 = Shape(x0 .+ dist .* cos.(angle0), y0 .+ dist .* sin.(angle0))
    trailer1 = Shape(x1 .+ dist .* cos.(angle1), y1 .+ dist .* sin.(angle1))
    width_pole = 0.05 * w
    angle = [θ1-π/2, θ1-atan(width_pole,d1), θ1+atan(width_pole,d1), θ1+π/2]
    dist_pole = [width_pole, sqrt(d1^2+width_pole^2), sqrt(d1^2+width_pole^2), width_pole]
    pole = Shape(x1 .+ dist_pole .* cos.(angle), y1 .+ dist_pole .* sin.(angle))
    push!(trailer_shapes, trailer0)
    push!(trailer_shapes, trailer1)
    push!(trailer_shapes, pole)
    # Tire 
    width_tire = 0.05 * w  # half
    length_tire = 0.1 * d0 # half
    β = atan(w/2, d0)
    angle0 = [θ0-π/2, θ0-β, θ0+β, θ0+π/2]
    angle1 = [θ1-π/2, θ1+π/2]
    dist0 = [w/2, sqrt((w/2)^2+d0^2), sqrt((w/2)^2+d0^2), w/2]
    dist1 = [w/2, w/2]
    loca0_tire_x = x0 .+ dist0 .* cos.(angle0)
    loca0_tire_y = y0 .+ dist0 .* sin.(angle0)
    loca1_tire_x = x1 .+ dist1 .* cos.(angle1)
    loca1_tire_y = y1 .+ dist1 .* sin.(angle1)
    α = atan(length_tire, width_tire)
    β = atan(width_tire, length_tire)
    d_tire = sqrt(width_tire^2+length_tire^2)
    dist_tire = [d_tire, d_tire, d_tire, d_tire]
    for i in 1:4
        if i ∈ [2,3]
            angle0 = [θ0-π/2-α+ϕ, θ0-β+ϕ, θ0+β+ϕ, θ0+π/2+α+ϕ]
        else
            angle0 = [θ0-π/2-α, θ0-β, θ0+β, θ0+π/2+α]
        end
        tire = Shape(loca0_tire_x[i] .+ dist_tire .* cos.(angle0), loca0_tire_y[i] .+ dist_tire .* sin.(angle0))
        push!(trailer_shapes, tire)
    end
    for i in 1:2
        angle1 = [θ1-π/2-α, θ1-β, θ1+β, θ1+π/2+α]
        tire = Shape(loca1_tire_x[i] .+ dist_tire .* cos.(angle1), loca1_tire_y[i] .+ dist_tire .* sin.(angle1))
        push!(trailer_shapes, tire)
    end
    return trailer_shapes
end

function render_trailer!(fig::Plots.Plot, trailer_shapes; c0::Symbol, c1::Symbol)
    plot!(fig, trailer_shapes[1], c = c0, label="")
    plot!(fig, trailer_shapes[2], c = c1, label="")
    plot!(fig, trailer_shapes[3], c = :black, label="")
    plot!(fig, trailer_shapes[4], c = :black, label="")
    plot!(fig, trailer_shapes[5], c = :black, label="")
    plot!(fig, trailer_shapes[6], c = :black, label="")
    plot!(fig, trailer_shapes[7], c = :black, label="")
    plot!(fig, trailer_shapes[8], c = :black, label="")
    plot!(fig, trailer_shapes[9], c = :black, label="")
    return fig
end

function moose_env(body::Body, sec::Vector{Float64}=[0.0,15.0,45.0,70.0,85.0], scope::Vector{Float64}=[-5.0,5.0,0.0,85.0], width::Vector{Float64}=[-2.0,2.0,2.0,6.0,-2.0,2.0])
    w = body.w
    section1 = collect(sec[1]:0.01:sec[2])
    section3 = collect(sec[3]:0.01:sec[4])
    section5 = collect(sec[5]:0.01:sec[6])
    fig_env = plot(section1, width[1]*ones(size(section1)), c=:black, label="", aspect_ratio = 1)
    plot!(fig_env, section1, width[2]*ones(size(section1)), c=:black, label="")
    plot!(fig_env, section3, width[3]*ones(size(section3)), c=:black, label="")
    plot!(fig_env, section3, width[4]*ones(size(section3)), c=:black, label="")
    plot!(fig_env, section5, width[5]*ones(size(section5)), c=:black, label="")
    plot!(fig_env, section5, width[6]*ones(size(section5)), c=:black, label="")
    plot!(fig_env, collect(range(sec[2],sec[3],100)), collect(range(width[1],width[3],100)), c=:black, label="")
    plot!(fig_env, collect(range(sec[2],sec[3],100)), collect(range(width[2],width[4],100)), c=:black, label="")
    plot!(fig_env, collect(range(sec[4],sec[5],100)), collect(range(width[3],width[5],100)), c=:black, label="")
    plot!(fig_env, collect(range(sec[4],sec[5],100)), collect(range(width[4],width[6],100)), c=:black, label="")
    ylims!(fig_env, scope[1], scope[2])
    xlims!(fig_env, scope[3]-1.0, scope[4]+body.d1+body.d0+1.0)
    return fig_env
end

function visual_traj(data::Vector{Any}, num::Int64, c0::Symbol, c1::Symbol, label::String)
    N = length(data)
    interval_frame = Int64(N / (num+1)) 
    index_frame = [1 + i * interval_frame for i in 0:num]
    push!(index_frame, N)
    fig_traj = plot([p.x0 for p in data], [p.y0 for p in data], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, c = c0, label=string(label,"-0"))
    plot!(fig_traj, [p.x1 for p in data], [p.y1 for p in data], c = c1, label=string(label,"-1"))
    for i in index_frame
        trailer_shapes = depict_trailer(Body(), data[i])
        render_trailer!(fig_traj, trailer_shapes; c0=c0, c1=c1)
    end
    return fig_traj
end

function visual_traj(fig::Plots.Plot, data::Vector{Any}; num::Int64, c0::Symbol, c1::Symbol, label::String)
    N = length(data)
    interval_frame = Int64(floor(N / (num+1))) 
    index_frame = [1 + i * interval_frame for i in 0:num]
    push!(index_frame, N)
    fig_traj = plot(fig, [p.x0 for p in data], [p.y0 for p in data], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, c = c0, label=string(label,"-0"))
    plot!(fig_traj, [p.x1 for p in data], [p.y1 for p in data], c = c1, label=string(label,"-1"))
    for i in index_frame
        trailer_shapes = depict_trailer(Body(), data[i])
        render_trailer!(fig_traj, trailer_shapes; c0=c0, c1=c1)
    end
    return fig_traj
end

function visual_data(data_opt::Vector{Any}, data_dyn::Vector{Any}, dt::Float64=0.01)
    N = length(data_opt)
    fig_x1 = plot((1:N) * dt, [p.x1 for p in data_opt], label = "x1-opt",legend = :outertopright)
    fig_y1 = plot((1:N) * dt, [p.y1 for p in data_opt], label = "y1-opt",legend = :outertopright)
    fig_x0 = plot((1:N) * dt, [p.x0 for p in data_opt], label = "x0-opt",legend = :outertopright)
    fig_y0 = plot((1:N) * dt, [p.y0 for p in data_opt], label = "y0-opt",legend = :outertopright)
    fig_vx1 = plot((1:N) * dt, [p.vx1 for p in data_opt], label = "vx1-opt",legend = :outertopright)
    fig_vy1 = plot((1:N) * dt, [p.vy1 for p in data_opt], label = "vy1-opt",legend = :outertopright)
    fig_vx0 = plot((1:N) * dt, [p.vx0 for p in data_opt], label = "vx0-opt",legend = :outertopright)
    fig_vy0 = plot((1:N) * dt, [p.vy0 for p in data_opt], label = "vy0-opt",legend = :outertopright)
    fig_θ1 = plot((1:N) * dt, [p.θ1 for p in data_opt], label = "θ1-opt",legend = :outertopright)
    fig_θ0 = plot((1:N) * dt, [p.θ0 for p in data_opt], label = "θ0-opt",legend = :outertopright)
    fig_v = plot((1:N) * dt, [p.v for p in data_opt], label = "v-opt",legend = :outertopright)
    fig_ϕ = plot((1:N) * dt, [p.ϕ for p in data_opt], label = "ϕ-opt",legend = :outertopright)
    plot!(fig_x1, (1:N) * dt, [p.x1 for p in data_dyn], label = "x1_dyn",legend = :outertopright)
    plot!(fig_y1, (1:N) * dt, [p.y1 for p in data_dyn], label = "y1_dyn")
    plot!(fig_x0, (1:N) * dt, [p.x0 for p in data_dyn], label = "x0_dyn")
    plot!(fig_y0, (1:N) * dt, [p.y0 for p in data_dyn], label = "y0_dyn")
    plot!(fig_vx1, (1:N) * dt, [p.vx1 for p in data_dyn], label = "vx1_dyn")
    plot!(fig_vy1, (1:N) * dt, [p.vy1 for p in data_dyn], label = "vy1_dyn")
    plot!(fig_vx0, (1:N) * dt, [p.vx0 for p in data_dyn], label = "vx0_dyn")
    plot!(fig_vy0, (1:N) * dt, [p.vy0 for p in data_dyn], label = "vy0_dyn")
    plot!(fig_θ1, (1:N) * dt, [p.θ1 for p in data_dyn], label = "θ1_dyn")
    plot!(fig_θ0, (1:N) * dt, [p.θ0 for p in data_dyn], label = "θ0_dyn")
    plot!(fig_v, (1:N) * dt, [p.v for p in data_dyn], label = "v_dyn")
    plot!(fig_ϕ, (1:N) * dt, [p.ϕ for p in data_dyn], label = "ϕ_dyn")

    fig_data = plot(fig_x1, fig_y1, fig_x0, fig_y0, fig_vx1, fig_vy1, fig_vx0, fig_vy0, fig_θ1, fig_θ0, fig_v, fig_ϕ, layout=(3,4), size=(1200,800))
    return fig_data
end

function animate_traj(fig::Plots.Plot, data::Vector{Any}; c0::Symbol, c1::Symbol, label::String, fps::Int64, interval::Int64)
    N = length(data)
    Anim = Animation()
    index_list = collect(1:interval:N)
    push!(index_list, N)
    for i in index_list
        trailer_shapes = depict_trailer(Body(), data[i])

        fig_traj = plot(fig, [p.x0 for p in data[1:i]], [p.y0 for p in data[1:i]], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, c = c0, label=string(label,"-0"))
        plot!(fig_traj, [p.x1 for p in data[1:i]], [p.y1 for p in data[1:i]], c = c1, label=string(label,"-1"))
        render_trailer!(fig_traj, trailer_shapes; c0=c0, c1=c1)

        fig_traj_zoom = plot(fig, [p.x0 for p in data[1:i]], [p.y0 for p in data[1:i]], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, c = c0, label=string(label,"-0"))
        plot!(fig_traj_zoom, [p.x1 for p in data[1:i]], [p.y1 for p in data[1:i]], c = c1, label=string(label,"-1"))
        render_trailer!(fig_traj_zoom, trailer_shapes; c0=c0, c1=c1)
        xlims!(fig_traj_zoom, data[i].x0-15, data[i].x0+15)
        annotate!(fig_traj_zoom, data[i].x0, 6.0, (string("velocity = ", round(data[i].v,digits=3), "m/s"), c0, 20))
        fig_traj_final = plot(fig_traj, fig_traj_zoom, layout=(2,1), size=(1200,800))
        frame(Anim, fig_traj_final)
    end
    return gif(Anim, fps=fps)
end

function animate_traj(fig::Plots.Plot, data_opt::Vector{Any}, data_dyn::Vector{Any}; c0::Symbol, c1::Symbol, c2::Symbol, c3::Symbol, fps::Int64, interval::Int64)
    N = length(data_opt)
    Anim = Animation()
    index_list = collect(1:interval:N)
    push!(index_list, N)
    for i in index_list
        trailer_shapes = depict_trailer(Body(), data_opt[i])
        fig_traj_opt = plot(fig, [p.x0 for p in data_opt[1:i]], [p.y0 for p in data_opt[1:i]], aspect_ratio = 1, title = "Trajectory-Optimization", legend = :outertopright, c = c0, label="opt-0")
        plot!(fig_traj_opt, [p.x1 for p in data_opt[1:i]], [p.y1 for p in data_opt[1:i]], c = c1, label="opt-1")
        render_trailer!(fig_traj_opt, trailer_shapes; c0=c0, c1=c1)
        fig_traj_opt_zoom = plot(fig, [p.x0 for p in data_opt[1:i]], [p.y0 for p in data_opt[1:i]], aspect_ratio = 1, title = "Trajectory-Optimization", legend = :outertopright, c = c0, label="opt-0")
        plot!(fig_traj_opt_zoom, [p.x1 for p in data_opt[1:i]], [p.y1 for p in data_opt[1:i]], c = c1, label="opt-1")
        render_trailer!(fig_traj_opt_zoom, trailer_shapes; c0=c0, c1=c1)
        xlims!(fig_traj_opt_zoom, data_opt[i].x0-15, data_opt[i].x0+15)
        annotate!(fig_traj_opt_zoom, data_opt[i].x0, 6.0, (string("velocity = ", round(data_opt[i].v,digits=3), "m/s"), c0, 20))

        trailer_shapes = depict_trailer(Body(), data_dyn[i])
        fig_traj_dyn = plot(fig, [p.x0 for p in data_dyn[1:i]], [p.y0 for p in data_dyn[1:i]], aspect_ratio = 1, title = "Trajectory-Kinematics", legend = :outertopright, c = c2, label="dyn-0")
        plot!(fig_traj_dyn, [p.x1 for p in data_dyn[1:i]], [p.y1 for p in data_dyn[1:i]], c = c3, label="dyn-1")
        render_trailer!(fig_traj_dyn, trailer_shapes; c0=c2, c1=c3)
        fig_traj_dyn_zoom = plot(fig, [p.x0 for p in data_dyn[1:i]], [p.y0 for p in data_dyn[1:i]], aspect_ratio = 1, title = "Trajectory-Kinematics", legend = :outertopright, c = c2, label="dyn-0")
        plot!(fig_traj_dyn_zoom, [p.x1 for p in data_dyn[1:i]], [p.y1 for p in data_dyn[1:i]], c = c3, label="dyn-1")
        render_trailer!(fig_traj_dyn_zoom, trailer_shapes; c0=c2, c1=c3)
        xlims!(fig_traj_dyn_zoom, data_dyn[i].x0-15, data_dyn[i].x0+15)
        annotate!(fig_traj_dyn_zoom, data_dyn[i].x0, 6.0, (string("velocity = ", round(data_dyn[i].v,digits=3), "m/s"), c2, 20))

        fig_traj = plot(fig_traj_opt, fig_traj_dyn, fig_traj_opt_zoom, fig_traj_dyn_zoom, layout=(4,1), size=(1200,1600))
        frame(Anim, fig_traj)
    end
    return gif(Anim, fps=fps)
end