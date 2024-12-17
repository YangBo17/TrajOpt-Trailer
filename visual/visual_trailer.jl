using Revise
using Plots
using Parameters

@with_kw struct TrailerBody
    trailer_length::Float64 = 0.35
    trailer_width::Float64 = 0.26
    trailer2_length::Float64 = 0.60
    link_length::Float64 = 0.856
    car_length::Float64 = trailer_length + link_length
end

@with_kw struct TrailerBodyReal
    trailer_length::Float64 = 0.35 *10
    trailer_width::Float64 = 0.26 * 10
    trailer2_length::Float64 = 0.60 * 10
    link_length::Float64 = 0.856 * 10
    car_length::Float64 = trailer_length + link_length
end

struct TrailerState
    t::Float64
    x0::Float64
    y0::Float64
    vx0::Float64
    vy0::Float64
    ax0::Float64
    ay0::Float64
    x1::Float64
    y1::Float64
    vx1::Float64
    vy1::Float64
    ax1::Float64
    ay1::Float64
    jx1::Float64
    jy1::Float64
    θ0::Float64
    θ1::Float64
    v::Float64
    a::Float64
    ϕ::Float64
    κ::Float64
end

function depict_trailer(trailer_body,trailer_state::Vector{Float64})
    d0, d1, dl, w = trailer_body.trailer_length, trailer_body.trailer2_length, trailer_body.link_length, trailer_body.trailer_width
    x0,y0,θ0,x1,y1,θ1,ϕ = trailer_state
    trailer_shapes = []
    # Body
    length_margin = 0.20 * d0
    width_margin = 0.20 * w
    α = atan(length_margin, width_margin+w/2)
    β0 = atan(width_margin+w/2, d0+length_margin)
    β1 = atan(width_margin+w/2, d1+length_margin)
    angle0 = [θ0-π/2-α, θ0-β0, θ0+β0, θ0+π/2+α, θ0-π/2-α]
    angle1 = [θ1-π/2-α, θ1-β1, θ1+β1, θ1+π/2+α, θ1-π/2-α]
    dist0 = [sqrt(length_margin^2 + (width_margin + w/2)^2), sqrt((d0+length_margin)^2 + (width_margin + w/2)^2), sqrt((d0+length_margin)^2 + (width_margin + w/2)^2), sqrt(length_margin^2 + (width_margin + w/2)^2), sqrt(length_margin^2 + (width_margin + w/2)^2)]
    dist1 = [sqrt(length_margin^2 + (width_margin + w/2)^2), sqrt((d1+length_margin)^2 + (width_margin + w/2)^2), sqrt((d1+length_margin)^2 + (width_margin + w/2)^2), sqrt(length_margin^2 + (width_margin + w/2)^2), sqrt(length_margin^2 + (width_margin + w/2)^2)]
    trailer0 = Shape(x0 .+ dist0 .* cos.(angle0), y0 .+ dist0 .* sin.(angle0))
    trailer1 = Shape(x1 .+ dist1 .* cos.(angle1), y1 .+ dist1 .* sin.(angle1))
    width_pole = 0.05 * w
    angle = [θ1-π/2, θ1-atan(width_pole,dl), θ1+atan(width_pole,dl), θ1+π/2]
    dist_pole = [width_pole, sqrt(dl^2+width_pole^2), sqrt(dl^2+width_pole^2), width_pole]
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

function depict_trailer(trailer_body::TrailerBody, trailer_state::TrailerState)
    d0, d1, dl, w = trailer_body.trailer_length, trailer_body.trailer2_length, trailer_body.link_length, trailer_body.trailer_width
    x0, y0, θ0 = trailer_state.x0, trailer_state.y0, trailer_state.θ0
    x1, y1, θ1 = trailer_state.x1, trailer_state.y1, trailer_state.θ1
    v, ϕ = trailer_state.v, trailer_state.ϕ
    trailer_shapes = []
    # Body
    length_margin = 0.20 * d0
    width_margin = 0.20 * w
    α = atan(length_margin, width_margin+w/2)
    β0 = atan(width_margin+w/2, d0+length_margin)
    β1 = atan(width_margin+w/2, d1+length_margin)
    angle0 = [θ0-π/2-α, θ0-β0, θ0+β0, θ0+π/2+α]
    angle1 = [θ1-π/2-α, θ1-β1, θ1+β1, θ1+π/2+α]
    dist0 = [sqrt(length_margin^2 + (width_margin + w/2)^2), sqrt((d0+length_margin)^2 + (width_margin + w/2)^2), sqrt((d0+length_margin)^2 + (width_margin + w/2)^2), sqrt(length_margin^2 + (width_margin + w/2)^2)]
    dist1 = [sqrt(length_margin^2 + (width_margin + w/2)^2), sqrt((d1+length_margin)^2 + (width_margin + w/2)^2), sqrt((d1+length_margin)^2 + (width_margin + w/2)^2), sqrt(length_margin^2 + (width_margin + w/2)^2)]
    trailer0 = Shape(x0 .+ dist0 .* cos.(angle0), y0 .+ dist0 .* sin.(angle0))
    trailer1 = Shape(x1 .+ dist1 .* cos.(angle1), y1 .+ dist1 .* sin.(angle1))
    width_pole = 0.05 * w
    angle = [θ1-π/2, θ1-atan(width_pole,dl), θ1+atan(width_pole,dl), θ1+π/2]
    dist_pole = [width_pole, sqrt(dl^2+width_pole^2), sqrt(dl^2+width_pole^2), width_pole]
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

function render_trailer!(fig::Plots.Plot, trailer_shapes; c::Symbol)
    plot!(fig, trailer_shapes[1], c = c, label="")
    plot!(fig, trailer_shapes[2], c = c, label="")
    plot!(fig, trailer_shapes[3], c = :black, label="")
    plot!(fig, trailer_shapes[4], c = :black, label="")
    plot!(fig, trailer_shapes[5], c = :black, label="")
    plot!(fig, trailer_shapes[6], c = :black, label="")
    plot!(fig, trailer_shapes[7], c = :black, label="")
    plot!(fig, trailer_shapes[8], c = :black, label="")
    plot!(fig, trailer_shapes[9], c = :black, label="")
    return fig
end



function visual_traj(fig::Plots.Plot, data::Vector{Any}; num::Int64, c0::Symbol, c1::Symbol, label::String)
    N = length(data)
    interval_frame = Int64(floor(N / (num+1))) 
    index_frame = [1 + i * interval_frame for i in 0:num]
    # push!(index_frame, N)
    # fig_traj = plot(fig, [p.x0 for p in data], [p.y0 for p in data], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, color = c0, label=string(label,"-front"))
    # plot!(fig_traj, [p.x1 for p in data], [p.y1 for p in data], color = c1, label=string(label,"-rear"))
    fig_traj = plot(fig, [p.x0 for p in data], [p.y0 for p in data], title = "", legend = :outertopright, color = c0, label="", linewidth=5)
    plot!(fig_traj, [p.x1 for p in data], [p.y1 for p in data], color = c1, label="", linewidth=5)
    for i in index_frame
        trailer_shapes = depict_trailer(TrailerBody(), data[i])
        render_trailer!(fig_traj, trailer_shapes; c0=c0, c1=c1)
    end
    return fig_traj
end

function animate_traj(fig::Plots.Plot, data::Vector{Any}; c0::Symbol, c1::Symbol, label::String, fps::Int64, interval::Int64)
    N = length(data)
    Anim = Animation()
    index_list = collect(1:interval:N)
    push!(index_list, N)
    for i in index_list
        trailer_shapes = depict_trailer(TrailerBody(), data[i])

        fig_traj = plot(fig, [p.x0 for p in data[1:i]], [p.y0 for p in data[1:i]], aspect_ratio = 1, legend = :outertopright, color = c0, label=s"")
        plot!(fig_traj, [p.x1 for p in data[1:i]], [p.y1 for p in data[1:i]], color = c1, label="")
        render_trailer!(fig_traj, trailer_shapes; c0=c0, c1=c1)
        # annotate!(fig_traj, 5, 2, (string("velocity = ", round(data[i].v,digits=3), "m/s"), c0, 15))

        fig_traj_zoom = plot(fig, [p.x0 for p in data[1:i]], [p.y0 for p in data[1:i]], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, c = c0, label="")
        plot!(fig_traj_zoom, [p.x1 for p in data[1:i]], [p.y1 for p in data[1:i]], c = c1, label="")
        render_trailer!(fig_traj_zoom, trailer_shapes; c0=c0, c1=c1)
        xlims!(fig_traj_zoom, data[i].x0-15, data[i].x0+15)
        annotate!(fig_traj_zoom, data[i].x0, 6.0, (string("velocity = ", round(data[i].v,digits=3), "m/s"), c0, 20))
        fig_traj_final = plot(fig_traj, fig_traj_zoom, layout=(2,1), size=(800,800))
        frame(Anim, fig_traj)
    end
    return gif(Anim, fps=fps)
end