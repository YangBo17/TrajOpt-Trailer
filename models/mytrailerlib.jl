module mytrailerlib
```
- tractor-trialer dynamics
- tractor trialer visualization
```

using DifferentialEquations
using Parameters
using Plots

export FullState
export Body

struct State
    x0::Float64
    y0::Float64
    θ0::Float64
    θ1::Float64
end

struct StateDerivative
    dx0::Float64
    dy0::Float64 
    dθ0::Float64
    dθ1::Float64
end

struct Input
    v::Float64
    ϕ::Float64
end

struct FullState
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

@with_kw mutable struct Body # standard tractor-trailer
    d0::Float64 = 0.35 # [m] wheel base, distance from tractor's rear axis to tractor's front axis
    d1::Float64 = 0.85 # [m] distance from tractor's rear axis to trailer's rear axis
    dl::Float64 = 0.60 # length of trailer
    w::Float64 = 0.26 # [m] width of vehicle
    tr::Float64 = 0.05 # [m] tire radius
    tw::Float64 = 0.05 # [m] tire width
    margin::Float64 = 0.2 # distance ratio from tire to vehicle boundary, the basis is w and d0
    scale::Float64 = 0.1 # scale of different platform, 1:1 or 1:10
end

@with_kw mutable struct MyTrailer
    body::Body = Body()
    state::State
    input::Input
    time::Real = 0.
end

function Body(body::Body, scale::Float64)::Body
    d0 = body.d0 * scale / body.scale
    d1 = body.d1 * scale / body.scale
    dl = body.dl * scale / body.scale
    w = body.w * scale / body.scale
    tr = body.tr * scale / body.scale
    tw = body.tw * scale / body.scale
    return Body(d0, d1, dl, w, tr, tw, body.margin, scale)
end

# Tractor-Trailer Dynamics
function to_array(s::State)::Vector
    return [s.x0, s.y0, s.θ0, s.θ1]
end

function to_array(st::StateDerivative)::Vector
    return [st.dx0, st.dy0, st.dθ0, st.dθ1]
end

function to_array(u::Input)::Vector
    return [u.v, u.ϕ]
end

function dynamics(s::Array, p::Body, u::Array)::StateDerivative
    x0 = s[1]
    y0 = s[2]
    θ0 = s[3]
    θ1 = s[4]
    v = u[1]
    ϕ = u[2]
    dx0 = cos(θ0) * v
    dy0 = sin(θ0) * v
    dθ0 = 1 / p.d0 * tan(ϕ) * v
    dθ1 = 1 / p.d1 * sin(θ0 - θ1) * v
    return StateDerivative(dx0, dy0, dθ0, dθ1)
end

function trailer_step(mytrailer::MyTrailer, u::Input, dt::Real)
    s = mytrailer.state
    # p = mytrailer.body
    mytrailer.input = u
    mytrailer.time += dt
    tspan = (0.0, dt)
    function ode_dyn!(ds::Array, s::Array, u::Array, t)
        st_deriv = dynamics(s, mytrailer.body, u)
        ds[1:end] = to_array(st_deriv)
    end
    prob = ODEProblem(ode_dyn!, to_array(s), tspan, to_array(u))
    sol = DifferentialEquations.solve(prob)
    new_state = State(sol(dt)...)
    mytrailer.state = new_state
end

# Tractor-trailer Plot
function depict_trailer(body::Body, state::FullState)
    d0, d1, dl, w = body.d0, body.d1, body.dl, body.w 
    x0, y0, θ0 = state.x0, state.y0, state.θ0
    x1, y1, θ1 = state.x1, state.y1, state.θ1
    x1 = x0 - d1 * cos(θ1)
    y1 = y0 - d1 * sin(θ1)
    margin = body.margin
    # body
    length_margin = margin * d0
    width_margin = margin * w
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
    # tire
    width_tire = body.tw  # half
    length_tire = 2 * body.tr  # half
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

function visual_traj(fig::Plots.Plot, data::Vector{Any}; num::Int64, c0::Symbol, c1::Symbol, label::String)
    N = length(data)
    interval_frame = Int64(floor(N / (num+1))) 
    index_frame = [1 + i * interval_frame for i in 0:num]
    push!(index_frame, N)
    fig_traj = plot(fig, [p.x0 for p in data], [p.y0 for p in data], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, color = c0, label=string(label,"-front"))
    plot!(fig_traj, [p.x1 for p in data], [p.y1 for p in data], color = c1, label=string(label,"-rear"))
    for i in index_frame
        trailer_shapes = depict_trailer(Body(), data[i])
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
        trailer_shapes = depict_trailer(Body(), data[i])

        fig_traj = plot(fig, [p.x0 for p in data[1:i]], [p.y0 for p in data[1:i]], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, color = c0, label=s"")
        plot!(fig_traj, [p.x1 for p in data[1:i]], [p.y1 for p in data[1:i]], color = c1, label="")
        render_trailer!(fig_traj, trailer_shapes; c0=c0, c1=c1)
        annotate!(fig_traj, 5, 2, (string("velocity = ", round(data[i].v,digits=3), "m/s"), c0, 15))

        fig_traj_zoom = plot(fig, [p.x0 for p in data[1:i]], [p.y0 for p in data[1:i]], aspect_ratio = 1, title = "Trajectory", legend = :outertopright, c = c0, label="")
        plot!(fig_traj_zoom, [p.x1 for p in data[1:i]], [p.y1 for p in data[1:i]], c = c1, label="")
        render_trailer!(fig_traj_zoom, trailer_shapes; c0=c0, c1=c1)
        xlims!(fig_traj_zoom, data[i].x0-15, data[i].x0+15)
        annotate!(fig_traj_zoom, data[i].x0, 6.0, (string("velocity = ", round(data[i].v, digits=3), "m/s"), c0, 20))
        fig_traj_final = plot(fig_traj, fig_traj_zoom, layout=(2,1), size=(800,800))
        frame(Anim, fig_traj)
    end
    return gif(Anim, fps=fps)
end

end