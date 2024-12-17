## env
using JLD2
includet("../../common/common.jl")
@load "data/cons_limits2.jld2" cons_limits
@load "data/cons_obs2.jld2" cons_obs
includet("../../visual/visual_trailer.jl")
trailer_body = TrailerBody()

# trailer scalability env
init_pos = [0.0, 0.0]
routes = [init_pos]
xdistance = 4.0 .+ (6.0 - 4.0) * rand(segment)
ydistance = 4.0 .+ (6.0 - 4.0) * rand(segment)
ydirec = -1.0 .+ 2.0 * rand(segment)
ydirec = sign.(ydirec)

Ts = Float64[]
vel_scale = 1.5
for i in 1:segment
    if i%8 == 1
        push!(routes, routes[end] + [+xdistance[i], 0])
        push!(Ts, abs(xdistance[i]) / vel_scale)
    elseif i%8 == 2
        push!(routes, routes[end] + [0, ydirec[i]*ydistance[i]])
        push!(Ts, abs(ydistance[i]) / vel_scale)
    elseif i%8 == 3
        push!(routes, routes[end] + [+xdistance[i], 0])
        push!(Ts, abs(xdistance[i]) / vel_scale)
    elseif i%8 == 4
        push!(routes, routes[end] + [0, ydirec[i]*ydistance[i]])
        push!(Ts, abs(ydistance[i]) / vel_scale)
    elseif i%8 == 5
        push!(routes, routes[end] + [+xdistance[i], 0])
        push!(Ts, abs(xdistance[i]) / vel_scale)
    elseif i%8 == 6
        push!(routes, routes[end] + [0, ydirec[i]*ydistance[i]])
        push!(Ts, abs(ydistance[i]) / vel_scale)
    elseif i%8 == 7
        push!(routes, routes[end] + [+xdistance[i], 0])
        push!(Ts, abs(xdistance[i]) / vel_scale)
    elseif i%8 == 0
        push!(routes, routes[end] + [0, ydirec[i]*ydistance[i]])
        push!(Ts, abs(ydistance[i]) / vel_scale)
    end
end
route = Matrix(hcat(routes...)')
route = round.(route, digits=3)
T0 = Ts
# @show route

# parameter
d0 = trailer_body.trailer_length
d1 = trailer_body.link_length
Wid = trailer_body.trailer_width
Len = max(trailer_body.trailer_length, trailer_body.trailer2_length)
Lt = Len * 1.4
Wt = Wid * 1.4
dr = sqrt(Lt^2+Wt^2) / 2

#
seg_N = length(route[:,1]) - 1
sdp_N = seg_N
sdp_s = 3
corridors, obstacles, border = gen_env(route)
cons_init = gen_init(route)
fig_env = plot_env(route, corridors, obstacles, cons_init, border)
limits_list, obs_list = gen_cons(route, corridors, cons_limits, cons_obs)

L1 = maximum([length(obs_list[i].B) for i in eachindex(obs_list)])
L2 = maximum([length(limits_list[i].B) for i in eachindex(limits_list)])

function gen_start_goal(cons_init::Cons_Init)
    s_x = cons_init.pos[1,1] + d1*cons_init.vel[1,1]/norm(cons_init.vel[1,:])
    s_y = cons_init.pos[1,2] + d1*cons_init.vel[1,2]/norm(cons_init.vel[1,:])
    s_vx = cons_init.vel[1,1]
    s_vy = cons_init.vel[1,2]
    s_ψ = atan(s_vy, s_vx)
    s_v = sqrt(s_vx^2 + s_vy^2)

    g_x = cons_init.pos[end,1] + d1*cons_init.vel[end,1]/norm(cons_init.vel[end,:]) 
    g_y = cons_init.pos[end,2] + d1*cons_init.vel[end,2]/norm(cons_init.vel[end,:])
    g_vx = cons_init.vel[end,1]
    g_vy = cons_init.vel[end,2]
    g_ψ = atan(g_vy, g_vx)
    g_v = sqrt(g_vx^2 + g_vy^2)

    start = [s_x, s_y, s_ψ, s_ψ, s_v]
    goal = [g_x, g_y, g_ψ, g_ψ, g_v]
    return start, goal
end
start, goal = gen_start_goal(cons_init)

function plot_bnd_cir(fig, obs_bnd_x, obs_bnd_y, obs_bnd_r)
        for i in eachindex(obs_bnd_r)
                θ = LinRange(0, 2*π, 100) # 生成100个点来近似圆形
                cir_x = obs_bnd_x[i] .+ obs_bnd_r[i] .* cos.(θ)
                cir_y = obs_bnd_y[i] .+ obs_bnd_r[i] .* sin.(θ)
                plot!(fig, cir_x, cir_y, color=:gray, fill=(0, :gray), label="")
        end
end

# limits
κ_max = 2.0
jackknife_buffer = 0.5
δ_min = -atan(d0*κ_max)
δ_max = atan(d0*κ_max)
v_min = 0.5
v_max = +2.0
a_min = -2.0
a_max = 2.0

# trust region for Li2022RAL
Δs = 0.15
Δa = 0.15

struct Traj_trailer
    t::Array
    x0::Array
    y0::Array
    ψ0::Array
    x1::Array
    y1::Array
    ψ1::Array
    v0::Array
    a0::Array
    δ0::Array
    v1::Array
end