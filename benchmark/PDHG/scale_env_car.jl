## env
using JLD2
includet("../../common/common.jl")
@load "data/car_cons_limits2.jld2" cons_limits
@load "data/car_cons_obs2.jld2" cons_obs
include("../../visual/visual_car.jl")
car_body = Body()

# trailer scalability env
init_pos = [0, 0]
routes = [init_pos]
distance = 4.0
for i in 1:N
    if i%8 == 1
        push!(routes, routes[end] + [+distance, 0])
    elseif i%8 == 2
        push!(routes, routes[end] + [0, +distance])
    elseif i%8 == 3
        push!(routes, routes[end] + [+distance, 0])
    elseif i%8 == 4
        push!(routes, routes[end] + [0, -distance])
    elseif i%8 == 5
        push!(routes, routes[end] + [+distance, 0])
    elseif i%8 == 6
        push!(routes, routes[end] + [0, -distance])
    elseif i%8 == 7
        push!(routes, routes[end] + [+distance, 0])
    elseif i%8 == 0
        push!(routes, routes[end] + [0, +distance])
    end
end
route = Matrix(hcat(routes...)')
T0 = ones(N) * distance/1.0

# parameter
seg_N = length(route[:,1]) - 1
sdp_N = seg_N
sdp_s = 2
corridors, obstacles, border = gen_env(route)
cons_init = gen_init(route)
fig_env = plot_env(route, corridors, obstacles, cons_init, border)
limits_list, obs_list = gen_cons_car(route, corridors, cons_limits, cons_obs)

L1 = maximum([length(obs_list[i].B) for i in eachindex(obs_list)])
L2 = maximum([length(limits_list[i].B) for i in eachindex(limits_list)])

function gen_start_goal(cons_init::Cons_Init)
    s_x = cons_init.pos[1,1] 
    s_y = cons_init.pos[1,2] 
    s_vx = cons_init.vel[1,1]
    s_vy = cons_init.vel[1,2]
    s_ψ = atan(s_vy, s_vx)
    s_v = sqrt(s_vx^2 + s_vy^2)

    g_vx = cons_init.vel[end,1]
    g_vy = cons_init.vel[end,2]
    g_ψ = atan(g_vy, g_vx)
    g_x = cons_init.pos[end,1] + g_vx
    g_y = cons_init.pos[end,2] + g_vy
    g_v = sqrt(g_vx^2 + g_vy^2)

    start = [s_x, s_y, s_ψ, s_v]
    goal = [g_x, g_y, g_ψ, g_v]
    return start, goal
end
start, goal = gen_start_goal(cons_init)


L = 0.353 # wheelbase
W = 0.264 # width
Lt = L * 1.4
Wt = W * 1.4

rectangle(x_min, x_max, y_min, y_max) = Shape([x_min, x_max, x_max, x_min], [y_min, y_min, y_max, y_max])
function ellipse(x0, y0, ψ, a, b)
    t = range(0, stop=2π, length=100)
    x = x0 .+ a .* cos.(t .+ ψ)
    y = y0 .+ b .* sin.(t .+ ψ)
    return Shape(x, y)
end

function plot_car_trans!(fig_env, car_trans, car_cons)
    for i in eachindex(car_cons[:,1])
        plot!(fig_env, rectangle(car_cons[i,:]...), fillcolor=:gray, label="")
    end
    for i in eachindex(car_trans)
        plot!(fig_env, ellipse(car_trans[i].x, car_trans[i].y, car_trans[i].ψ, ae, be), color=:black, fillcolor=:gray, fillopacity=0.3, label="")
    end
    return fig_env
end

function plot_bnd_cir(fig, obs_bnd_x, obs_bnd_y, obs_bnd_r)
    for i in eachindex(obs_bnd_r)
        θ = LinRange(0, 2*π, 100) # 生成100个点来近似圆形
        cir_x = obs_bnd_x[i] .+ obs_bnd_r[i] .* cos.(θ)
        cir_y = obs_bnd_y[i] .+ obs_bnd_r[i] .* sin.(θ)
        plot!(fig, cir_x, cir_y, color=:gray, fill=(0, :gray), label="")
    end
end

δ_min = -0.7
δ_max = 0.7
v_min = 0.5
v_max = +2.0
a_min = -2.0
a_max = 2.0

dδ_min = -0.18
dδ_max = 0.18
da_min = -0.5
da_max = 0.5

# MILP settings
vx_min = 0.0
vx_max = 2.0
vy_min = -2.0
vy_max = 2.0
ax_min = -2.0
ax_max = 2.0
ay_min = -2.0
ay_max = +2.0


ωx = 1.0
ωy = 1.0
ωv = 1.0
ωψ = 1.0
ωa = 10.0
ωδ = 10.0

Ωx = 1.0
Ωy = 1.0
Ωvx = 0.1
Ωvy = 0.1
Ωax = 0.03
Ωay = 0.03

struct Traj_car
    t::Array
    x::Array
    y::Array
    ψ::Array
    v::Array
    δ::Array
    a::Array
end