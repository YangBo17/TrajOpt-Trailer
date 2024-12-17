using JLD2
includet("../../common/common.jl")
@load "data/car_cons_limits2.jld2" cons_limits
@load "data/car_cons_obs2.jld2" cons_obs
include("../../visual/visual_car.jl")
car_body = Body()
# car env
route = [1.0 0.0;
        11.0 0.0
]

# seg_N = length(route[:,1]) - 1
# sdp_N = seg_N
seg_N = 5
sdp_N = 5
sdp_s = 3
T_total = 7.0
T0 = ones(seg_N) * T_total / seg_N
corridors, obstacles, border = gen_env(route)
cons_init = gen_init(route)
fig_env = plot_env(route, corridors, obstacles, cons_init, border)
limits_list, obs_list = gen_cons_car(route, corridors, cons_limits, cons_obs)

L1 = maximum([length(obs_list[i].B) for i in eachindex(obs_list)])
L2 = maximum([length(limits_list[i].B) for i in eachindex(limits_list)])

function gen_start_goal(cons_init::Cons_Init)
    s_x = cons_init.pos[1,1] 
    s_y = cons_init.pos[1,2] - 0.5
    s_vx = cons_init.vel[1,1]
    s_vy = cons_init.vel[1,2]
    s_ψ = atan(s_vy, s_vx)
    s_v = sqrt(s_vx^2 + s_vy^2)

    g_vx = cons_init.vel[end,1]
    g_vy = cons_init.vel[end,2]
    g_ψ = atan(g_vy, g_vx)
    g_x = cons_init.pos[end,1] + g_vx 
    g_y = cons_init.pos[end,2] + g_vy + 0.5
    g_v = sqrt(g_vx^2 + g_vy^2)

    start = [s_x, s_y, s_ψ, s_v]
    goal = [g_x, g_y, g_ψ, g_v]
    return start, goal
end
start, goal = gen_start_goal(cons_init)
# start += [0.0, -0.5, 0.0, 0.0]
# goal += [0.0, +0.5, 0.0, 0.0]
L = 0.353 # wheelbase
W = 0.264 # width
Lt = L * 1.4
Wt = W * 1.4
ae = 0.9
be = 0.9
bl = +0.9
br = -0.9
env = [0.0, 12.0, br, bl]
struct CarPos
    x::Float64
    y::Float64
    ψ::Float64
    v::Float64
end
# initial condition for Eira2022TRO
start_line = CarPos(0.0, -0.5, 0.0, 1.0)
goal_line = CarPos(12.0, +0.5, 0.0, 1.0)

car1_line = CarPos(3.5, -0.5, 0.0, 0.0)
car2_line = CarPos(8.5, 0.5, 0.0, 0.0)
car_line = [car1_line, car2_line]
Lcar = Lt/1.5
Wcar = Wt/1.5
cons_line = [
    car1_line.x-Lcar car1_line.x+Lcar car1_line.y-Wcar car1_line.y+Wcar;
    car2_line.x-Lcar car2_line.x+Lcar car2_line.y-Wcar car2_line.y+Wcar;
]
A = [[1, 1], [-1, 1], [-1, -1], [1, -1]]


rectangle(x_min, x_max, y_min, y_max) = Shape([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min])
function ellipse(x0, y0, ψ, a, b)
    t = range(0, stop=2π, length=100)
    x = x0 .+ a .* cos.(t .+ ψ)
    y = y0 .+ b .* sin.(t .+ ψ)
    return Shape(x, y)
end

function plot_car_trans!(fig_env, car_trans, car_cons)
    for i in eachindex(car_cons[:,1])
        plot!(fig_env, rectangle(car_cons[i,:]...), fillcolor=:gray, label="", linewidth=1)
    end
    # for i in eachindex(car_trans)
    #     plot!(fig_env, ellipse(car_trans[i].x, car_trans[i].y, car_trans[i].ψ, ae, be), color=:black, fillcolor=:gray, fillopacity=0.3, label="")
    # end
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

κ_max = 2.0
δ_min = -atan(L*κ_max)
δ_max = atan(L*κ_max)
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


ωx = 0.0
ωy = 0.0
ωv = 0.0
ωψ = 0.0
ωa = 1.0
ωδ = 1.0

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

plot_car_trans!(fig_env, car_line, cons_line)


##
limits_list_new = [limits_list[1], limits_list[1], limits_list[1], limits_list[1], limits_list[1]]
limits_list = limits_list_new

obs_ymax = corridors[1,4]
obs_ymin = corridors[1,3]
obs_xmax = corridors[1,2]
obs_xmin = corridors[1,1]

obs_A = [1 0;
         0 1;
         -1 0;
         0 -1] 
obs_B1 = hcat([cons_line[1,1], obs_ymax, -obs_xmin, -obs_ymin])
obs_B2 = hcat([cons_line[2,1], obs_ymax, -obs_xmin, -(cons_line[1,4]+0.5)])
obs_B3 = hcat([cons_line[2,1], obs_ymax, -cons_line[1,2], -obs_ymin])
obs_B4 = hcat([obs_xmax, cons_line[2,3]-0.5, -cons_line[1,2], -obs_ymin])
obs_B5 = hcat([obs_xmax, obs_ymax, -cons_line[2,2], -obs_ymin])
obs_C = obs_list[1].C 
obs_D = obs_list[1].D

obs1 = Cons_Corrs(obs_A, obs_B1, obs_C, obs_D)
obs2 = Cons_Corrs(obs_A, obs_B2, obs_C, obs_D)
obs3 = Cons_Corrs(obs_A, obs_B3, obs_C, obs_D)
obs4 = Cons_Corrs(obs_A, obs_B4, obs_C, obs_D)
obs5 = Cons_Corrs(obs_A, obs_B5, obs_C, obs_D)

obs_list = [obs1, obs2, obs3, obs4, obs5]