using JLD2
includet("../../common/common.jl")
@load "data/car_cons_limits2.jld2" cons_limits
@load "data/car_cons_obs2.jld2" cons_obs
include("../../visual/visual_car.jl")
car_body = Body()
# car env
route1 = [-2 -1;
          -2 3;
          2 3;
          2 -2.8;
          -0 -2.8] * 1.0
route2 = [-2 -3;
          -2 0;
          2 0;
          2 3] * 1.0
route3 = [2.5 -3;
          2.5 -1;
          1 -1;
          1 1
          -2.5 1
          -2.5 3] * 1.0
route4 = [2.0 -3;
          2.0 0;
          -2.0 0;
          -2.0 3;
          2.0 3] * 1.0
route5 = [-0.8 -3;
          -0.8 0;
          -2.0 0;
          -2.0 3;
          2.0 3;
          2.0 -3] * 1.0
route6 = [-3.0 -2.0;
          -1.3 -2.0;
          -1.3 2.0;
          1.5 2.0;
          1.5 0.0;
          3.0 0.0] * 1.0
route7 = [-2.0 -2.0;
          -2.0 2.0;
          +2.0 2.0] * 1.0
route8 = [0.8 -3.0;
          0.8 0.0;
          -0.8 0.0;
          -0.8 3.0] * 1.0
route9 = [-4.5 -2.0;
          -1.5 -2.0;
          -1.5 2.0;
          1.3 2.0;
          1.3 0.0;
          4.5 0.0] * 1.0
route10 = [-3 -2;
          0 -2;
          0 2;
          3 2;] * 1.0
# route10 = [-4 -2;
#         0 -2.5;
#         0 2.5;
#         4 2;] * 1.0
Time1 = [4.0, 4.0, 5.8, 2.0] * 0.8
Time2 = [3.0, 4.0, 3.0] * 0.8
Time3 = [2.0, 1.5, 2.0, 3.5, 2.0] * 0.8
Time4 = [3.0, 4.0, 3.0, 4.0] * 0.8
Time5 = [3.0, 1.2, 3.0, 4.0, 6.0] * 0.8
Time6 = [1.7, 4.0, 2.8, 2.0, 1.5] * 0.8
Time7 = [4.0, 4.0] * 0.8
Time8 = [3.0, 1.6, 3.0] * 0.8
Time9 = [3.0, 4.0, 2.8, 2.0, 3.2] * 0.8
Time10 = [3.0, 4.0, 3.0] * 0.8

Route = [route1, route2, route3, route4, route5, route6, route7, route8, route9, route10]
Time = [Time1, Time2, Time3, Time4, Time5, Time6, Time7, Time8, Time9, Time10]

# car benchamrk route: 10
num = 9
route = Route[num]
T0 = Time[num]

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
# start += [0.0, -0.5, 0.0, 0.0]
# goal += [0.0, +0.5, 0.0, 0.0]
L = 0.353 # wheelbase
W = 0.264 # width
Lt = L * 1.4
Wt = W * 1.4
ae = 0.5
be = 0.5
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

car1_line = CarPos(2.0, -0.5, 0.0, 0.0)
car2_line = CarPos(6.0, 0.5, 0.0, 0.0)
car3_line = CarPos(10.0, -0.5, 0.0, 0.0)
car_line = [car1_line, car2_line, car3_line]
Lcar = Lt/2
Wcar = Wt/2
cons_line = [
    car1_line.x-Lcar car1_line.x+Lcar car1_line.y-Wcar car1_line.y+Wcar;
    car2_line.x-Lcar car2_line.x+Lcar car2_line.y-Wcar car2_line.y+Wcar;
    car3_line.x-Lcar car3_line.x+Lcar car3_line.y-Wcar car3_line.y+Wcar;
]
A = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

car1_raw = CarPos(-2.0, -2.5, 0.0, 0.0)
car2_raw = CarPos(-0.5, 0.0, π/2, 0.0)
car3_raw = CarPos(2.0, 1.5, 0.0, 0.0)
car_raw = [car1_raw, car2_raw, car3_raw]
cons_raw = [
    car1_raw.x-Lcar car1_raw.x+Lcar car1_raw.y-Wcar car1_raw.y+Wcar;
    car2_raw.x-Wcar car2_raw.x+Wcar car2_raw.y-Lcar car2_raw.y+Lcar;
    car3_raw.x-Lcar car3_raw.x+Lcar car3_raw.y-Wcar car3_raw.y+Wcar;
]
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