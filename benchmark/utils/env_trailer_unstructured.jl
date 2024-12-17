using JLD2
includet("../../common/common.jl")
@load "data/cons_limits2.jld2" cons_limits
@load "data/cons_obs2.jld2" cons_obs
includet("../../visual/visual_trailer.jl")
trailer_body = TrailerBody()

# parameters
d0 = trailer_body.trailer_length
d1 = trailer_body.link_length
W = trailer_body.trailer_width
L = max(trailer_body.trailer_length, trailer_body.trailer2_length)


# trailer env
route1 = [0. 0.;
        3. 3.;
        6. 3.;
        9. 6.] * 1.0
Time1 = [3., 3., 3.] * 1.0

route2 = [0. 0.;
        3. 3.;
        6. 3.;
        9. 0.;
        6. -3.;
        3. -3.;] * 1.0
Time2 = [3., 3., 3., 3., 3., 3.] * 1.0

# route kargo
route3 = [-23341.2 7173.53;
                                -23323.6 7167.39;
                                -23306.7 7156.59;
                                -23298.5 7140.4;
                                -23279.9 7134.07;
                                -23263.4 7142.07;
                                -23256.2 7160.89;
                                -23256.6 7179.06;
                                -23285.4 7171.03;
                                -23307.6 7183.35;
                                -23335.5 7193.31] / 5.0 .- 10000.0/ 5.0

Time3 = [3., 3., 3., 3., 3., 3., 3., 3., 3., 3., 3.] * 2.0

route4 = [-23341.2 7173.53;
                                -23323.6 7167.39;
                                -23306.7 7156.59;
                                -23298.5 7140.4;
                                -23279.9 7134.07;
                                -23263.4 7142.07;] / 10.0
Time4 = ones(5) * 3.0

route5 = [-23341.7 7173.67;
                -23323.8 7167.59;
                -23309.1 7153.37;
                -23291.9 7135.95;
                -23264.7 7141.49;
                -23261.3 7169.12;
                -23289.4 7171.41;
                -23308.4 7187.1;
                -23335.7 7196.82] / 10.0
Time5 = [3., 3., 3., 3., 3., 3., 3., 3.] * 1.0

route6 = [-23345.1 7174.65;
                -23324.0 7167.37;
                -23306.6 7150.86;
                -23293.8 7129.95;
                -23267.5 7137.99;
                -23281.1 7161.54;
                -23290.1 7180.41;
                -23267.2 7188.43;
                -23260.4 7209.47] / 10.0
Time6 = [3., 3., 3., 3., 3., 3., 3., 3.] * 1.0

route7 = [-23344.9 7174.6;
                -23323.9 7167.34;
                -23308.0 7154.2;
                -23299.6 7135.13;
                -23277.5 7132.76;
                -23262.5 7145.93;
                -23276.9 7161.57;
                -23284.9 7180.33;
                -23266.7 7190.75;
                -23259.9 7210.05] / 10.0
Time7 = [3., 3., 3., 3., 3., 3., 3., 3., 3.] * 1.0

route8 = [-23333.9 7171.12
                -23311.0 7162.49;
                -23293.0 7136.12;
                -23263.1 7153.36;
                -23283.6 7174.82;
                -23263.3 7190.66;
                -23253.1 7219.07] / 10.0
Time8 = [3., 3., 3., 3., 3., 3.] * 1.0

route9 = [-23323.9 7167.19;
                -23305.9 7160.51;
                -23285.8 7170.97;
                -23284.5 7138.44;
                -23257.9 7143.81;
                -23231.5 7134.01] / 10.0
Time9 = [3., 3., 3., 3., 4.] * 0.8

Route = [route1, route2, route3, route4, route5, route6, route7, route8, route9]
Time = [Time1, Time2, Time3, Time4, Time5, Time6, Time7, Time8, Time9]

# trailer benchamrk route: 9
num = 9
route = Route[num]
T0 = Time[num]
env_name = "E$num"
N = length(route[:,1]) - 1

seg_N = length(route[:,1]) - 1
sdp_N = seg_N
sdp_s = 3
corridors = gen_unstructured_env(route; type=:simple)
cons_init = gen_unstructured_init(route)
fig_env = plot_unstructured_env(corridors)
limits_list, obs_list = gen_unstructured_cons_trailer(corridors, cons_limits)

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
# weight
ωx0 = 0.0
ωy0 = 0.0
ωψ0 = 0.0
ωψ1 = 0.0
ωv0 = 0.0
ωa0 = 1.0
ωδ0 = 1.0

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
Δs = 0.3
Δa = 0.3

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
