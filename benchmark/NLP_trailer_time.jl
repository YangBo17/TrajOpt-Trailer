#=
NLP: A* + NLP, use IPOPT as the NLP solver, SNOPT is commertial solver, IPOPT is open-source. JuMP.jl currently do not support SNOPT.
=#
using Revise
using JuMP, Ipopt
using LinearAlgebra
using Plots, JLD2, CSV, DataFrames
using Interpolations
include("utils/astar.jl")

#=
Curve driving, the environment is same as that of Tractor-Trailer experiments
=#
# env and trailer settings
include("utils/env_trailer.jl")

# number settings
wt = 3.

t0 = 0.0
tf = 10.0
N = 300 
Δt = tf / N

ω = 0.008
ωx0 = ω
ωy0 = ω
ωψ0 = ω
ωψ1 = ω
ωv0 = ω
ωa0 = 1.0
ωδ0 = 1.0

#=
A* to get the initial guess of trajectory
=#
struct Path
    path0::Matrix
    path1::Matrix
end

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

struct Map
    border::Vector
    obstacles::Array
    obs::Array
    grid::Float64
end

struct Task
    init::Cons_Init
end

struct Corr
    corr0::Matrix
    corr1::Matrix
end

obs = dilate_obs(obstacles, dilate)
map_env = Map(border, obstacles, obs, grid)
task = Task(cons_init)

function Astar_Trailer(map_env, task)
    border = map_env.border
    obs = map_env.obs
    grid = map_env.grid
    maze = gen_maze(border, obs, grid)
    cons_init = task.init
    start1 = CartesianIndex(real2maze(border, cons_init.pos[1,:]..., grid)...)
    goal1 = CartesianIndex(real2maze(border, cons_init.pos[2,:]..., grid)...)
    start0 = CartesianIndex(real2maze(border, start[1], start[2], grid)...)
    goal0 = CartesianIndex(real2maze(border, goal[1], goal[2], grid)...)
    res0 = solvemaze(maze, start0, goal0)
    res1 = solvemaze(maze, start1, goal1)
    path1 = get_path(res1)
    path0 = get_path(res0)
    path = Path(path0, path1)
    return path
end

path = Astar_Trailer(map_env, task)

function path2traj(path::Matrix, vmax::Float64)
    t_idx = zeros(size(path,2))
    for i in axes(path,2)[2:end]
        delta_t = norm(path[:,i]-path[:,i-1]) / vmax
        t_idx[i] = t_idx[i-1] + delta_t
    end
    traj = zeros(3, size(path,2))
    traj[1,:] = t_idx
    traj[2:3,:] = path
    return traj
end

function path2traj(path::Path, vmax::Float64)
    traj0 = path2traj(path.path0, vmax)
    traj1 = path2traj(path.path1, vmax)
    interp_x0 = LinearInterpolation(traj0[1,:], traj0[2,:])
    interp_y0 = LinearInterpolation(traj0[1,:], traj0[3,:])
    interp_x1 = LinearInterpolation(traj1[1,:], traj1[2,:])
    interp_y1 = LinearInterpolation(traj1[1,:], traj1[3,:])
    ts = collect(LinRange(0.0, min(traj1[1,end],traj0[1,end]), N))
    x0 = interp_x0.(ts)
    y0 = interp_y0.(ts)
    x1 = interp_x1.(ts)
    y1 = interp_y1.(ts)
    ψ0 = vcat(atan(y0[2]-y0[1],x0[2]-x0[1]), atan.(diff(y0), diff(x0)))
    ψ1 = vcat(atan(y1[2]-y1[1],x1[2]-x1[1]), atan.(diff(y1), diff(x1)))
    v0 = vcat(0, sqrt.(diff(x0).^2+diff(y0).^2) ./ diff(ts)) 
    v1 = vcat(0, sqrt.(diff(x1).^2+diff(y1).^2) ./ diff(ts))
    a0 = zeros(length(ts))
    δ0 = zeros(length(ts))
    return Traj_trailer(ts, x0, y0, ψ0, x1, y1, ψ1, v0, a0, δ0, v1)
end

traj = path2traj(path, 2.0)
astar_x0 = [traj.x0[i] for i in eachindex(traj.t)]
astar_y0 = [traj.y0[i] for i in eachindex(traj.t)]
astar_x1 = [traj.x1[i] for i in eachindex(traj.t)]
astar_y1 = [traj.y1[i] for i in eachindex(traj.t)]
fig_astar = plot(fig_env, astar_x0, astar_y0, label="astar0")
plot!(fig_astar, astar_x1, astar_y1, label="astar1")

# NLP
function NLP(traj0)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 2000)
    # set_optimizer_attribute(model, "print_level", 0)
    @variables(model, begin
        Δt >= 0.01
        x0[1:N]
        y0[1:N]
        ψ0[1:N]
        x1[1:N]
        y1[1:N]
        ψ1[1:N]
        v0[1:N]
        a0[1:N]
        δ0[1:N]
        v1[1:N]
    end)
    set_start_value.(x0, traj0.x0)
    set_start_value.(y0, traj0.y0)
    set_start_value.(ψ0, traj0.ψ0)
    set_start_value.(x1, traj0.x1)
    set_start_value.(y1, traj0.y1)
    set_start_value.(ψ1, traj0.ψ1)
    set_start_value.(v0, traj0.v0)
    set_start_value.(a0, traj0.a0)
    set_start_value.(δ0, traj0.δ0)
    set_start_value.(v1, traj0.v1)
    gx0 = goal[1]
    gy0 = goal[2]
    gψ0 = goal[3]
    gψ1 = goal[4]
    gv0 = goal[5]
    obj = @expression(model,
                sum([ωx0*(x0[k]-gx0)^2 for k in 1:N]) + 
                sum([ωy0*(y0[k]-gy0)^2 for k in 1:N]) + 
                sum([ωψ0*(ψ0[k]-gψ0)^2 for k in 1:N]) + 
                sum([ωψ1*(ψ1[k]-gψ1)^2 for k in 1:N]) + 
                sum([ωv0*(v0[k]-gv0)^2 for k in 1:N]) + 
                sum([ωa0*(a0[k])^2 for k in 1:N]) + 
                sum([ωδ0*(δ0[k])^2 for k in 1:N])
    )
    @objective(model, Min, obj + wt * Δt*N)
    @constraints(model, begin
        x0[1] == start[1]
        y0[1] == start[2]
        ψ0[1] == start[3]
        ψ1[1] == start[4]
        v0[1] == start[5]
        x0[end] == goal[1]
        y0[end] == goal[2]
        ψ0[end] == goal[3]
        # ψ1[end] == goal[4]
        v0[end] == goal[5]
    end)
    @constraints(model, begin
        [k=1:N-1], x0[k+1] == x0[k] + v0[k]*cos(ψ0[k])*Δt 
        [k=1:N-1], y0[k+1] == y0[k] + v0[k]*sin(ψ0[k])*Δt
        [k=1:N-1], v0[k+1] == v0[k] + a0[k]*Δt 
        [k=1:N-1], ψ0[k+1] == ψ0[k] + v0[k]*tan(δ0[k])/d0*Δt
    end)
    @constraints(model, begin
        [k=1:N], x1[k] == x0[k] - d1*cos(ψ1[k])
        [k=1:N], y1[k] == y0[k] - d1*sin(ψ1[k])
        [k=1:N], v1[k] == v0[k]*cos(ψ0[k]-ψ1[k])
    end)
    @constraints(model, begin
        [k=1:N-1], ψ1[k+1] == ψ1[k] + v0[k]*sin(ψ0[k]-ψ1[k])/d1*Δt
    end)
    @constraints(model, begin
        [k=1:N], abs(ψ1[k]-ψ0[k]) <= π/2-jackknife_buffer
        [k=1:N], a_min <= a0[k] <= a_max
        [k=1:N], v_min <= v0[k] <= v_max
        [k=1:N], δ_min <= δ0[k] <= δ_max
    end)
    N1,N2,_ = size(obstacles)
    for i in 1:N1, j in 1:N2
        x_min, x_max = obstacles[i,j,1], obstacles[i,j,2]
        y_min, y_max = obstacles[i,j,3], obstacles[i,j,4]
        r = 0.35
        num_a = Int(ceil((x_max - x_min) / r)) 
        obs_bnd1_x = collect(LinRange(x_min, x_max, num_a))
        obs_bnd1_y = ones(size(obs_bnd1_x)) * y_min
        obs_bnd2_x = collect(LinRange(x_min, x_max, num_a))
        obs_bnd2_y = ones(size(obs_bnd2_x)) * y_max

        num_b = Int(ceil((y_max - y_min) / r)) 
        obs_bnd3_y = collect(LinRange(y_min, y_max, num_b))
        obs_bnd3_x = ones(size(obs_bnd3_y)) * x_min
        obs_bnd4_y = collect(LinRange(y_min, y_max, num_b))
        obs_bnd4_x = ones(size(obs_bnd4_y)) * x_max

        obs_bnd_x = vcat(obs_bnd1_x, obs_bnd2_x, obs_bnd3_x, obs_bnd4_x)
        obs_bnd_y = vcat(obs_bnd1_y, obs_bnd2_y, obs_bnd3_y, obs_bnd4_y)
        obs_bnd_r = ones(size(obs_bnd_x)) * r

        for k in 1:N
            @constraint(model, (x0[k].-obs_bnd_x).^2 .+ (y0[k].-obs_bnd_y).^2 .>= obs_bnd_r.^2) 
            @constraint(model, (x1[k].-obs_bnd_x).^2 .+ (y1[k].-obs_bnd_y).^2 .>= obs_bnd_r.^2) 
        end
    end
    JuMP.optimize!(model)
    is_failed = false
    obj_value = objective_value(model)
    term_status = termination_status(model)
    sol_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
    if term_status != MOI.OPTIMAL
        is_failed = true
    end
    ts = collect(1:N)*value(Δt)
    traj = Traj_trailer(ts, value.(x0), value.(y0), value.(ψ0), value.(x1), value.(y1), value.(ψ1), value.(v0), value.(a0), value.(δ0), value.(v1))
    return is_failed, traj, obj_value, sol_time
end

@time is_failed, traj, obj_val, sol_time = NLP(traj)

@show sol_time
@show traj.t[end]
#
@show diff(traj.t)[1]
fig_opt = plot(fig_env, traj.x0, traj.y0, label="tractor")
plot!(fig_opt, traj.x1, traj.y1, label="trailer")  
# ylims!(-5,5)
# xlims!(-5,5)

##
using JLD2
using CSV, DataFrames
@save "benchmark/data/NLP2016T_trailer_traj.jld2" traj

method_name = "NLP16T"

df = CSV.read("benchmark/data/sol_time_trailer.csv", DataFrame)

row_index = findfirst(row -> row.Method == method_name, eachrow(df))
if row_index !== nothing
    # 如果存在，覆盖该行
    df[row_index, :SolTime] = sol_time
else
    # 如果不存在，添加新行
    new_row = DataFrame(Method = [method_name], SolTime = [sol_time])
    append!(df, new_row)
end

CSV.write("benchmark/data/sol_time_trailer.csv", df)
