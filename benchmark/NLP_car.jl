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
Curve driving, the environment is same as that of Car experiments 
=#
# env and car settings
include("utils/env_car2.jl")

# number settings
t0 = 0.0
tf = T_total
N = 600
Δt = tf / N
ω = 0.02
ωx = ω
ωy = ω
#= 
A* to get the initial guess of trajectory
=#
struct Traj_car
    t::Array
    x::Array
    y::Array
    ψ::Array
    v::Array
    δ::Array
    a::Array
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

obs = dilate_obs(obstacles, dilate)
map_env = Map(border, obstacles, obs, grid)
task = Task(cons_init)

function Astar_Car(map_env, task)
    border = map_env.border
    obs = map_env.obs
    grid = map_env.grid
    maze = gen_maze(border, obs, grid)
    @show start0 = CartesianIndex(real2maze(border, start[1], start[2], grid)...)
    @show goal0 = CartesianIndex(real2maze(border, goal[1], goal[2], grid)...)
    res0 = solvemaze(maze, start0, goal0)
    path0 = get_path(res0)
    return path0
end

path = Astar_Car(map_env, task)

#
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

function path2traj(path::Matrix, vmax::Float64)
    t_idx = zeros(size(path,2))
    for i in axes(path,2)[2:end]
        delta_t = norm(path[:,i]-path[:,i-1]) / vmax
        t_idx[i] = t_idx[i-1] + delta_t
    end
    traj0 = zeros(3, size(path,2))
    traj0[1,:] = t_idx
    traj0[2:3,:] = path
    interp_x0 = LinearInterpolation(traj0[1,:], traj0[2,:])
    interp_y0 = LinearInterpolation(traj0[1,:], traj0[3,:])
    ts = collect(LinRange(0.0,traj0[1,end],N))
    x0 = interp_x0.(ts)
    y0 = interp_y0.(ts)
    ψ0 = vcat(atan(y0[2]-y0[1],x0[2]-x0[1]), atan.(diff(y0), diff(x0)))
    v = vcat(0, sqrt.(diff(x0).^2+diff(y0).^2) ./ diff(ts))
    δ = zeros(length(ts))
    a = zeros(length(ts))
    traj = Traj_car(ts, x0, y0, ψ0, v, δ, a)
    return traj
end

traj = path2traj(path, 2.0)
x_car = [traj.x[i] for i in eachindex(traj.t)]
y_car = [traj.y[i] for i in eachindex(traj.t)]
fig_astar = plot(fig_env, x_car, y_car, label="astar")

# NLP
function NLP(traj0)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 10000)
    set_optimizer_attribute(model, "print_level", 5)
    @variables(model, begin
        x[1:N]
        y[1:N]
        ψ[1:N]
        v[1:N]
        δ[1:N]
        a[1:N]
    end)
    for k in 1:N
        set_start_value(x[k], traj0.x[k])
        set_start_value(y[k], traj0.y[k])
        set_start_value(ψ[k], traj0.ψ[k])
        set_start_value(v[k], traj0.v[k])
        set_start_value(δ[k], traj0.δ[k])
        set_start_value(a[k], traj0.a[k])
    end
    gx = goal[1]
    gy = goal[2]
    gψ = goal[3]
    gv = goal[4]
    obj = @expression(model,
                sum([ωx*(x[k]-gx)^2 for k in 1:N]) + 
                sum([ωy*(y[k]-gy)^2 for k in 1:N]) +  
                sum([ωv*(v[k]-gv)^2 for k in 1:N]) + 
                sum([ωψ*(ψ[k]-gψ)^2 for k in 1:N]) + 
                sum([ωa*(a[k])^2 for k in 1:N]) + 
                sum([ωδ*(δ[k])^2 for k in 1:N])
    )
    @objective(model, Min, obj)
    @constraints(model, begin
        [k=1:N-1], x[k+1] == x[k] + v[k] * cos(ψ[k]) * Δt
        [k=1:N-1], y[k+1] == y[k] + v[k] * sin(ψ[k]) * Δt
        [k=1:N-1], ψ[k+1] == ψ[k] + v[k] * tan(δ[k]) / L * Δt
        [k=1:N-1], v[k+1] == v[k] + a[k] * Δt
    end)
    @constraints(model, begin
        x[1] == start[1]
        y[1] == start[2]
        ψ[1] == start[3]
        v[1] == start[4]
        x[end] == goal[1] 
        y[end] == goal[2]
        ψ[end] == goal[3]
        v[end] == goal[4]
    end)
    @constraints(model, begin
        [k=1:N], δ_min <= δ[k] <= δ_max
        [k=1:N], a_min <= a[k] <= a_max
        [k=1:N], v_min <= v[k] <= v_max
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
            @constraint(model, (x[k].-obs_bnd_x).^2 + (y[k].-obs_bnd_y).^2 .>= obs_bnd_r.^2)
        end
    end
    @show car_x = [car_line[i].x for i in 1:2]
    @show car_y = [car_line[i].y for i in 1:2]
    @show car_r = [1.0  for i in 1:2]
    for k in 1:N
        @constraint(model, (x[k].-car_x).^2  + (y[k].-car_y).^2 .>= car_r.^2)
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
    traj = Traj_car(ts, value.(x), value.(y), value.(ψ), value.(v), value.(δ), value.(a))
    return is_failed, traj, obj_value, sol_time
end

@time _, traj, obj_val, sol_time = NLP(traj)

#1.0
@show diff(traj.t)[1]
@show maximum(traj.v)
@show maximum(traj.a)
@show maximum(traj.δ)
fig_opt = plot(fig_env, traj.x, traj.y, label="car_opt")
@show sol_time

##
using JLD2
using JLD2
using CSV, DataFrames
@save "benchmark/data/NLP2016_car_traj.jld2" traj

method_name = "NLP16"

df = CSV.read("benchmark/data/sol_time_car.csv", DataFrame)

row_index = findfirst(row -> row.Method == method_name, eachrow(df))
if row_index !== nothing
    # 如果存在，覆盖该行
    df[row_index, :SolTime] = sol_time
else
    # 如果不存在，添加新行
    new_row = DataFrame(Method = [method_name], SolTime = [sol_time])
    append!(df, new_row)
end

CSV.write("benchmark/data/sol_time_car.csv", df)