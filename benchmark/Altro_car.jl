#=
IROS, 2019, Taylor A. Howell
ALTRO: A Fast Solver for Constrained Trajectory Optimization
=#
using Revise
using Altro, TrajectoryOptimization, RobotDynamics # 0.4.7 is ok, but 0.4.8 is not
using StaticArrays, LinearAlgebra
using MathOptInterface, Ipopt
const TO = TrajectoryOptimization
const RD = RobotDynamics
using ForwardDiff, FiniteDiff
using Plots, JLD2, CSV, DataFrames
using Interpolations
using Test
include("utils/astar.jl")

@load "benchmark/data/NLP2016_car_traj.jld2" traj
traj0 = traj

#=
Urban driving
=#
# env and car settings
include("utils/env_car2.jl")

# number settings
t0 = 0.0
tf = T_total
N = 600
dt = tf / N
#= 
A* to get the initial guess of trajectory
=#
struct Traj_car_altro
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
    traj = Traj_car_altro(ts, x0, y0, ψ0, v, δ, a)
    return traj
end

traj = path2traj(path, 1.0)

#
# function
RD.@autodiff struct Car{T} <: RD.ContinuousDynamics
    L::T
    W::T
end

RD.state_dim(::Car) = 4
RD.control_dim(::Car) = 2

function RD.dynamics(car::Car, s, u)
    L = car.L
    x = s[1]
    y = s[2]
    ψ = s[3]
    v = s[4]
    δ = u[1]
    a = u[2]

    dx = v * cos(ψ)
    dy = v * sin(ψ)
    dψ = v * tan(δ) / L
    dv = a
    return [dx, dy, dψ, dv]
end

function RD.dynamics!(car::Car, sdot, s, u)
    sdot .= dynamics(car, s, u) 
end

model = Car{Float64}(L, W)
n,m = size(model)
s0 = SVector(start...)
sf = SVector(goal...)
# s0 = SVector(start[1:3]..., NaN)
# sf = SVector(goal[1:3]..., NaN)

Q = Diagonal(@SVector fill(0.03,n))
R = Diagonal(@SVector fill(1,m)) 
Qf = Diagonal(@SVector fill(0,n))  
obj = LQRObjective(Q, R, Qf, sf, N)

conSet = ConstraintList(n,m,N)
# control bounds
bnd = BoundConstraint(n,m, u_min=[δ_min, a_min], u_max=[δ_max, a_max], x_min=[-Inf,-Inf,-Inf,v_min], x_max=[Inf,Inf,Inf,v_max])
add_constraint!(conSet, bnd, 1:N)

# goal constraint 
goal_con = GoalConstraint(sf)
add_constraint!(conSet, goal_con, N)

# collision avoidance with drivable surface boundaries
# fig_env = plot(aspect_ratio=:equal)

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
    obs_circles = CircleConstraint(n, SVector(obs_bnd_x...), SVector(obs_bnd_y...), SVector(obs_bnd_r...))
    add_constraint!(conSet, obs_circles, 1:N)
    # plot_bnd_cir(fig_env, obs_bnd_x, obs_bnd_y, obs_bnd_r)
end

car_x = [car_line[i].x for i in 1:2]
car_y = [car_line[i].y for i in 1:2]
car_r = [0.75 for i in 1:2]
car_circles = CircleConstraint(n, SVector(car_x...), SVector(car_y...), SVector(car_r...))
add_constraint!(conSet, car_circles, 1:N)

# set problem, rollout and solve
traj_guess = traj0
X0 = [@SVector [traj_guess.x[i], traj_guess.y[i], traj_guess.ψ[i], traj_guess.v[i]] for i in 1:N]
U0 = [@SVector [traj_guess.δ[i], traj_guess.a[i]] for i in 1:N]
X0mat = hcat(Vector.(X0)...);
U0mat = hcat(Vector.(U0)...);

prob = Problem(model, obj, s0, tf, xf=sf, constraints=conSet);

initial_states!(prob, X0mat)
initial_controls!(prob, U0mat) 

# Simulate the system forward
rollout!(prob)

# Extract states, controls, and times
X = states(prob)
Xmat = hcat(Vector.(X)...)
U = controls(prob)
t = gettimes(prob)

Xrollout = [copy(s0) for k = 1:N]
for k = 1:N-1
    Xrollout[k+1] = RD.discrete_dynamics(
        get_model(prob, k), Xrollout[k], U0[k], dt*(k-1), dt
    )
end
# @test Xrollout ≈ X
# @test X0mat ≈ Xmat
#
#
opts = SolverOptions(
    cost_tolerance=1e-6,
    # cost_tolerance_intermediate=1e-5,
    # constraint_tolerance = 1e-6,
    penalty_scaling=10.,
    penalty_initial=10000.,
    iterations=10000
)

method = "altro"
# method = "altro"
if method == "sqp" 
    altro = Altro.iLQRSolver(prob, opts)
    Altro.solve!(altro);
elseif method == "altro"
    altro = ALTROSolver(prob, opts);
    set_options!(altro, show_summary=true);
    time0 = time()
    Altro.solve!(altro);
    time1 = time()
    sol_time = time1 - time0
elseif method == "ipopt"
    # Copy problem to avoid modifying the original problem
    prob_nlp = copy(prob)

    # Add the dynamics and initial conditions as explicit constraints
    TrajectoryOptimization.add_dynamics_constraints!(prob_nlp)

    # Reset our initial guess
    initial_controls!(prob_nlp, U0)
    rollout!(prob_nlp)

    # Create the NLP
    nlp = TrajOptNLP(prob_nlp, remove_bounds=true, jac_type=:vector);
    optimizer = Ipopt.Optimizer()
    TrajectoryOptimization.build_MOI!(nlp, optimizer)
    MOI.optimize!(optimizer)
    MOI.get(optimizer, MOI.TerminationStatus())
end

#
X = states(altro);
U = controls(altro);
Xmat = hcat(Vector.(X)...)
Umat = hcat(Vector.(U)...)
fig_traj = plot(fig_env, Xmat[1,:], Xmat[2,:], label="altro", aspect_ratio=:equal)
guess_x = [traj_guess.x[i] for i in eachindex(traj_guess.t)]
guess_y = [traj_guess.y[i] for i in eachindex(traj_guess.t)]
fig_traj_guess = plot!(fig_traj, guess_x, guess_y, label="initial")

@show sol_time

###
# fig_v = plot(traj.t, traj.v, label="v", legend=:outertopright)
# fig_a = plot(traj.t, traj.a, label="a", legend=:outertopright)
# fig_δ = plot(traj.t, traj.δ, label="δ", legend=:outertopright)
# plot(fig_v, fig_a, fig_δ, layout=(3,1), size=(500,500))

## 
ts = collect(1:N)*dt
traj = Traj_car(ts, Xmat[1,:], Xmat[2,:], Xmat[3,:], Xmat[4,:], Umat[1,:], Umat[2,:])

using JLD2
using CSV, DataFrames
@save "benchmark/data/Altro2019_car_traj.jld2" traj

method_name = "Altro19"
sol_time = round(sol_time, digits=3)

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