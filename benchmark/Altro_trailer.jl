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
using Test
using Interpolations
include("utils/astar.jl")
env_name = ""
file_input = "benchmark/data/NLP2016_trailer_traj"*env_name*".jld2"
@load file_input traj
# @load "benchmark/data/Li2022_trailer_traj.jld2" traj
traj0 = traj

#= 
Curve driving, the environment is same as that of Tractor-Trailer experiments
=#
# env and trailer settings
include("utils/env_trailer.jl")

# number settings
t0 = 0.0
tf = sum(T0)
dt = 0.05
N = Int(floor(tf/dt))

ω = 0.02
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
struct Traj_trailer_altro
    t::Array 
    x0::Array
    y0::Array
    ψ0::Array
    ψ1::Array
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
    ts = collect(LinRange(0.0,min(traj1[1,end],traj0[1,end]),N))
    x0 = interp_x0.(ts)
    y0 = interp_y0.(ts)
    x1 = interp_x1.(ts)
    y1 = interp_y1.(ts)
    ψ0 = vcat(atan(y0[2]-y0[1],x0[2]-x0[1]), atan.(diff(y0), diff(x0)))
    ψ1 = vcat(atan(y1[2]-y1[1],x1[2]-x1[1]), atan.(diff(y1), diff(x1)))
    v = vcat(0, sqrt.(diff(x0).^2+diff(y0).^2) ./ diff(ts))
    δ = zeros(length(ts))
    a = zeros(length(ts))
    traj = Traj_trailer_altro(ts, x0, y0, ψ0, ψ1, v, δ, a)
    return traj
end

traj = path2traj(path, 2.0)
astar_x0 = [traj.x0[i] for i in eachindex(traj.t)]
astar_y0 = [traj.y0[i] for i in eachindex(traj.t)]
fig_astar = plot(fig_env, astar_x0, astar_y0, label="astar0")

# function
RobotDynamics.@autodiff struct Trailer{T} <: RobotDynamics.ContinuousDynamics
    d0::T
    d1::T
    w::T
end

RobotDynamics.state_dim(::Trailer) = 5
RobotDynamics.control_dim(::Trailer) = 2

function RobotDynamics.dynamics(trailer::Trailer, s, u)
    d0 = trailer.d0
    d1 = trailer.d1
    x0 = s[1]
    y0 = s[2]
    ψ0 = s[3]
    ψ1 = s[4]
    v = s[5]
    δ = u[1]
    a = u[2]

    dx0 = v * cos(ψ0)
    dy0 = v * sin(ψ0)
    dψ0 = v * tan(δ) / d0
    dψ1 = v * sin(ψ0 - ψ1) / d1
    dv = a
    return [dx0, dy0, dψ0, dψ1, dv]
end

function RobotDynamics.dynamics!(trailer::Trailer, sdot, s, u)
    sdot .= dynamics(trailer, s, u) 
end

model = Trailer{Float64}(d0, d1, W)
n,m = size(model)
s0 = SVector(start...)
sf = SVector(goal...)

Q = Diagonal(@SVector fill(ω, n))
R = Diagonal(@SVector fill(1, m))
Qf = Diagonal(@SVector fill(0, n))
obj = LQRObjective(Q, R, Qf, sf, N)

conSet = ConstraintList(n,m,N)
# control bounds
ctrl_bnd = BoundConstraint(n,m, u_min=[δ_min, a_min], u_max=[δ_max, a_max], x_min=[-Inf,-Inf,-Inf,-Inf,v_min], x_max=[Inf,Inf,Inf,Inf,v_max])
add_constraint!(conSet, ctrl_bnd, 1:N)

# goal constraint 
goal_con = GoalConstraint(sf)
add_constraint!(conSet, goal_con, N)

# collision avoidance: 
# fig_env = plot(aspect_ratio=:equal)

N1,N2,_ = size(obstacles)
for i in 1:N1, j in 1:N2
    x_min, x_max = obstacles[i,j,1], obstacles[i,j,2]
    y_min, y_max = obstacles[i,j,3], obstacles[i,j,4]
    r = 0.65
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

#
traj_guess = traj0
# set problem, rollout and solve
X0 = [@SVector [traj_guess.x0[i], traj_guess.y0[i], traj_guess.ψ0[i], traj_guess.ψ1[i], traj_guess.v0[i]] for i in 1:N]
U0 = [@SVector [traj_guess.δ0[i], traj_guess.a0[i]] for i in 1:N]
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
opts = SolverOptions(
    cost_tolerance=1e-6,
    # cost_tolerance_intermediate=1e-5,
    # constraint_tolerance = 1e-6,
    penalty_scaling=10., 
    penalty_initial=1000.,
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
Xmat = hcat(Vector.(X)...);
Umat = hcat(Vector.(U)...);
x0 = Xmat[1,:]
y0 = Xmat[2,:]
x1 = x0 - d1 * cos.(Xmat[4,:])
y1 = y0 - d1 * sin.(Xmat[4,:])
fig_traj = plot(fig_env, x0, y0, aspect_ratio=:equal, label="altro_tractor")
fig_traj = plot(fig_traj, x1, y1, aspect_ratio=:equal, label="altro_trailer")
# ylims!(fig_traj, -1,+1)
guess_x0 = [traj_guess.x0[i] for i in eachindex(traj.t)]
guess_y0 = [traj_guess.y0[i] for i in eachindex(traj.t)]
guess_x1 = [traj_guess.x1[i] for i in eachindex(traj.t)]
guess_y1 = [traj_guess.y1[i] for i in eachindex(traj.t)]
# fig_traj_guess = plot!(fig_traj, guess_x0, guess_y0, label="initial_tractor")
# fig_traj_guess = plot!(fig_traj, guess_x1, guess_y1, label="initial_trailer")

@show sol_time
@show fig_traj

###
# Xroll = [copy(s0) for k = 1:N]
# for i in 1:N-1
#     Xroll[i+1][1] =  Xroll[i+1][1] + Xroll[i+1][5] * cos(Xroll[i+1][3]) * dt
#     Xroll[i+1][2] =  Xroll[i+1][2] + Xroll[i+1][5] * sin(Xroll[i+1][3]) * dt
#     Xroll[i+1][3] =  Xroll[i+1][3] + Xroll[i+1][5] * tan(U0[i][1]) / d0 * dt
# end

##
ts = collect(1:N)*dt
traj = Traj_trailer(ts, Xmat[1,:], Xmat[2,:], Xmat[3,:], x1, y1, Xmat[4,:], Xmat[5,:], Umat[2,:], Umat[1,:], Xmat[5,:])

using JLD2
using CSV, DataFrames
filename = "benchmark/data/Altro2019_trailer_traj.jld2"
@save filename traj

# method_name = "Altro"*env_name
# sol_time = round(sol_time, digits=3)

# df = CSV.read("benchmark/data/sol_time_trailer.csv", DataFrame)

# row_index = findfirst(row -> row.Method == method_name, eachrow(df))
# if row_index !== nothing
#     # 如果存在，覆盖该行
#     df[row_index, :SolTime] = sol_time
# else
#     # 如果不存在，添加新行
#     new_row = DataFrame(Method = [method_name], SolTime = [sol_time])
#     append!(df, new_row)
# end

# CSV.write("benchmark/data/sol_time_trailer.csv", df)
