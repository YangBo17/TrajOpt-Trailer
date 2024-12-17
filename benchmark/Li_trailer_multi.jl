#=
RAL, 2022, Bai Li 
Optimization-Based Maneuver Planning for a Tractor-Trailer Vehicle in a Curvy Tunnel: A Weak Reliance on Sampling and Search
=#
using Revise
using Plots
using AStarSearch
using JuMP, Ipopt
using Interpolations
include("utils/astar.jl")

# number settings
Δt = 0.1
N = Int(floor(sum(T0)/Δt))
w_obj = 0.05
w_penalty = 1000.0

ω = 0.00
ωx0 = ω
ωy0 = ω
ωψ0 = ω
ωψ1 = ω
ωv0 = ω
ωa0 = 1.0
ωδ0 = 1.0 

w = 0.0
wx0 = w
wy0 = w
wψ0 = w
wψ1 = w
wv0 = w
wa0 = 1.0
wδ0 = 1.0 

#=
Stage 1: AStar generate course paths
    - preparation of a grid map_env 
    - generation of coarse paths
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
    return path, maze
end

path, maze = Astar_Trailer(map_env, task)
maze = reverse(maze, dims=1)
fig_maze = heatmap(maze)
fig_env_maze = plot(fig_env, fig_maze, layout=(1,2),size=(800,400))
display(fig_env_maze)

#
#=
Stage 2: LIOS
    - generation of reference trajectory
    - construction of safe travel corridors
=#
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


traj = path2traj(path, v_max)
astar_x0 = [traj.x0[i] for i in eachindex(traj.t)]
astar_y0 = [traj.y0[i] for i in eachindex(traj.t)]
astar_x1 = [traj.x1[i] for i in eachindex(traj.t)]
astar_y1 = [traj.y1[i] for i in eachindex(traj.t)]
fig_astar = plot(fig_env, astar_x0, astar_y0, label="astar0")
plot!(fig_astar, astar_x1, astar_y1, label="astar1")
display(fig_astar)
#
#
function overlap(box1, box2)
    x_overlap = (box1[2]-box1[1]) + (box2[2]-box2[1]) - (maximum([box1[2],box2[2]])-minimum([box1[1],box2[1]])) >= 0
    y_overlap = (box1[4]-box1[3]) + (box2[4]-box2[3]) - (maximum([box1[4],box2[4]])-minimum([box1[3],box2[3]])) >= 0

    if x_overlap && y_overlap
        return true 
    else
        return false
    end
end

function overlaps(border, box, obs)
    if box[1] >= border[1] && box[2] <= border[2] && box[3] >= border[3] && box[4] <= border[4]
        for i in axes(obs,1), j in axes(obs,2)
            result = overlap(box, obs[i,j,:])
            if result == true
                return true
            end
        end
        return false
    else
        return true
    end
end

function gen_corr(map_env, traj)
    border = map_env.border
    obs = map_env.obs
    ds = 0.01
    corr0 = zeros(4, length(traj.t))
    corr1 = zeros(4, length(traj.t))
    corr0[1,:] = traj.x0
    corr0[2,:] = traj.x0
    corr0[3,:] = traj.y0 
    corr0[4,:] = traj.y0
    corr1[1,:] = traj.x1
    corr1[2,:] = traj.x1
    corr1[3,:] = traj.y1 
    corr1[4,:] = traj.y1

    function process!(corr)
        for i in axes(corr,2)
            expanding = [true, true, true, true] 
            while expanding != [false, false, false, false]
                if expanding[1] == true
                    tmp = corr[:,i] + ds*[-1,0,0,0]
                    tmp_bool = overlaps(border, tmp, obs)
                    if tmp_bool == true
                        expanding[1] = false
                    else
                        corr[:,i] = tmp
                    end
                end
                if expanding[2] == true
                    tmp = corr[:,i] + ds*[0,+1,0,0]
                    tmp_bool = overlaps(border, tmp, obs)
                    if tmp_bool == true
                        expanding[2] = false
                    else
                        corr[:,i] = tmp
                    end
                end
                if expanding[3] == true
                    tmp = corr[:,i] + ds*[0,0,-1,0]
                    tmp_bool = overlaps(border, tmp, obs)
                    if tmp_bool == true
                        expanding[3] = false
                    else
                        corr[:,i] = tmp
                    end
                end
                if expanding[4] == true
                    tmp = corr[:,i] + ds*[0,0,0,+1]
                    tmp_bool = overlaps(border, tmp, obs)
                    if tmp_bool == true
                        expanding[4] = false
                    else
                        corr[:,i] = tmp
                    end
                end
            end
        end
    end

    process!(corr0)
    process!(corr1)
    return Corr(corr0, corr1)
end

function solve_ocp(task::Task, Γ::Corr, traj::Traj_trailer)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 1000)
    # set_optimizer_attribute(model, "print_level", 1)
    @variables(model, begin
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
    set_start_value.(x0, traj.x0)
    set_start_value.(y0, traj.y0)
    set_start_value.(ψ0, traj.ψ0)
    set_start_value.(x1, traj.x1)
    set_start_value.(y1, traj.y1)
    set_start_value.(ψ1, traj.ψ1)
    set_start_value.(v0, traj.v0)
    set_start_value.(a0, traj.a0)
    set_start_value.(δ0, traj.δ0)
    set_start_value.(v1, traj.v1)
    penalty1 = @expression(model, 
                sum([x0[k+1] - (x0[k]+v0[k]*cos(ψ0[k])*Δt) for k in 1:N-1].^2) + 
                sum([y0[k+1] - (y0[k]+v0[k]*sin(ψ0[k])*Δt) for k in 1:N-1].^2) +
                sum([v0[k+1] - (v0[k]+a0[k]*Δt) for k in 1:N-1].^2) + 
                sum([ψ0[k+1] - (ψ0[k]+v0[k]*tan(δ0[k])/d1*Δt) for k in 1:N-1].^2)
    )
    penalty2 = @expression(model,
                sum([x1[k] - (x0[k]-d1*cos(ψ1[k])) for k in 1:N].^2) + 
                sum([y1[k] - (y0[k]-d1*sin(ψ1[k])) for k in 1:N].^2)
    )
    penalty3 = @expression(model, 
                sum([ψ1[k+1] - (ψ1[k]+v0[k]*sin(ψ0[k]-ψ1[k])/d1*Δt) for k in 1:N-1].^2)
    )
    penalty4 = @expression(model,
                sum([v1[k] - (v0[k]*cos(ψ0[k]-ψ1[k])) for k in 1:N].^2)
    )
    penalty5 = @expression(model,
                sum([min(0, (x0[k]+Len/2*cos(ψ0[k])) - Γ.corr0[1,k])^2 for k in 1:N]) + 
                sum([max(0, (x0[k]+Len/2*cos(ψ0[k])) - Γ.corr0[2,k])^2 for k in 1:N]) +
                sum([min(0, (y0[k]+Len/2*sin(ψ0[k])) - Γ.corr0[3,k])^2 for k in 1:N]) + 
                sum([max(0, (y0[k]+Len/2*sin(ψ0[k])) - Γ.corr0[4,k])^2 for k in 1:N]) +
                sum([min(0, (x1[k]+Len/2*cos(ψ1[k])) - Γ.corr1[1,k])^2 for k in 1:N]) + 
                sum([max(0, (x1[k]+Len/2*cos(ψ1[k])) - Γ.corr1[2,k])^2 for k in 1:N]) +
                sum([min(0, (y1[k]+Len/2*sin(ψ1[k])) - Γ.corr1[3,k])^2 for k in 1:N]) + 
                sum([max(0, (y1[k]+Len/2*sin(ψ1[k])) - Γ.corr1[4,k])^2 for k in 1:N])
    )
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
    @objective(model, Min, w_obj * obj + w_penalty * (penalty1+penalty2+penalty3+penalty4+penalty5))
    @constraints(model, begin
        x0[1] == start[1]
        y0[1] == start[2]
        ψ0[1] == start[3]
        ψ1[1] == start[4]
        v0[1] == start[5]
        v1[1] == start[5]
        x0[end] == goal[1]
        y0[end] == goal[2]
        ψ0[end] == goal[3]
        ψ1[end] == goal[4]
        v0[end] == goal[5]
        v1[end] == goal[5]
    end)
    @constraints(model, begin
        [k=1:N], abs(ψ1[k]-ψ0[k]) <= π/2-jackknife_buffer
        [k=1:N], a_min <= a0[k] <= a_max
        [k=1:N], v_min <= v0[k] <= v_max
        [k=1:N], δ_min <= δ0[k] <= δ_max
    end)
    JuMP.optimize!(model)
    is_failed = false
    obj_value = objective_value(model)
    term_status = termination_status(model)
    if term_status != MOI.OPTIMAL
        is_failed = true
    end
    ts = collect(1:N)*value(Δt)
    traj = Traj_trailer(ts, value.(x0), value.(y0), value.(ψ0), value.(x1), value.(y1), value.(ψ1), value.(v0), value.(a0), value.(δ0), value.(v1))
    return is_failed, traj, obj_value
end

function measure_infeasibility(traj::Traj_trailer)
    Δt = traj.t[2] - traj.t[1]
    ζ = (
        sum([traj.x0[k+1] - (traj.x0[k]+traj.v0[k]*cos(traj.ψ0[k])*Δt) for k in 1:N-1].^2) + 
        sum([traj.y0[k+1] - (traj.y0[k]+traj.v0[k]*sin(traj.ψ0[k])*Δt) for k in 1:N-1].^2) +
        sum([traj.v0[k+1] - (traj.v0[k]+traj.a0[k]*Δt) for k in 1:N-1].^2) + 
        sum([traj.ψ0[k+1] - (traj.ψ0[k]+traj.v0[k]*tan(traj.δ0[k])/d1*Δt) for k in 1:N-1].^2)
    )
    return ζ
end

function LIOS(map_env::Map, task::Task, path::Path)
    traj = path2traj(path, v_max)
    iter = 0
    ζ = 1e8
    ζ_tol = 1.0
    iter_max = 10
    while ζ >= ζ_tol
        iter += 1
        if iter > iter_max 
            break
        end
        Γ = gen_corr(map_env, traj)
        is_failed, traj, obj_value= solve_ocp(task, Γ, traj)
        ζ = measure_infeasibility(traj)
    end
    return traj
end


#=
Stage 3: TRMO, trust-region-based iterative optimization method
    - formulation of intermediate OCPs, according to trust region
=#
function gen_box(model, x, y, ψ, num)
    if num == 0
        bL = d0
    elseif num == 1
        bL = d1
    end
    xc = @expression(model, x + bL/2 * cos(ψ))
    yc = @expression(model, y + bL/2 * sin(ψ))
    β = atan(bL, Wid)
    α1 = ψ+β-π/2
    α2 = ψ+π/2-β
    α3 = ψ+π/2+β
    α4 = ψ+3*π/2-β
    αs = [α1, α2, α3, α4]
    tL = sqrt(bL^2+Wid^2)/2*2
    tbox = Array{Any}(undef, 2, 4)
    for i in 1:4
        tbox[1, i] = @expression(model, xc + tL * cos(αs[i]))
        tbox[2, i] = @expression(model, yc + tL * sin(αs[i]))
    end
    return tbox
end

function box2tbox(box)
    tbox = zeros(2,4)
    tbox[1,1] = box[2] # x_max
    tbox[2,1] = box[4] # y_max

    tbox[1,2] = box[1] # x_min
    tbox[2,2] = box[4] # y_max

    tbox[1,3] = box[1] # x_min
    tbox[2,3] = box[3] # y_min

    tbox[1,4] = box[2] # x_max 
    tbox[2,4] = box[3] # y_min
    return tbox
end

function tbox_area(model, tbox, type)
    if type == :obs
        L1 = sqrt((tbox[1,1] - tbox[1,2])^2 + (tbox[2,1] - tbox[2,2])^2)
        L2 = sqrt((tbox[1,2] - tbox[1,3])^2 + (tbox[2,2] - tbox[2,3])^2)
        area = L1 * L2
        return area
    elseif type == :car 
        L1 = @expression(model, sqrt((tbox[1,1] - tbox[1,2])^2 + (tbox[2,1] - tbox[2,2])^2))
        L2 = @expression(model, sqrt((tbox[1,2] - tbox[1,3])^2 + (tbox[2,2] - tbox[2,3])^2))
        area = L1 * L2
        return area
    end
end

function tbox_point_area(model, tbox, x, y)
    x1, y1 = x, y
    area = 0
    for i in axes(tbox,2)
        x2, y2 = tbox[1,i], tbox[2,i]
        if i == size(tbox,2)
            x3, y3 = tbox[1,1], tbox[2,1]
        else
            x3, y3 = tbox[1,i+1], tbox[2,i+1]
        end
        area = @expression(model, area + 1/2 * abs(x1*(y2-y3) + x2*(y3-y1) + x3*(y1-y2)))
    end
    return area
end

function triangle_criterion(model, box, x, y, ψ, num)
    tbox_car = gen_box(model, x, y, ψ, num)
    tbox_obs = box2tbox(box)
    for i in axes(tbox_car,2) 
        @constraint(model, tbox_point_area(model, tbox_obs, tbox_car[:,i]...) >= tbox_area(model, tbox_obs, :obs))
    end
    for i in axes(tbox_obs,2)
        @constraint(model, tbox_point_area(model, tbox_car, tbox_obs[:,i]...) >= tbox_area(model, tbox_car, :car))
    end
end

function solve_ocp2(task::Task, map_env::Map, traj::Traj_trailer)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 1000)
    # set_optimizer_attribute(model, "print_level", 1)
    obs = map_env.obs
    @variables(model, begin  
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
    set_start_value.(x0, traj.x0)
    set_start_value.(y0, traj.y0)
    set_start_value.(ψ0, traj.ψ0)
    set_start_value.(x1, traj.x1)
    set_start_value.(y1, traj.y1)
    set_start_value.(ψ1, traj.ψ1)
    set_start_value.(v0, traj.v0)
    set_start_value.(a0, traj.a0)
    set_start_value.(δ0, traj.δ0)
    set_start_value.(v1, traj.v1)
    gx0 = goal[1]
    gy0 = goal[2]
    gψ0 = goal[3]
    gψ1 = goal[4]
    gv0 = goal[5]
    obj = @expression(model,
                sum([wx0*(x0[k]-gx0)^2 for k in 1:N]) + 
                sum([wy0*(y0[k]-gy0)^2 for k in 1:N]) + 
                sum([wψ0*(ψ0[k]-gψ0)^2 for k in 1:N]) + 
                sum([wψ1*(ψ1[k]-gψ1)^2 for k in 1:N]) + 
                sum([wv0*(v0[k]-gv0)^2 for k in 1:N]) + 
                sum([wa0*(a0[k])^2 for k in 1:N]) + 
                sum([wδ0*(δ0[k])^2 for k in 1:N])
    )
    @objective(model, Min, w_obj * obj)
    @constraints(model, begin
        x0[1] == start[1]
        y0[1] == start[2]
        ψ0[1] == start[3]
        ψ1[1] == start[4]
        v0[1] == start[5]
        v1[1] == start[5]
        x0[end] == goal[1]
        y0[end] == goal[2]
        ψ0[end] == goal[3]
        ψ1[end] == goal[4]
        v0[end] == goal[5]
        v1[end] == goal[5]
    end)
    @constraints(model, begin
        [k=1:N-1], x0[k+1] == x0[k] + v0[k]*cos(ψ0[k])*Δt 
        [k=1:N-1], y0[k+1] == y0[k] + v0[k]*sin(ψ0[k])*Δt
        [k=1:N-1], v0[k+1] == v0[k] + a0[k]*Δt 
        [k=1:N-1], ψ0[k+1] == ψ0[k] + v0[k]*tan(δ0[k])/d1*Δt
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
    @constraints(model, begin
        [k=1:N], abs(x0[k]-traj.x0[k]) <= Δs
        [k=1:N], abs(y0[k]-traj.y0[k]) <= Δs
        [k=1:N], abs(ψ0[k]-traj.ψ0[k]) <= Δa
        [k=1:N], abs(x1[k]-traj.x1[k]) <= Δs
        [k=1:N], abs(y1[k]-traj.y1[k]) <= Δs
        [k=1:N], abs(ψ1[k]-traj.ψ1[k]) <= Δa
    end)
    for k in 1:N
        box0 = [traj.x0[k]-Δs, traj.x0[k]+Δs, traj.y0[k]-Δs, traj.y0[k]+Δs]
        box1 = [traj.x1[k]-Δs, traj.x1[k]+Δs, traj.y1[k]-Δs, traj.y1[k]+Δs]
        for i in axes(obs,1), j in axes(obs,2)
            if overlap(box0, obs[i,j,:])
                triangle_criterion(model, obstacles[i,j,:], x0[k], y0[k], ψ0[k], 0)
            end
            if overlap(box1, obs[i,j,:])
                triangle_criterion(model, obstacles[i,j,:], x1[k], y1[k], ψ1[k], 1)
            end
        end
    end
    JuMP.optimize!(model)
    is_failed = false
    obj_value = objective_value(model)
    term_status = termination_status(model)
    if term_status != MOI.OPTIMAL
        is_failed = true
    end
    ts = collect(1:N)*value(Δt)
    traj = Traj_trailer(ts, value.(x0), value.(y0), value.(ψ0), value.(x1), value.(y1), value.(ψ1), value.(v0), value.(a0), value.(δ0), value.(v1))
    return is_failed, traj, obj_value
end

function TRMO(map_env::Map, task::Task, traj::Traj_trailer)
    iter = 0
    obj_previous = 1e8
    obj_current = 0
    obj_tol = 0.1
    iter_max2 = 10
    while iter <= iter_max2
        iter += 1
        @show iter
        is_failed, traj, obj_current = solve_ocp2(task,map_env,traj)
        @show obj_current
        if is_failed
            return traj
        end
        if abs(obj_previous - obj_current) <= obj_tol
            return traj
        else
            obj_previous = obj_current
        end
    end
    return traj
end

#

function TRTT(map_env::Map, task::Task)
    time0 = time()
    path, maze = Astar_Trailer(map_env, task)
    traj = LIOS(map_env, task, path)
    traj = TRMO(map_env, task, traj)
    time1 = time()
    sol_time = time1 - time0
    return traj, sol_time
end

traj, sol_time = TRTT(map_env,task)
sol_time = round(sol_time, digits=3)
@show sol_time
@show traj.t[end]
@show maximum(traj.v0)

fig_traj = plot(fig_env, traj.x0, traj.y0, label="tractor", title="Li-E$seed")
plot!(fig_traj, traj.x1, traj.y1, label="trailer")

##
fig_v0 = plot(traj.t, traj.v0, label="v0", legend=:outertopright)
fig_δ0 = plot(traj.t, traj.δ0, label="δ0", legend=:outertopright)
fig_a0 = plot(traj.t, traj.a0, label="a0", legend=:outertopright)
plot(fig_v0, fig_δ0, fig_a0, layout=(3,1), size=(500,500))


##
using JLD2
using CSV, DataFrames
filename = "benchmark/data/Li2022_trailer_traj_"*data_name*".jld2"
@save filename traj

method_name = "Li"*data_name

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