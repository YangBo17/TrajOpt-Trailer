#=
TRO, 2022, Francisco Eiras
A Two-Stage Optimization-Based Motion Planner for Safe Urban Driving
=#
using Revise
using JLD2
using Plots
using AStarSearch
using JuMP, Ipopt, Cbc, HiGHS, COPT
include("../common/common.jl")
@load "data/cons_limits.jld2" cons_limits
@load "data/cons_obs.jld2" cons_obs

#=
Urban driving, the environment generation is different from previous simulation
=#
# env settings
include("utils/env_car2.jl")

# number settings
N = 300
Δt = T_total/N
H = 30
ρ = 1.0
d = 0.2

# data struct
mutable struct LTraj
    t::Array
    x::Array
    y::Array
    vx::Array
    vy::Array
    ax::Array
    ay::Array
end

# function
function NLP(traj0)
    model = Model(Ipopt.Optimizer)
    set_optimizer_attribute(model, "max_iter", 2000)
    # set_optimizer_attribute(model, "print_level", 1)
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
    @constraints(model, begin
        [k=1:N-1], x[k+1] == x[k] + v[k] * cos(ψ[k]) * Δt
        [k=1:N-1], y[k+1] == y[k] + v[k] * sin(ψ[k]) * Δt
        [k=1:N-1], ψ[k+1] == ψ[k] + v[k] * tan(δ[k]) / L * Δt
        [k=1:N-1], v[k+1] == v[k] + a[k] * Δt
    end)
    @constraints(model, begin
        x[1] == start_line.x
        y[1] == start_line.y
        ψ[1] == start_line.ψ
        v[1] == start_line.v
        x[end] == goal_line.x
        y[end] == goal_line.y
        ψ[end] == goal_line.ψ
        v[end] == goal_line.v
    end) 
    gx = goal_line.x
    gy = goal_line.y
    gψ = goal_line.ψ
    gv = goal_line.v
    @constraints(model, begin
        [k=1:N], δ_min <= δ[k] <= δ_max
        [k=1:N], a_min <= a[k] <= a_max
        [k=1:N], v_min <= v[k] <= v_max
    end)
    @variables(model, begin
        x_α[1:N, 1:4]
        y_α[1:N, 1:4]
    end)
    for k in 1:N
        for j in 1:4
            @constraints(model, begin
                x_α[k,j] == x[k] + A[j][1] * Wt/2 * cos(ψ[k]) - A[j][2] * Lt/2 * sin(ψ[k])
                y_α[k,j] == y[k] + A[j][1] * Wt/2 * sin(ψ[k]) + A[j][2] * Lt/2 * cos(ψ[k])
            end)
        end
    end
    @constraints(model, begin
        [k=1:N, j=1:4], y_α[k,j] >= br
        [k=1:N, j=1:4], y_α[k,j] <= bl
    end)
    R = zeros(2,2,3)
    car = car_line
    for i in eachindex(car)
        R[1,1,i] = cos(car[i].ψ)^2/ae^2 + sin(car[i].ψ)^2/be^2
        R[1,2,i] = -sin(car[i].ψ)*cos(car[i].ψ)/ae^2 + sin(car[i].ψ)*cos(car[i].ψ)/be^2
        R[2,1,i] = -sin(car[i].ψ)*cos(car[i].ψ)/ae^2 + sin(car[i].ψ)*cos(car[i].ψ)/be^2
        R[2,2,i] = sin(car[i].ψ)^2/ae^2 + cos(car[i].ψ)^2/be^2
        # @show R[:,:,i]
    end
    # @constraints(model, begin
    #     [k=1:N, j=1:4, i=1:3], (x_α[k,j]-car[i].x)^2*R[1,1,i] + (x_α[k,j]-car[i].x)*(y_α[k,j]-car[i].y)*R[1,2,i] + (x_α[k,j]-car[i].x)*(y_α[k,j]-car[i].y)*R[2,1,i] + (y_α[k,j]-car[i].y)^2*R[2,2,i] >= 1
    # end)

    @constraints(model, begin
        [k=1:N, i=1:2], (x[k]-car[i].x)^2*R[1,1,i] + (x[k]-car[i].x)*(y[k]-car[i].y)*R[1,2,i] + (x[k]-car[i].x)*(y[k]-car[i].y)*R[2,1,i] + (y[k]-car[i].y)^2*R[2,2,i] >= 1
    end)

    obj = @expression(model,
                sum([ωx*(x[k]-gx)^2 for k in 1:N]) + 
                sum([ωy*(y[k]-gy)^2 for k in 1:N]) +  
                sum([ωv*(v[k]-gv)^2 for k in 1:N]) + 
                sum([ωψ*(ψ[k]-gψ)^2 for k in 1:N]) + 
                sum([ωa*(a[k])^2 for k in 1:N]) + 
                sum([ωδ*(δ[k])^2 for k in 1:N])
    )
    @objective(model, Min, obj)
    JuMP.optimize!(model)
    ts = collect(1:N)*Δt
    traj = Traj_car(ts, value.(x), value.(y), value.(ψ), value.(v), value.(δ), value.(a))
    return traj
end

function MILP(init, ltraj0, H)
    cons = cons_line
    # model = Model(Cbc.Optimizer)
    # model = Model(HiGHS.Optimizer)
    model = Model(COPT.Optimizer)
    set_optimizer_attribute(model, "Logging", 0)
    # set_optimizer_attribute(model, "maxIterations", 1000)
    @variables(model, begin
        x[1:H]
        y[1:H]
        vx[1:H]
        vy[1:H]
        ax[1:H]
        ay[1:H]
    end)
    for k in 1:H
        set_start_value(x[k], ltraj0.x[k])
        set_start_value(y[k], ltraj0.y[k])
        set_start_value(vx[k], ltraj0.vx[k])
        set_start_value(vy[k], ltraj0.vy[k])
        set_start_value(ax[k], ltraj0.ax[k])
        set_start_value(ay[k], ltraj0.ay[k])
    end
    @constraints(model, begin
        x[1] == init[1]
        y[1] == init[2]
        vx[1] == init[3]
        vy[1] == init[4]
    end)
    # @constraints(model, begin
    #     x[end] == goal_line.x
    #     y[end] == goal_line.y
    #     vx[end] == goal_line.v*cos(goal_line.ψ)
    #     vy[end] == goal_line.v*sin(goal_line.ψ)
    # end)
    @constraints(model, begin
        [k=1:H-1], x[k+1] == x[k] + vx[k] * Δt
        [k=1:H-1], y[k+1] == y[k] + vy[k] * Δt
        [k=1:H-1], vx[k+1] == vx[k] + ax[k] * Δt
        [k=1:H-1], vy[k+1] == vy[k] + ay[k] * Δt  
    end)
    @variables(model, begin
        vy_abs[1:H]
    end)
    @constraints(model, begin
        [k=1:H], vy_abs[k] >= vy[k]
        [k=1:H], vy_abs[k] >= -vy[k]
        [k=1:H], vx[k] >= ρ * vy_abs[k]
    end)
    @constraints(model, begin
        [k=1:H], vx[k] >= 0.0
        [k=1:H], vx[k] <= vx_max 
        [k=1:H], vy[k] >= vy_min
        [k=1:H], vy[k] <= vy_max
        [k=1:H], ax[k] >= ax_min
        [k=1:H], ax[k] <= ax_max
        [k=1:H], ay[k] >= ay_min
        [k=1:H], ay[k] <= ay_max
    end)
    # @constraints(model, begin
    #     [k=1:H-1], ax[k+1] - ax[k] >= -da_max * Δt
    #     [k=1:H-1], ax[k+1] - ax[k] <= da_max * Δt
    #     [k=1:H-1], ay[k+1] - ay[k] >= -da_max * Δt
    #     [k=1:H-1], ay[k+1] - ay[k] <= da_max * Δt
    # end)
    @constraints(model, begin
        [k=1:H], y[k] >= br + d
        [k=1:H], y[k] <= bl - d
    end)
    M = 1e4
    @variables(model, begin
        z1[1:H,1:3], Bin
        z2[1:H,1:3], Bin
        μ1[1:H,1:3] >= 0
        μ2[1:H,1:3] >= 0
    end)
    @constraints(model, begin
        # car1
        [k=1:H], μ1[k,1] >= cons[1,1] - x[k]
        [k=1:H], μ1[k,1] <= M * z1[k,1]
        [k=1:H], μ1[k,1] <= cons[1,1] - x[k] + M * (1-z1[k,1])

        [k=1:H], μ1[k,2] >= x[k] - cons[1,2]
        [k=1:H], μ1[k,2] <= M * z1[k,2]
        [k=1:H], μ1[k,2] <= x[k] - cons[1,2] + M * (1-z1[k,2])

        [k=1:H], μ1[k,3] >= cons[1,3] - x[k]
        [k=1:H], μ1[k,3] <= M * z1[k,3]
        [k=1:H], μ1[k,3] <= cons[1,3] - x[k] + M * (1-z1[k,3])

        [k=1:H], cons[1,4] - M * (μ1[k,1] + μ1[k,2] + μ1[k,3]) <= y[k]
        # car2
        [k=1:H], μ2[k,1] >= cons[2,1] - x[k]
        [k=1:H], μ2[k,1] <= M * z2[k,1]
        [k=1:H], μ2[k,1] <= cons[2,1] - x[k] + M * (1-z2[k,1]) 

        [k=1:H], μ2[k,2] >= x[k] - cons[2,2]
        [k=1:H], μ2[k,2] <= M * z2[k,2]
        [k=1:H], μ2[k,2] <= x[k] - cons[2,2] + M * (1-z2[k,2])

        [k=1:H], μ2[k,3] >= cons[2,4] - x[k]
        [k=1:H], μ2[k,3] <= M * z2[k,3]
        [k=1:H], μ2[k,3] <= cons[2,4] - x[k] + M * (1-z2[k,3])
        
        [k=1:H], cons[2,3] + M * (μ2[k,1] + μ2[k,2] + μ2[k,3]) >= y[k]
    end)
    @variables(model, begin
        Θx_pos[1:H] >= 0
        Θx_neg[1:H] >= 0
        Θy_pos[1:H] >= 0
        Θy_neg[1:H] >= 0
        Θvx_pos[1:H] >= 0
        Θvx_neg[1:H] >= 0
        Θvy_pos[1:H] >= 0
        Θvy_neg[1:H] >= 0
        Θax_pos[1:H] >= 0
        Θax_neg[1:H] >= 0
        Θay_pos[1:H] >= 0
        Θay_neg[1:H] >= 0
    end)
    @constraints(model, begin
        [k=1:H], Θx_pos[k] - Θx_neg[k] == x[k] - goal_line.x
        [k=1:H], Θy_pos[k] - Θy_neg[k] == y[k] - goal_line.y
        [k=1:H], Θvx_pos[k] - Θvx_neg[k] == vx[k] - goal_line.v*cos(goal_line.ψ)
        [k=1:H], Θvy_pos[k] - Θvy_neg[k] == vy[k] - goal_line.v*sin(goal_line.ψ)
        [k=1:H], Θax_pos[k] - Θax_neg[k] == ax[k] - 0.0
        [k=1:H], Θay_pos[k] - Θay_neg[k] == ay[k] - 0.0
    end)
    @objective(model, Min, sum([Ωx*(Θx_pos[k]+Θx_neg[k])+Ωy*(Θy_pos[k]+Θy_neg[k])+Ωvx*(Θvx_pos[k]+Θvx_neg[k])+Ωvy*(Θvy_pos[k]+Θvy_neg[k])+Ωax*(Θax_pos[k]+Θax_neg[k])+Ωay*(Θay_pos[k]+Θay_neg[k]) for k in 1:H]))
    # @objective(model, Min, sum([Θay_pos[k]+Θay_neg[k] for k in 1:H]))
    JuMP.optimize!(model);
    # @show termination_status(model)
    ts = collect(1:H)*Δt
    # @show value.(x)
    # @show value.(y)
    ltraj = LTraj(ts, value.(x), value.(y), value.(vx), value.(vy), value.(ax), value.(ay))
    return ltraj
end

function pure_MILP(init, ltraj0, H)
    cons = cons_line
    # model = Model(Cbc.Optimizer)
    # model = Model(HiGHS.Optimizer)
    model = Model(COPT.Optimizer)
    set_optimizer_attribute(model, "Logging", 0)
    @variables(model, begin
        x[1:H]
        y[1:H]
        vx[1:H]
        vy[1:H]
        ax[1:H]
        ay[1:H]
    end)
    for k in 1:H
        set_start_value(x[k], ltraj0.x[k])
        set_start_value(y[k], ltraj0.y[k])
        set_start_value(vx[k], ltraj0.vx[k])
        set_start_value(vy[k], ltraj0.vy[k])
        set_start_value(ax[k], ltraj0.ax[k])
        set_start_value(ay[k], ltraj0.ay[k])
    end
    @constraints(model, begin
        x[1] == init[1]
        y[1] == init[2]
        vx[1] == init[3]
        vy[1] == init[4]
    end)
    # @constraints(model, begin
    #     x[end] == goal_line.x
    #     y[end] == goal_line.y
    #     vx[end] == goal_line.v*cos(goal_line.ψ)
    #     vy[end] == goal_line.v*sin(goal_line.ψ)
    # end)
    @constraints(model, begin
        [k=1:H-1], x[k+1] == x[k] + vx[k] * Δt
        [k=1:H-1], y[k+1] == y[k] + vy[k] * Δt
        [k=1:H-1], vx[k+1] == vx[k] + ax[k] * Δt
        [k=1:H-1], vy[k+1] == vy[k] + ay[k] * Δt  
    end)
    @variables(model, begin
        vy_abs[1:H]
    end)
    @constraints(model, begin
        [k=1:H], vy_abs[k] >= vy[k]
        [k=1:H], vy_abs[k] >= -vy[k]
        [k=1:H], vx[k] >= ρ * vy_abs[k]
    end)
    @constraints(model, begin
        [k=1:H], vx[k] >= 0.0
        [k=1:H], vx[k] <= vx_max 
        [k=1:H], vy[k] >= vy_min
        [k=1:H], vy[k] <= vy_max
        [k=1:H], ax[k] >= ax_min
        [k=1:H], ax[k] <= ax_max
        [k=1:H], ay[k] >= ay_min
        [k=1:H], ay[k] <= ay_max
    end)
    # @constraints(model, begin
    #     [k=1:H-1], ax[k+1] - ax[k] >= -da_max * Δt
    #     [k=1:H-1], ax[k+1] - ax[k] <= da_max * Δt
    #     [k=1:H-1], ay[k+1] - ay[k] >= -da_max * Δt
    #     [k=1:H-1], ay[k+1] - ay[k] <= da_max * Δt
    # end)
    @constraints(model, begin
        [k=1:H], y[k] >= br + d
        [k=1:H], y[k] <= bl - d
    end)
    M = 1e4
    @variables(model, begin
        z1[1:H,1:3], Bin
        z2[1:H,1:3], Bin
        μ1[1:H,1:3] >= 0
        μ2[1:H,1:3] >= 0
    end)
    @constraints(model, begin
        # car1
        [k=1:H], μ1[k,1] >= cons[1,1] - x[k]
        [k=1:H], μ1[k,1] <= M * z1[k,1]
        [k=1:H], μ1[k,1] <= cons[1,1] - x[k] + M * (1-z1[k,1])

        [k=1:H], μ1[k,2] >= x[k] - cons[1,2]
        [k=1:H], μ1[k,2] <= M * z1[k,2]
        [k=1:H], μ1[k,2] <= x[k] - cons[1,2] + M * (1-z1[k,2])

        [k=1:H], μ1[k,3] >= cons[1,3] - x[k]
        [k=1:H], μ1[k,3] <= M * z1[k,3]
        [k=1:H], μ1[k,3] <= cons[1,3] - x[k] + M * (1-z1[k,3])

        [k=1:H], cons[1,4] - M * (μ1[k,1] + μ1[k,2] + μ1[k,3]) <= y[k]
        # car2
        [k=1:H], μ2[k,1] >= cons[2,1] - x[k]
        [k=1:H], μ2[k,1] <= M * z2[k,1]
        [k=1:H], μ2[k,1] <= cons[2,1] - x[k] + M * (1-z2[k,1]) 

        [k=1:H], μ2[k,2] >= x[k] - cons[2,2]
        [k=1:H], μ2[k,2] <= M * z2[k,2]
        [k=1:H], μ2[k,2] <= x[k] - cons[2,2] + M * (1-z2[k,2])

        [k=1:H], μ2[k,3] >= cons[2,4] - x[k]
        [k=1:H], μ2[k,3] <= M * z2[k,3]
        [k=1:H], μ2[k,3] <= cons[2,4] - x[k] + M * (1-z2[k,3])
        
        [k=1:H], cons[2,3] + M * (μ2[k,1] + μ2[k,2] + μ2[k,3]) >= y[k]
    end)
    @variables(model, begin
        Θx_pos[1:H] >= 0
        Θx_neg[1:H] >= 0
        Θy_pos[1:H] >= 0
        Θy_neg[1:H] >= 0
        Θvx_pos[1:H] >= 0
        Θvx_neg[1:H] >= 0
        Θvy_pos[1:H] >= 0
        Θvy_neg[1:H] >= 0
        Θax_pos[1:H] >= 0
        Θax_neg[1:H] >= 0
        Θay_pos[1:H] >= 0
        Θay_neg[1:H] >= 0
    end)
    @constraints(model, begin
        [k=1:H], Θx_pos[k] - Θx_neg[k] == x[k] - goal_line.x
        [k=1:H], Θy_pos[k] - Θy_neg[k] == y[k] - goal_line.y
        [k=1:H], Θvx_pos[k] - Θvx_neg[k] == vx[k] - goal_line.v*cos(goal_line.ψ)
        [k=1:H], Θvy_pos[k] - Θvy_neg[k] == vy[k] - goal_line.v*sin(goal_line.ψ)
        [k=1:H], Θax_pos[k] - Θax_neg[k] == ax[k] - 0.0
        [k=1:H], Θay_pos[k] - Θay_neg[k] == ay[k] - 0.0
    end)
    @objective(model, Min, sum([Ωx*(Θx_pos[k]+Θx_neg[k])+Ωy*(Θy_pos[k]+Θy_neg[k])+Ωvx*(Θvx_pos[k]+Θvx_neg[k])+Ωvy*(Θvy_pos[k]+Θvy_neg[k])+Ωax*(Θax_pos[k]+Θax_neg[k])+Ωay*(Θay_pos[k]+Θay_neg[k]) for k in 1:H]))
    # @objective(model, Min, sum([Θay_pos[k]+Θay_neg[k] for k in 1:H]))
    JuMP.optimize!(model);
    @show termination_status(model)
    ts = collect(1:H)*Δt
    # @show value.(x)
    # @show value.(y)
    ltraj = LTraj(ts, value.(x), value.(y), value.(vx), value.(vy), value.(ax), value.(ay))
    return ltraj
end

function rh_MILP(H,N)
    ltrajN = LTraj(collect(1:N)*Δt, zeros(N), zeros(N), zeros(N), zeros(N), zeros(N), zeros(N))
    ltraj = LTraj(collect(1:H)*Δt, zeros(H), zeros(H), zeros(H), zeros(H), zeros(H), zeros(H))
    for i in 1:N-H
        if i == 1
            init = [start_line.x, start_line.y, cos(start_line.ψ), sin(start_line.ψ)]
            ltraj0 = ltraj
        else
            init = [ltraj.x[2], ltraj.y[2], ltraj.vx[2], ltraj.vy[2]]
            ltraj0 = LTraj(collect(1:H)*Δt, vcat(ltraj.x[2:end],ltraj.x[end]), vcat(ltraj.y[2:end],ltraj.y[end]), vcat(ltraj.vx[2:end],ltraj.vx[end]), vcat(ltraj.vy[2:end],ltraj.vy[end]), vcat(ltraj.ax[2:end],ltraj.ax[end]), vcat(ltraj.ay[2:end],ltraj.ay[end]))
            # ltraj0 = LTraj(collect(1:H)*Δt, zeros(H), zeros(H), zeros(H), zeros(H), zeros(H), zeros(H))
        end
        # @show init
        # @show i
        ltraj = MILP(init, ltraj0, H)

        ltrajN.x[i] = ltraj.x[1]
        ltrajN.y[i] = ltraj.y[1]
        ltrajN.vx[i] = ltraj.vx[1]
        ltrajN.vy[i] = ltraj.vy[1]
        ltrajN.ax[i] = ltraj.ax[1]
        ltrajN.ay[i] = ltraj.ay[1]
    end
    for i in N-H+1:N
        ltrajN.x[i] = ltraj.x[i-N+H]
        ltrajN.y[i] = ltraj.y[i-N+H]
        ltrajN.vx[i] = ltraj.vx[i-N+H]
        ltrajN.vy[i] = ltraj.vy[i-N+H]
        ltrajN.ax[i] = ltraj.ax[i-N+H]
        ltrajN.ay[i] = ltraj.ay[i-N+H]
    end
    return ltrajN
end

function ltraj2traj(ltraj)
    Nl = length(ltraj.t)
    traj = Traj_car(collect(1:Nl)*Δt, zeros(Nl), zeros(Nl), zeros(Nl), zeros(Nl), zeros(Nl), zeros(Nl))
    for i in eachindex(ltraj.t)
        traj.x[i] = ltraj.x[i]
        traj.y[i] = ltraj.y[i]
        traj.ψ[i] = atan(ltraj.y[i], ltraj.x[i])
        traj.v[i] = sqrt(ltraj.vx[i]^2+ltraj.vy[i]^2)
        a_lateral = ltraj.ax[i] * sin(traj.ψ[i]) - ltraj.ay[i] * cos(traj.ψ[i])
        traj.δ[i] = atan(L*a_lateral, ltraj.vx[i]^2+ltraj.vy[i]^2)
        traj.a[i] = sqrt(ltraj.ax[i]^2+ltraj.ay[i]^2)
    end
    return traj
end


function test_MILP(H)
    ltraj0 = LTraj(collect(1:H)*Δt, zeros(H), zeros(H), zeros(H), zeros(H), zeros(H), zeros(H))
    init = [start_line.x, start_line.y, start_line.v*cos(start_line.ψ), start_line.v*sin(start_line.ψ)]
    ltraj = MILP(init, ltraj0, H)
    traj = ltraj2traj(ltraj)
    return traj
end

function test_pureMILP(N)
    ltraj0 = LTraj(collect(1:N)*Δt, zeros(N), zeros(N), zeros(N), zeros(N), zeros(N), zeros(N))
    init = [start_line.x, start_line.y, start_line.v*cos(start_line.ψ), start_line.v*sin(start_line.ψ)]
    ltraj = pure_MILP(init, ltraj0, N)
    traj = ltraj2traj(ltraj)
    return traj

end

function test_rhMILP(H,N)
    ltraj = rh_MILP(H, N)
    traj = ltraj2traj(ltraj)
    return traj
end

function test_NLP()
    @show X0 = range(0,goal_line.x,N) |> collect
    traj0 = Traj_car(collect(1:N)*Δt, X0, zeros(N), zeros(N), zeros(N), zeros(N), zeros(N))
    traj = NLP(traj0)
    return traj
end

# @time traj = test_NLP()
# fig_traj = plot(fig_env_line, traj.x, traj.y, aspect_ratio=:equal, title="NLP_traj", label="")
# ylims!(fig_traj, -1.5, 1.5)
# xlims!(fig_traj, -1.0,13.0)
# fig_v = plot(traj.t, traj.v, label="v", legend=:outertopright)
# fig_a = plot(traj.t, traj.a, label="a", legend=:outertopright)
# fig_δ = plot(traj.t, traj.δ, label="δ", legend=:outertopright)
# plot(fig_traj, fig_v, fig_a, fig_δ, layout=(4,1), size=(500,500))

#
function TSTO(H,N)
    ltraj = rh_MILP(H, N)
    traj = ltraj2traj(ltraj)
    traj = NLP(traj)
    return traj
end
time0 = time()
traj = TSTO(H,N)
time1 = time()
sol_time = time1 - time0
sol_time = round(sol_time, digits=3)
fig_traj = plot(fig_env, traj.x, traj.y, aspect_ratio=:equal, title="NLP_traj", label="")


##
using JLD2
using CSV, DataFrames
@save "benchmark/data/Eira2022_car_traj.jld2" traj

method_name = "Eira22"

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


##
@time traj = test_pureMILP(N)
fig_traj = plot(fig_env, traj.x, traj.y, aspect_ratio=:equal, title="pureMILP_traj", label="") 

##
@time traj = test_rhMILP(H,N)
fig_traj = plot(fig_env, traj.x, traj.y, aspect_ratio=:equal, title="rhMILP_traj", label="")
# fig_v = plot(traj.t, traj.v, label="v", legend=:outertopright)
# fig_a = plot(traj.t, traj.a, label="a", legend=:outertopright)
# fig_δ = plot(traj.t, traj.δ, label="δ", legend=:outertopright)
# plot(fig_v, fig_a, fig_δ, layout=(3,1), size=(500,500))

##
traj = test_MILP(N)
fig_traj = plot(fig_env, traj.x, traj.y, aspect_ratio=:equal, title="MILP_traj", label="")


