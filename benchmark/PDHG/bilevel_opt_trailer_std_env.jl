# N = 5
# include("bilevel_env_trailer.jl")
# include("get_PDHG_para_trailer.jl")
# include("PDHG_solver_trailer.jl")
# include("SOS_trailer.jl")

include("../utils/env_trailer.jl")
include("get_PDHG_para_trailer.jl")
include("PDHG_solver_trailer.jl")

α = 0.2; # PDHG primal step size
β = 0.4; # PDHG dual step size
η = 0.01; # Time update step size

max_iter = 320
data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar = TrajOpt_space(T0, α, β, FG; device=:GPU, max_iter=max_iter); 
@show 2*(L1+L2)*seg_N
@show sol_time

#
ts = [data[i].t for i in eachindex(data)];
x0 = [data[i].x0 for i in eachindex(data)];
y0 = [data[i].y0 for i in eachindex(data)];
x1 = [data[i].x1 for i in eachindex(data)];
y1 = [data[i].y1 for i in eachindex(data)];
ψ0 = [atan(data[i].vy0, data[i].vx0) for i in eachindex(data)];
ψ1 = [atan(data[i].vy1, data[i].vx1) for i in eachindex(data)];
v0 = [sqrt(data[i].vx0^2+data[i].vy0^2) for i in eachindex(data)];
v1 = [sqrt(data[i].vx1^2+data[i].vy1^2) for i in eachindex(data)];
a0 = [sqrt(data[i].ax0^2+data[i].ay0^2) for i in eachindex(data)];
δ0 = [data[i].ϕ for i in eachindex(data)];
fig_traj = plot(fig_env, x0, y0, label="tractor")
plot!(fig_traj, x1, y1, label="trailer")

##
# traj = Traj_trailer(ts, x0, y0, ψ0, x1, y1, ψ1, v0, a0, δ0, v1)
# using JLD2
# using CSV, DataFrames
# @save "benchmark/data/Ours2024_trailer_traj.jld2" traj

# method_name = "Ours24"

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

##
outer_iter = 5
Tw = ones(N) * 1.0
ObjLag = :Lag
grad_method = :AG
alpha = 0.08
h = 1e-2

if grad_method == :AG
    inner_iter = 500
    warm_start = false
elseif grad_method == :FD
    inner_iter = 500
    warm_start = false
end

cstars, Xstars, λstars, νstars, Times, Objs, Objts, Lags, Lagts, bilevel_soltime, gradFD_times, gradAG_times, grads_FD, grads_AG, sol_times, para_times = BilevelOpt_FDAG(T0, Tw, α, β, FG; warm_start=warm_start, outer_iter=outer_iter, inner_iter=inner_iter, alpha=alpha, h=h, Obj_Lag=ObjLag, FD_AG=grad_method) 
cstar = cstars[end]
Xstar = Xstars[end]
λstar = λstars[end]
νstar = νstars[end]
Tstar = Times[end]

#
Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(Tstar)
sopt = calc_sopt(FG, Xstar)
@show Tstar 
@show T_opt = sum(Tstar)
@show Times
@show sol_time_ave = sum(sol_times) / length(sol_times) 
if grads_FD != []
    grad_times = gradFD_times
else
    grad_times = gradAG_times
end
@show grad_time_ave = sum(grad_times) / length(grad_times)
@show para_time_ave = sum(para_times) / length(para_times)
@show bilevel_soltime
@show grads_AG

@show traj_time = sum(Tstar)
@show final_cost = Objts[end]
@show cstar' * Popt * cstar

@show cstar' * Popt * cstar + Tw' * Tstar
@show cstar' * Popt * cstar + λstar' * (Lopt * cstar - gopt - sopt) + νstar' * (Hopt * cstar - ropt) + Tw' * Tstar 
@show λstar' * (Lopt * cstar - gopt - sopt) 
@show νstar' * (Hopt * cstar - ropt)

kkt_01 = 2Popt * cstar + Lopt' * λstar + Hopt' * νstar
kkt_02 = Lopt * cstar - sopt - gopt
kkt_03 = Hopt * cstar - ropt
@show norm(kkt_01)
@show norm(kkt_02)
@show norm(kkt_03)


data = post_Bilevel(cstar, Tstar);
ts = [data[i].t for i in eachindex(data)];
x0 = [data[i].x0 for i in eachindex(data)];
y0 = [data[i].y0 for i in eachindex(data)];
x1 = [data[i].x1 for i in eachindex(data)];
y1 = [data[i].y1 for i in eachindex(data)];
ψ0 = [atan(data[i].vy0, data[i].vx0) for i in eachindex(data)];
ψ1 = [atan(data[i].vy1, data[i].vx1) for i in eachindex(data)];
v0 = [sqrt(data[i].vx0^2+data[i].vy0^2) for i in eachindex(data)];
v1 = [sqrt(data[i].vx1^2+data[i].vy1^2) for i in eachindex(data)];
a0 = [sqrt(data[i].ax0^2+data[i].ay0^2) for i in eachindex(data)];
δ0 = [data[i].ϕ for i in eachindex(data)];  
κ0 = tan.(δ0) ./ d0
@show maximum(v0)
@show maximum(a0)
@show maximum(κ0)
fig_trajT = plot(fig_env, x0, y0, label="tractor")
plot!(fig_trajT, x1, y1, label="trailer")

fig_objt = plot(collect(1:length(Objts)), Objts, label="Obj", xlabel="Iteration", ylabel="Objective")


if grad_method == :AG
    fig_traj_AG = fig_trajT
    Objts_AG = Objts
    Times_AG = Times
elseif grad_method == :FD
    fig_traj_FD = fig_trajT
    Objts_FD = Objts
    Times_FD = Times
end

@show fig_trajT


##
using PGFPlotsX
pgfplotsx()
label_font = 16
tick_font = 14
legend_font = 16

fig_grad_descend = plot(labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font, legend=:topright)
plot!(fig_grad_descend, collect(1:outer_iter), Objts_AG, label="AG", xlabel="Iteration", ylabel="Objective",linewidth=2,color=:darkgreen)
plot!(fig_grad_descend, collect(1:outer_iter), Objts_FD, label="FD", xlabel="Iteration", ylabel="Objective",linewidth=2, color=:darkred)
savefig(fig_grad_descend, "benchmark/figs/trailer_obj_pgf.tex")


##
traj = Traj_trailer(ts, x0, y0, ψ0, x1, y1, ψ1, v0, a0, δ0, v1)
using JLD2
using CSV, DataFrames
@save "benchmark/data/Ours2024T_trailer_traj.jld2" traj


