N = 1
include("bilevel_env_trailer.jl")
include("get_PDHG_para_trailer.jl")
include("PDHG_solver_trailer.jl")
include("SOS_trailer.jl")
Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T0);
@show 2*(L1+L2)*seg_N
@show Mat_size = size(Popt,1) + size(Lopt, 1) + size(Hopt, 1)
α = 0.2; # PDHG primal step size
β = 0.4; # PDHG dual step size
η = 0.05; # Time update step size
@show Base.gc_bytes()
dim1_P, dim2_P = size(Popt)
dim1_R, dim2_R = size(Ropt)
Z_22 = zeros(dim1_R, dim1_R)
Z_1 = zeros(dim1_P)

max_iter = 500

data, X_list, λ_list, sol_time, obj, Lag, Lag_grad, cstar, λstar, νstar = TrajOpt_space_new(T0, α, β, FG; device=:GPU, max_iter=max_iter); 
Xstar = X_list[end];
sopt = calc_sopt(FG, Xstar);
@show obj
@show Lag
@show Lag_grad

@show cstar' * Popt * cstar
@show cstar' * Popt * cstar + λstar' * (Lopt * cstar - gopt - sopt) + νstar' * (Hopt * cstar - ropt)
@show λstar' * (Lopt * cstar - gopt - sopt) 
@show νstar' * (Hopt * cstar - ropt)

kkt_01 = 2Popt * cstar + Lopt' * λstar + Hopt' * νstar
kkt_02 = Lopt * cstar - sopt - gopt
kkt_03 = Hopt * cstar - ropt
@show norm(kkt_01)
@show norm(kkt_02)
@show norm(kkt_03)

##
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
fig_opt1 = plot(fig_env, x0, y0, label="tractor")
plot!(fig_opt1, x1, y1, label="trailer")

##
traj = Traj_trailer(ts, x0, y0, ψ0, x1, y1, ψ1, v0, a0, δ0, v1)
using JLD2
using CSV, DataFrames
@save "benchmark/data/Ours2024T0_trailer_traj.jld2" traj

##
outer_iter = 100
inner_iter = 500
Tw = ones(N) * 1.0

cstars, Times, Objs, Objts, Lags, Lagts, bilevel_soltime, grad_times, grads_FD, grads_AG, kkt01s, kkt02s, kkt03s, lag_λs, lag_νs = BilevelOpt_FDAG(T0, Tw, α, β, FG; outer_iter=outer_iter, inner_iter=inner_iter, alpha=0.05, h=1e-1, Obj_Lag=:Obj, FD_AG=:FD) 

@show grads_FD  
@show grad_time_ave = sum(grad_times) / length(grad_times)
@show bilevel_soltime

data = calc_traj(Times[end], cstars[end]; dt=0.02);
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

fig_traj = plot(fig_env, x0, y0, label="tractor")
plot!(fig_traj, x1, y1, label="trailer")


len = 50
fig_grad_FD = plot(collect(1:len), [grads_FD[i][1] for i in 1:len], label="FD-grad-sec1", xlabel="Iteration", ylabel="Gradient", ylims=(-2.0,+2.0))
# plot!(fig_grad_FD, collect(1:len), [grads_FD[i][2] for i in 1:len], label="FD-grad-sec2", xlabel="Iteration", ylabel="Gradient")

fig_grad_AG = plot(collect(1:len), [grads_AG[i][1] for i in 1:len], label="AG-grad-sec1", xlabel="Iteration", ylabel="Gradient", ylims=(-2.0,+2.0))
# plot!(fig_grad_AG, collect(1:len), [grads_AG[i][2] for i in 1:len], label="AG-grad-sec2", xlabel="Iteration", ylabel="Gradient")
fig_grad = plot(fig_grad_FD, fig_grad_AG, layout=(2,1))

fig_Lagt = plot(collect(1:len), [Lagts[i] for i in 1:len], label="Lagrange", xlabel="Iteration")
fig_Objt = plot(collect(1:len), [Objts[i] for i in 1:len], label="Objective", xlabel="Iteration")
fig_Lagt_Objt = plot(fig_Lagt, fig_Objt, layout=(2,1))


fig_time = plot(collect(1:outer_iter), [Times[i][1] for i in 1:outer_iter], label="Time_sec1", xlabel="Iteration", ylabel="Time")
# plot!(fig_time, collect(1:outer_iter), [Times[i][2] for i in 1:outer_iter], label="Time_sec2", xlabel="Iteration", ylabel="Time")
fig_kkt = plot(collect(1:outer_iter), kkt01s, label="KKT01", xlabel="Iteration")
plot!(fig_kkt, collect(1:outer_iter), kkt02s, label="KKT02", xlabel="Iteration")
plot!(fig_kkt, collect(1:outer_iter), kkt03s, label="KKT03", xlabel="Iteration")
fig_lag_λν = plot(collect(1:outer_iter), lag_λs, label="Lag_λ", xlabel="Iteration")
plot!(fig_lag_λν, collect(1:outer_iter), lag_νs, label="Lag_ν", xlabel="Iteration")


plot(fig_traj, fig_kkt, fig_lag_λν, layout=(3,1), size=(600,600))

plot(fig_grad, fig_Lagt_Objt, fig_time, layout=(3,1),size=(800,1000))


##
outer_iter = 100
inner_iter = 500
Tw = ones(N) * 1.0

cstar, Xstar, λstar, νstar, Tstar, sopt, c_list, obj_list, Lag_list, Lag_grad_list, T_list, sol_times, bilevel_soltime, grad_times, para_times = BiLevelOpt_new(T0, Tw, α, β, FG; outer_iter=outer_iter, inner_iter=inner_iter) 
Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(Tstar)

grad_AG = Lag_grad_list
@show Tstar 
@show T_opt = sum(Tstar)
@show T_list
@show sol_time_ave = sum(sol_times) / length(sol_times) 
@show grad_time_ave = sum(grad_times) / length(grad_times)
@show para_time_ave = sum(para_times) / length(para_times)
@show bilevel_soltime
@show Lag_grad_list

@show traj_time = sum(Tstar)
@show final_cost = obj_list[end]

@show cstar' * Popt * cstar + traj_time
@show cstar' * Popt * cstar + λstar' * (Lopt * cstar - gopt - sopt) + νstar' * (Hopt * cstar - ropt) + traj_time 
@show λstar' * (Lopt * cstar - gopt - sopt) 
@show νstar' * (Hopt * cstar - ropt)

kkt_01 = 2Popt * cstar + Lopt' * λstar + Hopt' * νstar
kkt_02 = Lopt * cstar - sopt - gopt
kkt_03 = Hopt * cstar - ropt
@show norm(kkt_01)
@show norm(kkt_02)
@show norm(kkt_03)


data = post_Bilevel(cstar, Xstar, Tstar);
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
fig_opt2 = plot(fig_env, x0, y0, label="tractor")
plot!(fig_opt2, x1, y1, label="trailer")

len = 20
fig_grad = plot(collect(1:len), [grad_AG[i][1] for i in 1:len], label="grad-sec1", xlabel="Iteration", ylabel="Gradient")
# plot!(fig_grad, collect(1:len), [grad_AG[i][2] for i in 1:len], label="grad-sec2", xlabel="Iteration", ylabel="Gradient")

fig_time = plot(collect(1:length(T_list)), [T_list[i][1] for i in 1:length(T_list)], label="Time_sec1", xlabel="Iteration", ylabel="Time")
# plot!(fig_time, collect(1:length(T_list)), [T_list[i][2] for i in 1:length(T_list)], label="Time_sec2", xlabel="Iteration", ylabel="Time")

##
traj = Traj_trailer(ts, x0, y0, ψ0, x1, y1, ψ1, v0, a0, δ0, v1)
using JLD2
using CSV, DataFrames
@save "benchmark/data/Ours2024Ts_trailer_traj.jld2" traj

@save "benchmark/data/trailer_T_list.jld2" T_list
@save "benchmark/data/trailer_Lag_list.jld2" Lag_list
@save "benchmark/data/trailer_obj_list.jld2" obj_list

##
# T0 = Tstar
p0, model, pre_time = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list; solver_name=:Mosek)
solve_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
obj_val = objective_value(model)
@show status = termination_status(model)
@show solve_time = round(solve_time, digits=3)
@show obj_val
@show pre_time

data = process_data(p0, T0, trailer_body);
x0 = [data[i].x0 for i in eachindex(data)];
y0 = [data[i].y0 for i in eachindex(data)];
x1 = [data[i].x1 for i in eachindex(data)];
y1 = [data[i].y1 for i in eachindex(data)];

fig_opt1_sos = plot(fig_env, x0, y0, label="tractor")
plot!(fig_opt1_sos, x1, y1, label="trailer")


##
outer_iter = 100
Tw = ones(N) * 1.0
Times, Objs, Objts, ps, bilevel_soltime, grad_times, grad_FD = trajopt_time(T0, trajopt_set, cons_init, limits_list, obs_list; solver_name=:Mosek, outer_iter=outer_iter, tweight=Tw)
@show grad_FD  
@show grad_time_ave = sum(grad_times) / length(grad_times)
@show bilevel_soltime
  
@show final_cost = Objts[end] 

data = process_data(ps[end], Times[end], trailer_body);
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

fig_opt2_sos = plot(fig_env, x0, y0, label="tractor")
plot!(fig_opt2_sos, x1, y1, label="trailer")

len = 10
fig_grad = plot(collect(1:len), [grad_FD[i][1] for i in 1:len], label="grad-sec1", xlabel="Iteration", ylabel="Gradient")
plot!(fig_grad, collect(1:len), [grad_FD[i][2] for i in 1:len], label="grad-sec2", xlabel="Iteration", ylabel="Gradient")

fig_time = plot(collect(1:outer_iter), [Times[i][1] for i in 1:outer_iter], label="Time_sec1", xlabel="Iteration", ylabel="Time")
plot!(fig_time, collect(1:outer_iter), [Times[i][2] for i in 1:outer_iter], label="Time_sec2", xlabel="Iteration", ylabel="Time")

##
println("PDHG_Time")
T_list

##
println("Mosek_Time")
Times

##
println("PDHG_grad")
Lag_grad_list

##
println("Mosek_grad")
mosek_grads