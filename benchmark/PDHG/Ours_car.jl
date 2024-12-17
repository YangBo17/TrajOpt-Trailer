include("../utils/env_car2.jl")
include("get_PDHG_para_car.jl")
include("PDHG_solver_car.jl")

α = 0.2 # PDHG primal step size
β = 0.4 # PDHG dual step size
η = 0.01 # Time update step size
max_iter = 300
data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar = TrajOpt_space(T0, α, β, FG; device=:GPU, max_iter=max_iter, warm_start=false);
@show 2*(L1+L2)*seg_N
@show sol_time

#
t = [data[i].t for i in eachindex(data)];
x = [data[i].x for i in eachindex(data)];
y = [data[i].y for i in eachindex(data)];  
ψ = [data[i].θ for i in eachindex(data)];
v = [data[i].v for i in eachindex(data)];
δ = [data[i].δ for i in eachindex(data)];
a = [data[i].a for i in eachindex(data)];
fig_opt1 = plot(fig_env, x, y)

##
traj = Traj_car(t, x, y, ψ, v, δ, a)
using JLD2
using CSV, DataFrames
@save "benchmark/data/Ours2024_car_traj.jld2" traj

method_name = "Ours24"

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
T0 = [2.0, 2.0, 2.0] * 2.0
Tw = ones(3) * 50
cstar, Xstar, Tstar, c_list, obj_list, Lag_list, T_list, bilevel_soltime = BiLevelOpt(T0, Tw, α, β, FG) 
@show Tstar
@show T_opt = sum(Tstar)
@show T_list
@show bilevel_soltime

data = post_Bilevel(cstar, Xstar, Tstar)
x = [data[i].x for i in eachindex(data)]
y = [data[i].y for i in eachindex(data)]  
fig_opt2 = plot(fig_env, x, y)

###########################################
##
Tw = ones(3) * 0.0
T1 = [2.0, 2.0, 2.0]
T2 = [2.0, 2.0, 2.0]*0.9
max_iter = 2000

T = T1
Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T)
Xstar, λstar, X_list, λ_list = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=false)
cstar, ϵstar, cstar_grad2, ϵstar_grad2, qopt = QP_diff_solver(FG,Xstar,Popt,Hopt,Lopt,ropt,gopt,Ropt)
obj = calc_obj(T,cstar,Popt; time_penalty=false, Tw=Tw)
Lag = calc_Lag(T,Popt,Ropt,qopt,cstar,ϵstar; time_penalty=true, Tw=Tw)
Lag_grad2 = calc_Lag_grad(T,Popt,Ropt,qopt,cstar,ϵstar,cstar_grad2,ϵstar_grad2; time_penalty=true,Tw=Tw)
data1 = calc_traj(T, cstar; dt = 0.02)
obj1 = obj

T = T2
Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T)
Xstar, λstar, X_list, λ_list = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=false)
cstar, ϵstar, cstar_grad2, ϵstar_grad2, qopt = QP_diff_solver(FG,Xstar,Popt,Hopt,Lopt,ropt,gopt,Ropt)
obj = calc_obj(T,cstar,Popt; time_penalty=false, Tw=Tw)
Lag = calc_Lag(T,Popt,Ropt,qopt,cstar,ϵstar; time_penalty=true, Tw=Tw)
Lag_grad2 = calc_Lag_grad(T,Popt,Ropt,qopt,cstar,ϵstar,cstar_grad2,ϵstar_grad2; time_penalty=true,Tw=Tw)
data2 = calc_traj(T, cstar; dt = 0.02)
obj2 = obj

@show obj1
@show obj2

x1 = [data1[i].x for i in eachindex(data1)]
y1 = [data1[i].y for i in eachindex(data1)]
x2 = [data2[i].x for i in eachindex(data2)]
y2 = [data2[i].y for i in eachindex(data2)]

fig_opt = plot(fig_env, x1, y1, label="T1")
plot!(fig_opt, x2, y2, label="T2")