N = 10
include("scale_env_car.jl")
include("get_PDHG_para_car.jl")
include("PDHG_solver_car.jl")
include("SOS_car.jl")

@show Num_PSD = 2*K*N

# α = 0.2 # PDHG primal step size
# β = 0.4 # PDHG dual step size

# max_iter = 200
# data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar = TrajOpt_space(T0, α, β, FG; device=:GPU, max_iter=max_iter, warm_start=false);
# sol_time = round(sol_time, digits=3)
# @show 2*(L1+L2)*seg_N
# @show sol_time
# sol_time_ours = sol_time

# x = [data[i].x for i in eachindex(data)];
# y = [data[i].y for i in eachindex(data)];  model
# fig_opt1 = plot(fig_env, x, y)

# ## 1. COPT, commertial
# p0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list; solver_name=:COPT)
# sol_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
# @show sol_time
# sol_time_copt = sol_time

# data0 = process_data(p0, T0, car_body);
# data = data0;

# x = [data[i].x for i in eachindex(data)];
# y = [data[i].y for i in eachindex(data)];
# fig_opt = plot(fig_env, x, y)

# 2. Mosek, commertial
p0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list; solver_name=:Mosek)
sol_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
@show sol_time
sol_time_mosek = sol_time

data0 = process_data(p0, T0, car_body);
data = data0;

x = [data[i].x for i in eachindex(data)];
y = [data[i].y for i in eachindex(data)];
fig_opt = plot(fig_env, x, y)

# # 3. SCS, stanford
# p0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list; solver_name=:SCS)
# sol_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
# @show sol_time
# sol_time_scs = sol_time

# data0 = process_data(p0, T0, car_body);
# data = data0;

# x = [data[i].x for i in eachindex(data)];
# y = [data[i].y for i in eachindex(data)];
# fig_opt = plot(fig_env, x, y)

## 4. COSMO, oxford
p0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list; solver_name=:COSMO)
sol_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
@show sol_time
sol_time_cosmo = sol_time

data0 = process_data(p0, T0, car_body);
data = data0;

x = [data[i].x for i in eachindex(data)];
y = [data[i].y for i in eachindex(data)];
fig_opt = plot(fig_env, x, y)


# ## 5. SDPA
# p0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list; solver_name=:SDPA)
# sol_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
# @show sol_time
# sol_time_sdpa = sol_time

# data0 = process_data(p0, T0, car_body);
# data = data0;

# x = [data[i].x for i in eachindex(data)];
# y = [data[i].y for i in eachindex(data)];
# fig_opt = plot(fig_env, x, y)

# ## 6. ProxSDP
# p0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list; solver_name=:ProxSDP)
# sol_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
# @show sol_time
# sol_time_proxsdp = sol_time

# data0 = process_data(p0, T0, car_body);
# data = data0;

# x = [data[i].x for i in eachindex(data)];
# y = [data[i].y for i in eachindex(data)];
# fig_opt = plot(fig_env, x, y)

# ## 7. CSDP
# p0, model = trajopt_space(T0, trajopt_set, cons_init, limits_list, obs_list; solver_name=:CSDP)
# sol_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
# @show sol_time
# sol_time_csdp = sol_time

# data0 = process_data(p0, T0, car_body);
# data = data0;

# x = [data[i].x for i in eachindex(data)];
# y = [data[i].y for i in eachindex(data)];
# fig_opt = plot(fig_env, x, y)

## collect data
using CSV, DataFrames

df = CSV.read("benchmark/data/scalability_study.csv", DataFrame)

# solver_name = ["Cones", "Ours", "COPT", "Mosek", "COSMO", "SCS", "SDPA", "CSDP"]
# sol_time = [Num_PSD, sol_time_ours, sol_time_copt, sol_time_mosek, sol_time_cosmo, sol_time_scs, sol_time_sdpa, sol_time_csdp]
solver_name = ["Cones", "Ours", "COPT", "Mosek", "SCS", "COSMO"]
sol_times = [Num_PSD, sol_time_ours, sol_time_copt, sol_time_mosek, sol_time_scs, sol_time_cosmo]

column_name = Symbol("N_$N")

if !("Solver" in names(df))
    df.Solver = solver_name
end

if "N_$N" in names(df)
    col_index = findfirst(==("N_$N"), names(df))
    df[!, col_index] = sol_times
else
    df_temp = DataFrame(column_name => sol_times)
    df = hcat(df, df_temp)
end

CSV.write("benchmark/data/scalability_study.csv", df)




