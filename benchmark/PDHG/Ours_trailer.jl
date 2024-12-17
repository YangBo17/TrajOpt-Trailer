include("../utils/env_trailer.jl")
include("get_PDHG_para_trailer.jl")
include("PDHG_solver_trailer.jl")

# T0 =  [2.8, 2.5, 3.0, 4.0, 2.0] * 1.0
# T0= [1.0692094580862169, 0.6199361137959327, 1.5135679542069184, 1.5535277522049545, 1.386439078321449]*1.2
# T0 = [4.87985731342541, 3.8193685289081785, 5.394489285060812, 2.949933196037537] * 0.6

α = 0.2; # PDHG primal step size
β = 0.4; # PDHG dual step size
η = 0.01; # Time update step size

max_iter = 5000
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
fig_opt1 = plot(fig_env, x0, y0, label="tractor")
plot!(fig_opt1, x1, y1, label="trailer")


##
using LinearAlgebra
FG = cu(FG)
Y = cu(zeros(size(X_list[end])))
λ_star = λ_list[end]
λ_large = cu(zeros(2*(d+1), 2*K*N))
λ_large[1:d+1, 1:K*N] = λstar
λ_large[d+2:end, K*N+1:end] = λstar
@tensor begin
    Y[a,b,j] = λ_large[i,j] * FG[a,b,i]
end
for i in 1:size(Y,3)
    eigen_value = eigen(Array(Y[:,:,i])).values
    @show eigen_value
    # is_neg = all(eigen_value .<= 0)
    # is_pos = all(eigen_value .>= 0)
    # @show is_neg
    # @show is_pos
end

##
traj = Traj_trailer(ts, x0, y0, ψ0, x1, y1, ψ1, v0, a0, δ0, v1)
using JLD2
using CSV, DataFrames
@save "benchmark/data/Ours2024_trailer_traj.jld2" traj

method_name = "Ours24"

df = CSV.read("benchmark/data/sol_time_trailer_std.csv", DataFrame)

row_index = findfirst(row -> row.Method == method_name, eachrow(df))
if row_index !== nothing
    # 如果存在，覆盖该行
    df[row_index, :SolTime] = sol_time
else
    # 如果不存在，添加新行
    new_row = DataFrame(Method = [method_name], SolTime = [sol_time])
    append!(df, new_row)
end

CSV.write("benchmark/data/sol_time_trailer_std.csv", df)


##
T0 = [1.9809074750501214, 1.5865341886569475, 2.225725561734048, 1.8843437627318562, 2.729539520783188] * 1.5
Tw = ones(5) * 10
cstar, Xstar, Tstar, c_list, obj_list, Lag_list, T_list, bilevel_soltime = BiLevelOpt(T0, Tw, α, β, FG) 
@show Tstar
@show T_opt = sum(Tstar)
@show T_list
@show bilevel_soltime

data = post_Bilevel(cstar, Xstar, Tstar)
x0 = [data[i].x0 for i in eachindex(data)]
y0 = [data[i].y0 for i in eachindex(data)]
x1 = [data[i].x1 for i in eachindex(data)]
y1 = [data[i].y1 for i in eachindex(data)]   
fig_opt2 = plot(fig_env, x0, y0, label="tractor")
plot!(fig_opt2, x1, y1, label="trailer")