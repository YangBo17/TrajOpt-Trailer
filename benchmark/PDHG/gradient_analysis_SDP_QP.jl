using ProgressMeter
using Test
using Random
Random.seed!(1234)
N = 2
include("bilevel_env_trailer.jl")
include("get_PDHG_para_trailer.jl")
include("PDHG_solver_trailer.jl")
include("SOS_trailer.jl")
Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T0);
@show 2*(L1+L2)*seg_N
@show Mat_size = size(Popt,1) + size(Lopt, 1) + size(Hopt, 1)
α = 0.2; # PDHG primal step size
β = 0.4; # PDHG dual step size
η = 0.10; # Time update step size
@show Base.gc_bytes()
dim1_P, dim2_P = size(Popt)
dim1_R, dim2_R = size(Ropt)
@show dim1_P
@show dim1_R
max_iter = 5000

# calclate the optimal primal-dual pair by the KKT of the SDP problem: λstar is obtained from the PDHG solver
Xstar1, λstar1, X_list1, λ_list1, sol_time1 = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter);
λstar1 = calc_λstar(λstar1);
sopt1 = calc_sopt(FG, Xstar1);
cstar1 = calc_cstar(Hopt, ropt, Lopt, gopt, sopt1);
νstar1 = calc_νstar(cstar1, λstar1, Popt, Hopt, Lopt);
kkt_sdp_01 = 2Popt * cstar1 + Lopt' * λstar1 + Hopt' * νstar1
kkt_sdp_02 = Lopt * cstar1 - sopt1 - gopt
kkt_sdp_03 = Hopt * cstar1 - ropt
@show norm(kkt_sdp_01)
@show norm(kkt_sdp_02)
@show norm(kkt_sdp_03)

##
# calculate the optimal primal-dual pair by the KKT of the QP problem, where the X=Xstar is fixed
sopt_1 = sopt1
cstar_1, ϵstar_1, KKT_A_1, KKT_b_1, λstar_1, νstar_1 = calc_poly(Popt,Hopt,Lopt,ropt,gopt,sopt_1);
kkt_qp_01 = 2Popt * cstar_1 + Lopt' * λstar_1 + Hopt' * νstar_1
kkt_qp_02 = Lopt * cstar_1 - sopt_1 - gopt
kkt_qp_03 = Hopt * cstar_1 - ropt
@show norm(kkt_qp_01)
@show norm(kkt_qp_02)
@show norm(kkt_qp_03)




##

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

