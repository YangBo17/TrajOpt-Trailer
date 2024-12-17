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
λstar1 = Array(reshape(-λstar1, (d+1)*K*N));
sopt1 = calc_sopt(FG, Xstar1);
cstar1 = calc_cstar(Hopt, ropt, Lopt, gopt, sopt1);
νstar1 = calc_νstar(cstar1, λstar1, Hopt, Lopt);

Xstar2, λstar2, X_list2, λ_list2, sol_time2 = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter);
λstar2 = Array(reshape(-λstar2, (d+1)*K*N));
sopt2 = calc_sopt(FG, Xstar2);
cstar2 = calc_cstar(Hopt, ropt, Lopt, gopt, sopt2);
νstar2 = calc_νstar(cstar2, λstar2, Hopt, Lopt);

Xstar3, λstar3, X_list3, λ_list3, sol_time3 = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter);
λstar3 = Array(reshape(-λstar3, (d+1)*K*N));
sopt3 = calc_sopt(FG, Xstar3);
cstar3 = calc_cstar(Hopt, ropt, Lopt, gopt, sopt3);
νstar3 = calc_νstar(cstar3, λstar3, Hopt, Lopt);

kkt1_1 = 2Popt * cstar1 + Lopt' * λstar1 + Hopt' * νstar1
kkt1_2 = Lopt * cstar1 - sopt1 - gopt
kkt1_3 = Hopt * cstar1 - ropt
kkt2_1 = 2Popt * cstar2 + Lopt' * λstar2 + Hopt' * νstar2
kkt2_2 = Lopt * cstar2 - sopt2 - gopt
kkt2_3 = Hopt * cstar2 - ropt
kkt3_1 = 2Popt * cstar3 + Lopt' * λstar3 + Hopt' * νstar3
kkt3_2 = Lopt * cstar3 - sopt3 - gopt
kkt3_3 = Hopt * cstar3 - ropt

@show norm(kkt1_1)
@show norm(kkt1_2)
@show norm(kkt1_3)
@show norm(kkt2_1)
@show norm(kkt2_2)
@show norm(kkt2_3)
@show norm(kkt3_1)
@show norm(kkt3_2)
@show norm(kkt3_3)

##
@show norm(Xstar1)
@show norm(Xstar1 - Xstar2)
@show norm(Xstar2 - Xstar3)
@show norm(Xstar1 - Xstar3)
@show norm(sopt1)
@show norm(sopt1 - sopt2)
@show norm(sopt2 - sopt3)
@show norm(sopt1 - sopt3)
@show norm(cstar1)
@show norm(cstar1 - cstar2)
@show norm(cstar2 - cstar3)
@show norm(cstar1 - cstar3)
@show norm(λstar1)
@show norm(λstar1 - λstar2)
@show norm(λstar2 - λstar3)
@show norm(λstar1 - λstar3)
@show norm(νstar1)
@show norm(νstar1 - νstar2)
@show norm(νstar2 - νstar3)
@show norm(νstar1 - νstar3)

# calculate the optimal primal-dual pair by the KKT of the QP problem, where the X=Xstar is fixed
cstar_1, ϵstar_1, KKT_A_1, KKT_b_1, λstar_1, νstar_1 = calc_poly(Popt,Hopt,Lopt,ropt,gopt,sopt1);
cstar_2, ϵstar_2, KKT_A_2, KKT_b_2, λstar_2, νstar_2 = calc_poly(Popt,Hopt,Lopt,ropt,gopt,sopt2);
cstar_3, ϵstar_3, KKT_A_3, KKT_b_3, λstar_3, νstar_3 = calc_poly(Popt,Hopt,Lopt,ropt,gopt,sopt3);

@show norm(cstar1)
@show norm(cstar_1)
@show norm(cstar1 - cstar_1)
@show norm(cstar2 - cstar_2)
@show norm(cstar3 - cstar_3)
@show norm(cstar_1 - cstar_2)
@show norm(λstar1)
@show norm(λstar_1)
@show norm(λstar1 - λstar_1)
@show norm(λstar2 - λstar_2)
@show norm(λstar3 - λstar_3)
@show norm(λstar_1 - λstar_2)
@show norm(νstar1)
@show norm(νstar_1) 
@show norm(νstar1 - νstar_1)
@show norm(νstar2 - νstar_2)
@show norm(νstar3 - νstar_3)
@show norm(νstar_1 - νstar_2)

##

# calculate νstar by the "SDP method: calc_νstar(cstar,λstar,Popt,Hopt,Lopt)", but the λstar is replaced by the λstar_1
vstar_1_a = calc_νstar(cstar_1, λstar_1, Popt, Hopt, Lopt); 
@show norm(vstar_1_a - νstar1) 

@show 2Popt * cstar_1 + Lopt' * λstar_1 + Hopt' * νstar_1
@show norm(2Popt * cstar_1 + Lopt' * λstar_1 + Hopt' * νstar_1)

@show 2Popt * cstar1 + Lopt' * λstar1 + Hopt' * νstar1
@show norm(2Popt * cstar1 + Lopt' * λstar1 + Hopt' * νstar1)
##

max_iter = 500
data, X_list, λ_list, sol_time, obj0 = TrajOpt_space(T0, α, β, FG; device=:GPU, max_iter=max_iter); 
Xstar, λstar = collect(X_list[end]), collect(λ_list[end])
sopt2 = calc_sopt(FG, Xstar)
cstar, ϵstar, sopt, KKT_A, KKT_b = calc_poly(FG,Xstar,Popt,Hopt0,Lopt0,ropt,gopt,Ropt)
@show sol_time
@show obj0
@test sopt2 ≈ sopt

Ropt = vcat(Lopt0, Hopt0)
qopt = vcat(gopt + sopt, ropt)
qopt0 = [zeros(size(Popt,1)); qopt]
PLHopt = [2Popt Ropt'; Ropt zeros(size(Ropt,1),size(Ropt,1))]
cϵ = pinv(PLHopt) * qopt0
c1star = cϵ[1:size(Popt,1)]
ϵ1star = cϵ[size(Popt,1)+1:end]
norm(ϵ1star - ϵstar)
@test KKT_A ≈ PLHopt
@test c1star ≈ cstar
@test ϵ1star ≈ ϵstar 
@test 2Popt * c1star + Ropt' * ϵ1star ≈ zeros(size(Popt,1)) atol=1e-4
@test 2Popt * cstar + Ropt' * ϵ1star ≈ zeros(size(Popt,1)) atol=1e-4
norm(Ropt * c1star - qopt)
@test Ropt * c1star ≈ qopt atol=1e-1
c2star = pinv(Ropt) * qopt
@test Ropt * c2star ≈ qopt atol=1e-1

norm(c2star - cstar)
@test c2star ≈ c1star

Z_d1 = Matrix(0.0I, size(Lopt,1), size(Hopt,1))
I_d2 = Matrix(1.0I, size(Lopt,1), size(Lopt,1))
HLI = [Lopt' Hopt']
Pcλ = -2Popt*cstar 
d_12 = HLI \ Pcλ
d_1 = d_12[1:size(Lopt,1)]
d_2 = d_12[size(Lopt,1)+1:end]
ϵ_1 = ϵstar[1:size(Lopt,1)]
ϵ_2 = ϵstar[size(Lopt,1)+1:end]

norm(d_12 - ϵstar)
@test d_12 ≈ ϵstar atol=1e-1
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

