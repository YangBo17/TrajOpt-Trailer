using ProgressMeter
using Test
N = 2
include("bilevel_env_trailer.jl")
include("get_PDHG_para_trailer.jl")
include("PDHG_solver_trailer.jl")
include("SOS_trailer.jl")
Eopt, eopt, Popt, Hopt0, ropt, Lopt0, gopt, Ropt = calc_Para(T0);
@show 2*(L1+L2)*seg_N
@show Mat_size = size(Popt,1) + size(Lopt, 1) + size(Hopt, 1)
α = 0.2; # PDHG primal step size
β = 0.4; # PDHG dual step size
η = 0.10; # Time update step size
@show Base.gc_bytes()
max_iter = 500
data, X_list, λ_list, sol_time, obj0 = TrajOpt_space(T0, α, β, FG; device=:GPU, max_iter=max_iter); 
Xstar, λstar = collect(X_list[end]), collect(λ_list[end])
cstar0, ϵstar0, sopt0, KKT_A0, KKT_b0 = calc_poly(FG,Xstar,Popt,Hopt0,Lopt0,ropt,gopt,Ropt)
obj00 = cstar0' * Popt * cstar0
sopt = calc_sopt(Xstar, FG)

# All QP Parameters: P,L,H,g,s,r; R, q
Ropt = vcat(Lopt, Hopt)
qopt = vcat(gopt+sopt, ropt)

dim1_P, dim2_P = size(Popt)
dim1_R, dim2_R = size(Ropt)

Z_A = zeros(dim1_R, dim1_R)
Z_b = zeros(dim1_P)
KKT_A = [2Popt Ropt'; Ropt Z_A]
KKT_b = [Z_b; qopt]
opt_sol = pinv(KKT_A) * KKT_b
cstar = opt_sol[1:dim1_P]
ϵstar = opt_sol[dim1_P+1:end]

@test cstar ≈ cstar0 
@test ϵstar ≈ ϵstar0
@test KKT_A ≈ KKT_A0
@test KKT_b ≈ KKT_b0

cstar_z = SetZeros!(cstar)
ϵstar_z = SetZeros!(ϵstar)



