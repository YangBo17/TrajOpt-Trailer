using ProgressMeter
using Test
using Printf

N = 2
include("bilevel_env_trailer.jl")
include("get_PDHG_para_trailer.jl")
include("PDHG_solver_trailer.jl")
include("SOS_trailer.jl")
T0 = 2.6 * ones(N)
max_iter = 3000

Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T0);
Popt0 = Popt
Lopt0 = Lopt
Hopt0 = Hopt
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

data, X_list, λ_list, sol_time, obj, Lag, Lag_grad, cstar, λstar, νstar = TrajOpt_space_new(T0, α, β, FG; device=:GPU, max_iter=max_iter); 
Xstar = X_list[end];
sopt = calc_sopt(FG, Xstar);
obj0 = obj
Lag0 = Lag
cstar0 = cstar
λstar0 = λstar
νstar0 = νstar
@printf("Obj = %.3e\n", obj)
@printf("Lag = %.3e\n", Lag)
@show Lag_grad

@show cstar' * Popt * cstar
@show cstar' * Popt * cstar + λstar' * (Lopt * cstar - gopt - sopt) + νstar' * (Hopt * cstar - ropt)
@show λstar' * (Lopt * cstar - gopt - sopt) 
@show νstar' * (Hopt * cstar - ropt)

kkt_01 = 2Popt * cstar + Lopt' * λstar + Hopt' * νstar
kkt_02 = Lopt * cstar - sopt - gopt
kkt_03 = Hopt * cstar - ropt

@printf("Obj = %.3e\n", obj)
@printf("Lag = %.3e\n", Lag)
@printf("kkt01 = %.3e\n", norm(kkt_01))
@printf("kkt02 = %.3e\n", norm(kkt_02))
@printf("kkt03 = %.3e\n", norm(kkt_03))
@printf("c_norm = %.3e\n", norm(cstar))
@printf("λ_norm = %.3e\n", norm(λstar))
@printf("ν_norm = %.3e\n", norm(νstar))

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
T1 = T0 + 0.01 * ones(N)
Obj_list = []
Lag_list = []
lag_λ_list = []
lag_ν_list = []
norm_cstar_list = []
norm_λstar_list = []
norm_νstar_list = []
norm_Δcstar_list = []
norm_Δλstar_list = []
norm_Δνstar_list = []
Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T1)
for i in 1:20
    @show i
    data, X_list, λ_list, sol_time, obj, Lag, Lag_grad, cstar, λstar, νstar = TrajOpt_space_new(T1, α, β, FG; device=:GPU, max_iter=max_iter); 
    norm_cstar = norm(cstar)
    norm_λstar = norm(λstar)
    norm_νstar = norm(νstar)
    norm_Δcstar = norm(cstar - cstar0)
    norm_Δλstar = norm(λstar - λstar0)
    norm_Δνstar = norm(νstar - νstar0)
    lag_λ = λstar' * (Lopt * cstar - gopt - sopt)
    lag_ν = νstar' * (Hopt * cstar - ropt)
    push!(Obj_list, obj)
    push!(Lag_list, Lag)
    push!(lag_λ_list, lag_λ)
    push!(lag_ν_list, lag_ν)
    push!(norm_cstar_list, norm_cstar)
    push!(norm_λstar_list, norm_λstar)
    push!(norm_νstar_list, norm_νstar)
    push!(norm_Δcstar_list, norm_Δcstar)
    push!(norm_Δλstar_list, norm_Δλstar)
    push!(norm_Δνstar_list, norm_Δνstar)
end

fig_obj = plot(Obj_list, label="Obj")
plot!(fig_obj, Lag_list, label="Lag")
fig_lag_λν = plot(lag_λ_list, label="lag_λ")
plot!(fig_lag_λν, lag_ν_list, label="lag_ν")

fig_cλν = plot(norm_cstar_list, label="cstar")
plot!(fig_cλν, norm_λstar_list, label="λstar")
plot!(fig_cλν, norm_νstar_list, label="νstar")

fig_Δcλν = plot(norm_Δcstar_list, label="Δcstar")
plot!(fig_Δcλν, norm_Δλstar_list, label="Δλstar")
plot!(fig_Δcλν, norm_Δνstar_list, label="Δνstar")

fig_Δcλν_cλν = plot([norm_Δcstar_list ./ norm(cstar0)], label="Δcstar / cstar")
plot!(fig_Δcλν_cλν, [norm_Δλstar_list ./ norm(λstar0)], label="Δλstar / λstar")
plot!(fig_Δcλν_cλν, [norm_Δνstar_list ./ norm(νstar0)], label="Δνstar / νstar")

@show fig_cλν
@show fig_Δcλν
@show fig_Δcλν_cλν

plot(fig_Δcλν, fig_Δcλν_cλν, layout=(2,1))

##
δ = 0.5

dz_dP_AG = zeros(size(Popt))
for i in 1:size(Popt,1), j in 1:size(Popt,2)
    dz_dP_AG[i,j] = cstar[i] * cstar[j]
end

Len1 = 20
Len2 = 20

progress = Progress(length(eachindex(cstar[1:Len1])) * length(eachindex(cstar[1:Len2])))

Obj_list = []
Lag_list = []
lag_λ_list = []
lag_ν_list = []
norm_cstar_list = []
norm_λstar_list = []
norm_νstar_list = []
norm_Δcstar_list = []
norm_Δλstar_list = []
norm_Δνstar_list = []

dz_dP_FD = zeros(size(Popt))
for i in eachindex(cstar[1:Len1]), j in eachindex(cstar[1:Len2])
    dP = zeros(size(Popt))
    dP[i,j] = δ
    data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar = TrajOpt_space_dP(T0, α, β, FG; device=:GPU, max_iter=max_iter, dP=dP);
    # dz_dP_FD[i,j] = (obj - obj0) / δ
    dz_dP_FD[i,j] = (Lag - Lag0) / δ

    norm_cstar = norm(cstar)
    norm_λstar = norm(λstar)
    norm_νstar = norm(νstar)
    norm_Δcstar = norm(cstar - cstar0)
    norm_Δλstar = norm(λstar - λstar0)
    norm_Δνstar = norm(νstar - νstar0)
    lag_λ = λstar' * (Lopt * cstar - gopt - sopt)
    lag_ν = νstar' * (Hopt * cstar - ropt)
    push!(Obj_list, obj)
    push!(Lag_list, Lag)
    push!(lag_λ_list, lag_λ)
    push!(lag_ν_list, lag_ν)

    @show norm(norm_Δcstar)
    @show norm(norm_Δλstar)
    @show norm(norm_Δνstar)
    push!(norm_cstar_list, norm_cstar)
    push!(norm_λstar_list, norm_λstar)
    push!(norm_νstar_list, norm_νstar)
    push!(norm_Δcstar_list, norm_Δcstar)
    push!(norm_Δλstar_list, norm_Δλstar)
    push!(norm_Δνstar_list, norm_Δνstar)

    GC.gc()
    next!(progress)
end
finish!(progress)

println("dz_dP_AG[1:10,1:10]")
dz_dP_AG[1:10,1:10]

println("dz_dP_FD[1:10,1:10]")
dz_dP_FD[1:10,1:10]

println("dz_dP_AG[1:10,1:10] ./ dz_dP_FD[1:10,1:10]")
dz_dP_AG[1:10,1:10] ./ dz_dP_FD[1:10,1:10]

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

fig_obj = plot(Obj_list, label="Obj")
plot!(fig_obj, Lag_list, label="Lag")
fig_lag_λν = plot(lag_λ_list, label="lag_λ")
plot!(fig_lag_λν, lag_ν_list, label="lag_ν")

fig_cλν = plot(norm_cstar_list, label="cstar")
plot!(fig_cλν, norm_λstar_list, label="λstar")
plot!(fig_cλν, norm_νstar_list, label="νstar")

fig_Δcλν = plot(norm_Δcstar_list, label="Δcstar")
plot!(fig_Δcλν, norm_Δλstar_list, label="Δλstar")
plot!(fig_Δcλν, norm_Δνstar_list, label="Δνstar")

fig_Δcλν_cλν = plot([norm_Δcstar_list ./ norm(cstar0)], label="Δcstar / cstar")
plot!(fig_Δcλν_cλν, [norm_Δλstar_list ./ norm(λstar0)], label="Δλstar / λstar")
plot!(fig_Δcλν_cλν, [norm_Δνstar_list ./ norm(νstar0)], label="Δνstar / νstar")

@show fig_cλν
@show fig_Δcλν
@show fig_Δcλν_cλν

plot(fig_Δcλν, fig_Δcλν_cλν, layout=(2,1))

##
dz_dL_AG = zeros(size(Lopt))
for i in 1:size(Lopt,1), j in 1:size(Lopt,2)
    dz_dL_AG[i,j] = λstar[i] * cstar[j]
end

Len1 = 20
Len2 = 20

progress = Progress(length(eachindex(λstar[1:Len1])) * length(eachindex(cstar[1:Len2])))

Obj_list = []
Lag_list = []
lag_λ_list = []
lag_ν_list = []
norm_cstar_list = []
norm_λstar_list = []
norm_νstar_list = []
norm_Δcstar_list = []
norm_Δλstar_list = []
norm_Δνstar_list = []

dz_dL_FD = zeros(size(Lopt))
for i in eachindex(λstar[1:Len1]), j in eachindex(cstar[1:Len2])
    dL = zeros(size(Lopt))
    dL[i,j] = δ
    data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar = TrajOpt_space_dL(T0, α, β, FG; device=:GPU, max_iter=max_iter, dL=dL);
    dz_dL_FD[i,j] = (obj - obj0) / δ
    # dz_dL_FD[i,j] = (Lag - Lag0) / δ

    norm_cstar = norm(cstar)
    norm_λstar = norm(λstar)
    norm_νstar = norm(νstar)
    norm_Δcstar = norm(cstar - cstar0)
    norm_Δλstar = norm(λstar - λstar0)
    norm_Δνstar = norm(νstar - νstar0)
    lag_λ = λstar' * (Lopt * cstar - gopt - sopt)
    lag_ν = νstar' * (Hopt * cstar - ropt)
    push!(Obj_list, obj)
    push!(Lag_list, Lag)
    push!(lag_λ_list, lag_λ)
    push!(lag_ν_list, lag_ν)

    @show norm(norm_Δcstar)
    @show norm(norm_Δλstar)
    @show norm(norm_Δνstar)
    push!(norm_cstar_list, norm_cstar)
    push!(norm_λstar_list, norm_λstar)
    push!(norm_νstar_list, norm_νstar)
    push!(norm_Δcstar_list, norm_Δcstar)
    push!(norm_Δλstar_list, norm_Δλstar)
    push!(norm_Δνstar_list, norm_Δνstar)

    GC.gc()
    next!(progress)
end
finish!(progress)

println("dz_dL_AG[1:10,1:10]")
dz_dL_AG[1:10,1:10]

println("dz_dL_FD[1:10,1:10]")
dz_dL_FD[1:10,1:10]

println("dz_dL_AG[1:10,1:10] ./ dz_dL_FD[1:10,1:10]")
dz_dL_AG[1:10,1:10] ./ dz_dL_FD[1:10,1:10]

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

fig_obj = plot(Obj_list, label="Obj")
plot!(fig_obj, Lag_list, label="Lag")
fig_lag_λν = plot(lag_λ_list, label="lag_λ")
plot!(fig_lag_λν, lag_ν_list, label="lag_ν")

fig_cλν = plot(norm_cstar_list, label="cstar")
plot!(fig_cλν, norm_λstar_list, label="λstar")
plot!(fig_cλν, norm_νstar_list, label="νstar")

fig_Δcλν = plot(norm_Δcstar_list, label="Δcstar")
plot!(fig_Δcλν, norm_Δλstar_list, label="Δλstar")
plot!(fig_Δcλν, norm_Δνstar_list, label="Δνstar")

fig_Δcλν_cλν = plot([norm_Δcstar_list ./ norm(cstar0)], label="Δcstar / cstar")
plot!(fig_Δcλν_cλν, [norm_Δλstar_list ./ norm(λstar0)], label="Δλstar / λstar")
plot!(fig_Δcλν_cλν, [norm_Δνstar_list ./ norm(νstar0)], label="Δνstar / νstar")

@show fig_cλν
@show fig_Δcλν
@show fig_Δcλν_cλν

plot(fig_Δcλν, fig_Δcλν_cλν, layout=(2,1))



##
dz_dH_AG = zeros(size(Hopt))
for i in 1:size(Hopt,1), j in 1:size(Hopt,2)
    dz_dH_AG[i,j] = νstar[i] * cstar[j]
end

Len1 = 18
Len2 = 20
progress = Progress(length(eachindex(νstar[1:Len1])) * length(eachindex(cstar[1:Len2])))

Obj_list = []
Lag_list = []
lag_λ_list = []
lag_ν_list = []
norm_cstar_list = []
norm_λstar_list = []
norm_νstar_list = []
norm_Δcstar_list = []
norm_Δλstar_list = []
norm_Δνstar_list = []

dz_dH_FD = zeros(size(Hopt))
for i in eachindex(λstar[1:Len1]), j in eachindex(cstar[1:Len2])
    dH = zeros(size(Hopt))
    dH[i,j] = δ
    data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar = TrajOpt_space_dH(T0, α, β, FG; device=:GPU, max_iter=max_iter, dH=dH);
    dz_dH_FD[i,j] = (obj - obj0) / δ
    # dz_dH_FD[i,j] = (Lag - Lag0) / δ

    norm_cstar = norm(cstar)
    norm_λstar = norm(λstar)
    norm_νstar = norm(νstar)
    norm_Δcstar = norm(cstar - cstar0)
    norm_Δλstar = norm(λstar - λstar0)
    norm_Δνstar = norm(νstar - νstar0)
    lag_λ = λstar' * (Lopt * cstar - gopt - sopt)
    lag_ν = νstar' * (Hopt * cstar - ropt)
    push!(Obj_list, obj)
    push!(Lag_list, Lag)
    push!(lag_λ_list, lag_λ)
    push!(lag_ν_list, lag_ν)

    @show norm(norm_Δcstar)
    @show norm(norm_Δλstar)
    @show norm(norm_Δνstar)
    push!(norm_cstar_list, norm_cstar)
    push!(norm_λstar_list, norm_λstar)
    push!(norm_νstar_list, norm_νstar)
    push!(norm_Δcstar_list, norm_Δcstar)
    push!(norm_Δλstar_list, norm_Δλstar)
    push!(norm_Δνstar_list, norm_Δνstar)

    GC.gc()
    next!(progress)
end

println("dz_dH_AG[1:10,1:10]")
dz_dH_AG[1:10,1:10]

println("dz_dH_FD[1:10,1:10]")
dz_dH_FD[1:10,1:10]

println("dz_dH_AG[1:10,1:10] ./ dz_dH_FD[1:10,1:10]")
dz_dH_AG[1:10,1:10] ./ dz_dH_FD[1:10,1:10]

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

fig_obj = plot(Obj_list[1:end], label="Obj")
plot!(fig_obj, Lag_list[1:end], label="Lag")
fig_lag_λν = plot(lag_λ_list, label="lag_λ")
plot!(fig_lag_λν, lag_ν_list, label="lag_ν")

fig_cλν = plot(norm_cstar_list, label="cstar")
plot!(fig_cλν, norm_λstar_list, label="λstar")
plot!(fig_cλν, norm_νstar_list, label="νstar")

fig_Δcλν = plot(norm_Δcstar_list, label="Δcstar")
plot!(fig_Δcλν, norm_Δλstar_list, label="Δλstar")
plot!(fig_Δcλν, norm_Δνstar_list, label="Δνstar")

fig_Δcλν_cλν = plot([norm_Δcstar_list ./ norm(cstar0)], label="Δcstar / cstar")
plot!(fig_Δcλν_cλν, [norm_Δλstar_list ./ norm(λstar0)], label="Δλstar / λstar")
plot!(fig_Δcλν_cλν, [norm_Δνstar_list ./ norm(νstar0)], label="Δνstar / νstar")

@show fig_cλν
@show fig_Δcλν
@show fig_Δcλν_cλν

plot(fig_Δcλν, fig_Δcλν_cλν, layout=(2,1))



 