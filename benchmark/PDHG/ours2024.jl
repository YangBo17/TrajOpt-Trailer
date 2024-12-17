 using LinearAlgebra, NumericalIntegration
using DynamicPolynomials, Plots


include("get_system_matrix.jl")

Ac,Bc,Gflat,S,H=get_system_matrix()

global d = 5
n = 4 # dim x
m = 2 # dim u
Nineq = (m+n)*2
global d2=Int((d-1)/2+1)

global T = 10
global N = 10


x_min = zeros(n, d+1)
x_max = zeros(n, d+1)
x_max[:,end] .= 20

u_min = [zeros(1,d) 0; zeros(1,d) 0] 
u_max = [zeros(1,d) 8; zeros(1,d) 8]

x0=[1; 15; 15; 1]


xrefs=zeros(d+1,n,N)
for l in 1:N
    xrefs[end,1,l]=19.9
    xrefs[end,2,l]=19.9
    xrefs[end,3,l]=2.4
    xrefs[end,4,l]=2.4
end

p_step_size=0.2
d_step_size=0.4


include("get_PDHG_para.jl")

include("PDHG_GPU_solver.jl")

x=zeros(n)
ut_coef=zeros(m,d+1,N)
xt_coef=zeros(n,d+1,N)


F,G=calc_FG(T/N,d);

##
xt_coef,ut_coef,last_Xf,last_Xg,last_dual = PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ, F,G, 200, false);


# xt_coef,ut_coef,last_Xf,last_Xg,last_dual=PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ, F,G, 10000, false);

# xt_coef,ut_coef,last_Xf,last_Xg,last_dual=PDHG_solver(N,d,m,n,p_step_size,d_step_size,FF,μμ, F,G, 200, true, last_Xf,last_Xg,last_dual);

##
@polyvar t
Δt = 1e-5
n_points = Int(round(T/N/Δt))
tc=collect(0:Δt:T-Δt)
uc=zeros(m,n_points*N)
xc=zeros(n,n_points*N)
for i in 1:n 
    for j in 1:N
        px = dot(monomials([t], 0:d), xt_coef[i,:,j])
        t_list = 0:1/n_points:1-1/n_points
        for kk in 1:n_points
            xc[i,(j-1)*n_points+kk] = px(t=>t_list[kk])
        end
    end
end
for i in 1:m 
    for j in 1:N
        pu = dot(monomials([t], 0:d), ut_coef[i,:,j])
        t_list = 0:1/n_points:1-1/n_points
        for kk in 1:n_points
            uc[i,(j-1)*n_points+kk] = pu(t=>t_list[kk])
        end
    end
end
#
@show maximum(xc[1,:])
plot(tc[1:100:end], xc[1,1:100:end])
# plot(tc[1:100:end], uc[:,1:100:end]')

##
plot(tc[3400001:3500000], xc[1,3400001:3500000])
hline!([20.0])

##
using JLD2
data = Dict("tk" => tk, "xk" => xk, "uk" => uk, "tc" => tc, "xc" => xc, "uc" => uc)
@save "data_ours2024.jld2" data
