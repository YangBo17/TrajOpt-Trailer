using JuMP, LinearAlgebra
using COPT, Mosek, MosekTools 
using SCS, COSMO, ProxSDP, CSDP, SDPA
using BenchmarkTools
# Number of vertices
n = 100
# Graph weights
W = randn(n, n)
# Build Max-Cut SDP relaxation via JuMP
model = Model(COPT.ConeOptimizer)
# model = Model(Mosek.Optimizer)
# model = Model(COSMO.Optimizer)
# model = Model(SCS.Optimizer)
# model = Model(SDPA.Optimizer)
# model = Model(CSDP.Optimizer)
# model = Model(ProxSDP.Optimizer)
# set_silent(model)
@variable(model, X[1:n, 1:n], PSD)
@objective(model, Max, 0.25 * LinearAlgebra.dot(W, X))
@constraint(model, LinearAlgebra.diag(X) .== 1)
# Solve optimization problem 
@time optimize!(model)
@show JuMP.termination_status(model) 
Xsol = value.(X);

#
function gen_scale_sdp(N,K)
    model = Model()
    @variable(model, X[1:N, 1:N, 1:K])
    @constraint(model, [k = 1:K], X[:, :, k] in PSDCone())
    @constraint(model, [k = 1:K], diag(X[:, :, k]) .== 1)
    @objective(model, Min, sum(1/4*dot(W[:, :, k], X[:, :, k]) for k in 1:K))
    return model
end

function gen_scale_sdp2(N,K)
    model = Model() 
    X = [ @variable(model, [i=1:N, j=1:N], PSD) for k in 1:K ]
    for k in 1:K
        @constraint(model, [i = 1:N], X[k][i, i] == 1)
    end
    @objective(model, Min, sum(1/4 * dot(W[:, :, k], X[k]) for k in 1:K))
    return model
end


############################################ Benchmark Constraints type
N = 10
K = 100
W = randn(N, N, K)
model = gen_scale_sdp(N, K)
model2 = gen_scale_sdp2(N, K)

##
println("N = ", N, ", K = ", K)
println("COPT-affine-conic-cons")
set_optimizer(model, COPT.ConeOptimizer)
optimize!(model)
solve_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
JuMP.termination_status(model)

##
println("N = ", N, ", K = ", K)
println("COPT-matrix-variable")
set_optimizer(model2, COPT.ConeOptimizer)
optimize!(model2)
solve_time = round(MOI.get(model2, MOI.SolveTimeSec()), digits=3)
JuMP.termination_status(model2)

##
println("Mosek-affine-conic-cons")
set_optimizer(model, Mosek.Optimizer)
optimize!(model)
solve_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
JuMP.termination_status(model)

##
println("Mosek-matrix-variable")
set_optimizer(model2, Mosek.Optimizer)
optimize!(model2)
solve_time = round(MOI.get(model2, MOI.SolveTimeSec()), digits=3)
JuMP.termination_status(model2)

##
println("COSMO-affine-conic-cons")
set_optimizer(model, COSMO.Optimizer)
optimize!(model)
solve_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
JuMP.termination_status(model)

##
println("COSMO-matrix-variable")
set_optimizer(model2, COSMO.Optimizer)
optimize!(model2)
solve_time = round(MOI.get(model2, MOI.SolveTimeSec()), digits=3)
JuMP.termination_status(model2)

##
println("SCS-affine-conic-cons")
set_optimizer(model, SCS.Optimizer)
optimize!(model)
solve_time = round(MOI.get(model, MOI.SolveTimeSec()), digits=3)
JuMP.termination_status(model)

##
println("SCS-matrix-variable")
set_optimizer(model2, SCS.Optimizer)
optimize!(model2)
solve_time = round(MOI.get(model2, MOI.SolveTimeSec()), digits=3)
JuMP.termination_status(model2)

##
N = 100
K = 1
model = gen_scale_sdp(N, K)
println("COSMO")
set_optimizer(model, COSMO.Optimizer)
optimize!(model)
JuMP.termination_status(model)

##
N = 100
K = 1
model = gen_scale_sdp(N, K)
println("SCS")
set_optimizer(model, SCS.Optimizer)
optimize!(model)
JuMP.termination_status(model)

######################################

##
N = 100
K = 1
model = gen_scale_sdp(N, K)
println("SDPA")
set_optimizer(model, SDPA.Optimizer)
optimize!(model)
JuMP.termination_status(model)


##
N = 100
K = 1
model = gen_scale_sdp(N, K)
println("CSDP")
set_optimizer(model, CSDP.Optimizer)
optimize!(model)
JuMP.termination_status(model)

##
N = 100
K = 1
model = gen_scale_sdp(N, K)
println("ProxSDP")
set_optimizer(model, ProxSDP.Optimizer)
optimize!(model)
JuMP.termination_status(model)


##
N = 10
K = 100
W = randn(N, N, K)
model = gen_scale_sdp(N, K)
model2 = gen_scale_sdp2(N, K)
# set_silent(model)
# set_silent(model2)
##
println("N = ", N, ", K = ", K)
println("COPT")
set_optimizer(model, COPT.ConeOptimizer)
@btime optimize!($model)
@show JuMP.termination_status(model) 

println("Mosek")
set_optimizer(model, Mosek.Optimizer)
@btime optimize!($model)
@show JuMP.termination_status(model) 

println("COSMO")
set_optimizer(model, COSMO.Optimizer)
@btime optimize!($model)
@show JuMP.termination_status(model) 

println("SCS")
set_optimizer(model, SCS.Optimizer)
@btime optimize!($model)
@show JuMP.termination_status(model) 

println("SDPA")
set_optimizer(model, SDPA.Optimizer)
@btime optimize!($model)
@show JuMP.termination_status(model) 

println("CSDP")
set_optimizer(model, CSDP.Optimizer)
@btime optimize!($model)
@show JuMP.termination_status(model) 

println("ProxSDP")
set_optimizer(model, ProxSDP.Optimizer)
@btime optimize!($model)
@show JuMP.termination_status(model)


##
println("N = ", N, ", K = ", K)
println("COPT")
set_optimizer(model2, COPT.ConeOptimizer)
@btime optimize!($model2)
@show JuMP.termination_status(model2) 

println("Mosek")
set_optimizer(model2, Mosek.Optimizer)
@btime optimize!($model2)
@show JuMP.termination_status(model2) 

println("COSMO")
set_optimizer(model2, COSMO.Optimizer)
@btime optimize!($model2)
@show JuMP.termination_status(model2) 

println("SCS")
set_optimizer(model2, SCS.Optimizer)
@btime optimize!($model2)
@show JuMP.termination_status(model2) 

println("SDPA")
set_optimizer(model2, SDPA.Optimizer)
@btime optimize!($model2)
@show JuMP.termination_status(model2) 

println("CSDP")
set_optimizer(model2, CSDP.Optimizer)
@btime optimize!($model2)
@show JuMP.termination_status(model2) 

println("ProxSDP")
set_optimizer(model2, ProxSDP.Optimizer)
@btime optimize!($model2)
@show JuMP.termination_status(model2)