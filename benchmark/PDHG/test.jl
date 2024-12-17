## ForwardDiff
using ForwardDiff

function matrix_function1(A)
    return A * A
end

function matrix_function2(A)
    return inv(A * A)
end

A = [1.0 2.0; 3.0 4.0]
J1 = ForwardDiff.jacobian(matrix_function1, A)
J2 = ForwardDiff.jacobian(matrix_function2, A)

println(J)

## Kronecker Product and Direct Sum
A = [1 2; 3 4]
B = [0 1; 1 0]

K = kron(A, B)

function direct_sum(A::Matrix, B::Matrix)
    m, n = size(A)
    p, q = size(B)
    return [A zeros(m, q); zeros(p, n) B]
end

function direct_sum(a::Vector, b::Vector)
    return [a; b]
end

##
using LinearAlgebra
using BenchmarkTools
using BlockArrays

# 创建一个 10000x10000 的随机矩阵
A = rand(10000, 10000)

# 使用 LU 分解求逆矩阵
function inv_lu(A)
    F = lu(A)
    return inv(F)
end

# 使用 QR 分解求逆矩阵
function inv_qr(A)
    Q, R = qr(A)
    return inv(R) * Q'
end

# 使用 SVD 分解求逆矩阵
function inv_svd(A)
    U, S, V = svd(A)
    return V * Diagonal(1.0 ./ S) * U'
end
function inv_block(A)
    return BlockArrays.inv(A)
end

# 测量时间
# println("LU 分解求逆时间：")
# @btime inv_lu(A);

# println("QR 分解求逆时间：")
# @btime inv_qr(A);

# println("SVD 分解求逆时间：")
# @btime inv_svd(A);

# println("Block 分解求逆时间：")
# @btime inv_block(A);

println("inv 分解求逆时间：")
@btime inv(A);


println("pinv 分解求逆时间：")
@btime pinv(A);

##
using JLD2
using Test
@load "Popt.jld2" Popt 
@load "Ropt.jld2" Ropt
@load "qopt.jld2" qopt

@time x1 = Ropt \ qopt

@time x2 = Ropt'*Ropt \ Ropt'*qopt

@test x1 ≈ x2

##
using LinearAlgebra
X= cat(dims=3, [let A = rand(3,3); A' * A end for _ in 1:100]...)

@show eigvals(X[:,:,1])

##
K = 107
N = 3

@show 2*K*N

##
using JuMP
using MosekTools
using COPT
using BenchmarkTools


function generate_multi_psd_sdp_problem(n, k)
    model = Model()
    
    # 创建多个 PSD 锥约束
    PSD_cones = []
    @variable(model, X[i=1:n, j=1:n, h=1:k])
    for i in 1:k
        @constraint(model, X[:, :,i] in PSDCone())
    end
    
    # 目标函数：使所有 PSD 锥的元素之和最小化
    @objective(model, Min, sum(sum(X[i, j, h] for i in 1:n, j in 1:n, h in 1:k) for X in PSD_cones))
    
    # 添加一些约束条件
    for X in PSD_cones
        @constraint(model, diag(X) .== 1)  # 对角线元素必须为1
    end
    
    return model
end

function benchmark_sdp_solvers(n, k)
    model = generate_multi_psd_sdp_problem(n, k)
    
    # Using Mosek
    set_optimizer(model, Mosek.Optimizer)
    mosek_time = @btime optimize!($model)

    # 重新生成问题以便于公平比较
    model = generate_multi_psd_sdp_problem(n, k)
    
    # Using COPT
    set_optimizer(model, COPT.Optimizer)
    copt_time = @btime optimize!($model)
    
    return mosek_time, copt_time
end

# Benchmarking for different problem sizes
sizes = [(10, 2), (20, 2), (30, 3), (40, 3), (50, 4)]
for (n, k) in sizes
    mosek_time, copt_time = benchmark_sdp_solvers(n, k)
    println("Problem size: $n x $n with $k PSD cones")
    println("Mosek time: $mosek_time seconds")
    println("COPT time: $copt_time seconds")
end

##
using CUDA
using BenchmarkTools
A = rand(5000, 5000);
scalar = 3.5;
A_d = cu(A);
@btime A_d = cu(A);
C_d = scalar * A_d;
@btime C_d = scalar * A_d;
@btime C = Array(C_d);
@btime C = collect(C_d);



