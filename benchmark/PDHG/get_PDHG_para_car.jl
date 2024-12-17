using BlockDiagonals
using BlockArrays
using ForwardDiff

# load system and define parameters
include("get_system_matrix_car.jl")
Ac,Bc,G,S,H=get_system_matrix()

d = 7
n = 4
m = 2
s = sdp_s
N = seg_N
if s == 2
    W = Matrix(Diagonal([0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
elseif s == 3
    W = Matrix(Diagonal([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]))
end


# T = ones(N) * 3.0
# define tool functions
function direct_sum(A::Matrix, B::Matrix)
    m, n = size(A)
    p, q = size(B)
    return [A zeros(m, q); zeros(p, n) B]
end

function direct_sum_mat(As::Vector)
    len = length(As)
    m, n = size(As[1])
    As_dsum = As[1]
    for i in 2:len
        As_dsum = direct_sum(As_dsum, As[i])
    end
    return As_dsum
end

function direct_sum(a::Vector, b::Vector)
    return [a; b]
end

function direct_sum_vec(as::Vector)
    return vcat(as...)
end

# define tool matrix
Diff = zeros(d+1,d+1)
for i in 1:d 
    Diff[i+1,i] = d+1-i
end
Diffs = Matrix(1.0I, d+1, d+1)
Diffs_list = [Diffs]
Diffs1_list = [Diffs]
for i in 1:s
    push!(Diffs_list, (Diff^i)')
end
for i in 1:s-1
    push!(Diffs1_list, (Diff^i)')
end
Diffs = hcat(Diffs_list...)'
Diffs1 = hcat(Diffs1_list...)'
Πs = BlockDiagonal([Diffs, Diffs])
Πs1 = BlockDiagonal([Diffs1, Diffs1])

function γ_fun(τ, d)
    γ_vec = zeros(typeof(τ), d+1)
    for i in 1:d+1
        γ_vec[i] = τ^(d+1-i)
    end
    return γ_vec
end

function Γ_fun(T, d)
    Γ_vec = zeros(typeof(T), d+1)
    for i in 1:d+1
        Γ_vec[i] = T^(d+1-i)
    end
    Γ_mat = Matrix(Diagonal(Γ_vec))
    return Γ_mat
end

# Objective Function #
Popt_int = zeros(d+1, d+1)
for ii in d:-1:0
    for jj in d:-1:0
        Popt_int[d+1-ii,d+1-jj]=(1)^(ii+jj+1)/(ii+jj+1)
    end
end

function Popt_fun(T) 
    Popt_vec = []
    for i in 1:N
        temp = Πs' * kron(T[i]*W, Γ_fun(T[i],d)*Popt_int*Γ_fun(T[i],d)) * Πs
        push!(Popt_vec, temp)
    end
    Popt_mat = BlockDiagonal([Popt_vec...])
    return Popt_mat
end
# Popt = Popt_fun(T)
# Popt_grad = ForwardDiff.jacobian(Popt_fun, T)
# Popt_grad2 = reshape(Popt_grad, (size(Popt)...,N))

# Equality Constraints #
function hopt(τ, Tl)
    return kron(Matrix(1.0I, 2s, 2s), γ_fun(τ,d)'*Γ_fun(Tl,d)) * Πs1
end

function Hopt_fun(T)
    h0 = hopt(0.0, 1.0)
    Hopt_block = BlockArray{eltype(T),2}(undef, fill(size(h0,1),N+1), fill(size(h0,2),N))
    fill!(Hopt_block, 0.0)
    for i in 1:N+1
        if i == 1
            setblock!(Hopt_block, hopt(0.0, T[i]), i, i)
        elseif i == N+1
            setblock!(Hopt_block, hopt(1.0, T[i-1]), i, i-1)
        else
            setblock!(Hopt_block, hopt(0.0, T[i]), i, i)
            setblock!(Hopt_block, -hopt(1.0, T[i-1]), i, i-1)
        end
    end
    Hopt = Matrix(Hopt_block)
    return Hopt
end
# Hopt = Hopt_fun(T)
# Hopt_grad = ForwardDiff.jacobian(Hopt_fun, T)
# Hopt_grad2 = reshape(Hopt_grad, (size(Hopt)...,N))

function ropt_fun()
    init = cons_init
    ropt = []
    for i in 1:N+1
        if i == 1
            if s == 2
                temp = [start[1], init.vel[1,1], start[2], init.vel[1,2]]
                push!(ropt, temp)
            elseif s == 3
                temp = [start[1], init.vel[1,1], 0.65, start[2], init.vel[1,2], 0.65]
                push!(ropt, temp)
            end
        elseif i == N+1
            if s == 2
                temp = [goal[1], init.vel[2,1], goal[2], init.vel[2,2]]
                push!(ropt, temp)
            elseif s == 3
                temp = [goal[1], init.vel[2,1], -0.25, goal[2], init.vel[2,2], -0.25]
                push!(ropt, temp)
            end
        else
            push!(ropt, zeros(2s))
        end
    end
    ropt = vcat(ropt...)
    return ropt
end
# ropt = ropt_fun()

# Inequality Constraints #
function process_polyhedra_car(limits_list, obs_limits)
    limits_list_new = []
    obs_list_new = []
    A_cons = []
    b_cons = []
    for i in 1:N
        A_limits, B_limits = limits_list[i].A, limits_list[i].B
        C_limits, D_limits = limits_list[i].C, limits_list[i].D
        A_zero_limits = zeros(size(A_limits,1), 1)
        if s == 2
            A_new_limits = hcat(A_zero_limits, A_limits[:,1], A_limits[:,3], A_zero_limits, A_limits[:,2], A_limits[:,4])
        elseif s == 3
            A_new_limits = hcat(A_zero_limits, A_limits[:,1], A_limits[:,3], A_zero_limits, A_zero_limits, A_limits[:,2], A_limits[:,4], A_zero_limits)
        end
        push!(limits_list_new, Cons_Corrs(A_new_limits, B_limits, C_limits, D_limits))
        A_obs, B_obs = obs_limits[i].A, obs_limits[i].B
        C_obs, D_obs = obs_limits[i].C, obs_limits[i].D
        if s == 2
            A_zero_obs = zeros(size(A_obs,1), 2)
            A_new_obs = hcat(A_obs[:,1], A_zero_obs, A_obs[:,2], A_zero_obs)
        elseif s == 3
            A_zero_obs = zeros(size(A_obs,1), 3)
            A_new_obs = hcat(A_obs[:,1], A_zero_obs, A_obs[:,2], A_zero_obs)
        end
        push!(obs_list_new, Cons_Corrs(A_new_obs, B_obs, C_obs, D_obs))
        push!(A_cons, vcat(A_new_limits, A_new_obs))
        push!(b_cons, vcat(B_limits, B_obs))
    end
    return A_cons, b_cons
end

A_cons, b_cons = process_polyhedra_car(limits_list, obs_list)
A_cons = - A_cons
b_cons = - b_cons
K = length(b_cons[1])

function Lopt_fun(T)
    Lopt_00 = Γ_fun(1.0, d) * kron(A_cons[1][1,:]', Matrix(1.0I, d+1, d+1)) * Πs
    Lopt_lj = BlockArray{eltype(T),2}(undef, fill(size(Lopt_00,1),N), fill(size(Lopt_00,2),K))
    fill!(Lopt_lj, 0.0)
    for l in 1:N
        for j in 1:K
            Lopt_temp = Γ_fun(T[l], d) * kron(A_cons[l][j,:]', Matrix(1.0I, d+1, d+1)) * Πs
            setblock!(Lopt_lj, Lopt_temp, l, j)
        end
    end
    Lopt_l = []
    for l in 1:N
        Lopt_temp = Matrix(hcat([getblock(Lopt_lj, l, j)' for j in 1:K]...)')
        push!(Lopt_l, Lopt_temp)
    end
    Lopt = direct_sum_mat(Lopt_l)
    return Lopt
end
# Lopt = Lopt_fun(T)
# Lopt_grad = ForwardDiff.jacobian(Lopt_fun, T)
# Lopt_grad2 = reshape(Lopt_grad, (size(Lopt)...,N))

function gopt_fun()
    g_00 = reshape(b_cons[1][1] * vcat(zeros(d),1.0), d+1, 1)
    g_lj = BlockArray{Float64,2}(undef, fill(size(g_00,1),N), fill(size(g_00,2),K))
    fill!(g_lj, 0.0)
    for l in 1:N
        for j in 1:K
            g_temp = reshape(b_cons[l][j] * vcat(zeros(d),1.0), d+1, 1)
            setblock!(g_lj, g_temp, l, j)
        end
    end
    g_l = []
    for l in 1:N
        g_temp = Matrix(hcat([getblock(g_lj, l, j)' for j in 1:K]...)')[:,1]
        push!(g_l, g_temp)
    end
    gopt = direct_sum_vec(g_l)
    return gopt
end
# gopt = gopt_fun()

function Ropt_fun(T)
    Lopt = Lopt_fun(T)
    Hopt = Hopt_fun(T)
    Ropt = vcat(Lopt,Hopt)
    return Ropt
end
# Ropt = Ropt_fun(T)
# Ropt_grad = ForwardDiff.jacobian(Ropt_fun, T)
# Ropt_grad2 = reshape(Ropt_grad, (size(Ropt)...,N))

function RRopt_fun(T)
    Ropt = Ropt_fun(T)
    RRopt = Ropt' * Ropt
    return RRopt
end
# RRopt = RRopt_fun(T)
# RRopt_grad = ForwardDiff.jacobian(RRopt_fun, T)
# RRopt_grad2 = reshape(RRopt_grad, (size(RRopt)...,N))

# Calculate Mapping M: F and G #
function calc_FG(d)
    d2=Int((d-1)/2+1)
    Fconst = zeros(d2,d2,d)

    for ix in CartesianIndices(Fconst)
        i, j, k = ix.I
        if i + j == k + 1
            Fconst[ix] = 1.
        end
    end
    Fout=zeros(d2,d2,d+1)
    Gout=zeros(d2,d2,d+1)
    for i in 1:d+1
        if i==1
            Fout[:,:,i]=Fconst[:,:,i]
            Gout[:,:,i]=-Fconst[:,:,i]
        elseif i==d+1
            Fout[:,:,i]=0*Fconst[:,:,d]
            Gout[:,:,i]=Fconst[:,:,d]
        else
            Fout[:,:,i]=Fconst[:,:,i]
            Gout[:,:,i]=-Fconst[:,:,i]+Fconst[:,:,i-1]
        end

    end

    return Fout,Gout
end

Fopt,Gopt = calc_FG(d)
FG = cat(Fopt, Gopt, dims=3)

# Calculate Para of PDHG in one function #
function calc_PHLR(T)
    Popt = Popt_fun(T)
    Hopt = Hopt_fun(T)
    ropt = ropt_fun()
    Lopt = Lopt_fun(T)
    gopt = gopt_fun()
    Ropt = vcat(Lopt,Hopt)
    return Popt, Hopt, ropt, Lopt, gopt, Ropt
end

function calc_Para(T)
    Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_PHLR(T)
    dim1_P, dim2_P = size(Popt)
    dim1_L, dim2_L = size(Lopt)
    dim1_H, dim2_H = size(Hopt)
    I_22 = -1/β * Matrix(1.0I, dim1_L, dim1_L)
    Z_23 = zeros(dim1_L, dim1_H)
    Z_32 = zeros(dim1_H, dim1_L)
    Z_33 = zeros(dim1_H, dim1_H)
    PLH_mat = [2Popt Lopt' Hopt'; Lopt I_22 Z_23; Hopt Z_32 Z_33]
    PLH_inv = inv(PLH_mat)

    Z_1 = zeros(dim1_P, dim1_L)
    I_2 = Matrix(1.0I, dim1_L, dim1_L)
    Z_3 = zeros(dim1_H, dim1_L)
    I_sec = [Z_1; I_2; Z_3]
    gr_sec = vcat(zeros(dim1_P), gopt, ropt)
    

    IinE = Matrix(1.0I, dim1_L, dim1_L)
    Eopt = IinE + 1/β * I_sec' * PLH_inv * I_sec
    eopt = 1/β * I_sec' * PLH_inv * gr_sec
    return Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt
end

##
# Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T);