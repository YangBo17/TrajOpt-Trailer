using TensorOperations
using CUDA
using Base.Threads
using DynamicPolynomials
@polyvar t

BLOCK = 128
THREADS_PER_BLOCK = Int(ceil(N * 2(m+n) / BLOCK))

function eigenval_recover_batched!(A, D, m)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x
    for i = index:stride:length(D)
        if D[i] < 0.
	        @inbounds D[i] = 0.
		end
        for j = 1:m
            A[(i-1) * m + j] *= D[i]
        end
    end
    return
end

function projection_batched(X_d, m::Int64)
    D, A = CUSOLVER.syevjBatched!('V', 'U', X_d)
    A_c = copy(A)
    @cuda threads=THREADS_PER_BLOCK blocks=BLOCK eigenval_recover_batched!(A, D, m)
    X_d = CUBLAS.gemm_strided_batched('N', 'T', 1., A, A_c)
    return X_d
end

function projection(X_d)
    @threads for i in axes(X_d, 3)
        D, A = eigen((X_d[:, :, i] + X_d[:, :, i]') / 2) 
        D = max.(D, 0.0)
        X_d[:, :, i] = A * Diagonal(D) * A'
    end
    return X_d
end

function calc_pd_gap(X, X_prev, λ, λ_prev, α, β, FG)
    # calculate differences
    X_diff = (X - X_prev) / α
    λ_diff = (λ - λ_prev) / β

    # calculate the gap
    @tensor begin
        M_adj_λ_diff[a,b,j] = λ_diff[i,j] * FG[a,b,i]
    end
    @tensor begin
        M_X_diff[i,j] = FG[a,b,i] * X_diff[b,a,j]
    end

    # calculate the norm
    primal_gap = norm(X_diff - M_adj_λ_diff)
    dual_gap = norm(λ_diff - M_X_diff)
    gap = primal_gap + dual_gap
    return gap, primal_gap, dual_gap
end

function PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=false, last_X=nothing, last_λ=nothing)
    # PDHG iteration
    if !isnothing(last_λ)
        last_λ = reshape(last_λ, d+1, K*N)
    end
    begin
        d2 = Int((d-1)/2+1)
        if warm_start
            X = last_X
            λ = last_λ
        else
            X = randn(d2, d2, 2K*N) .- 0.5
            λ = zeros(d+1, K*N) .- 0.5
        end

        α = cu(α)
        β = cu(β)
        Eopt = cu(Eopt)
        eopt = cu(eopt)
        I_E = cu(Matrix(1.0I, size(Eopt)))
        IminusEopt = cu(I_E - Eopt)

        FG = cu(FG)
        X = cu(X)
        dX = cu(zeros(size(X)))

        λ_large = zeros(2*(d+1), 2*K*N)
        λ_large[1:d+1, 1:K*N] = λ
        λ_large[d+2:end, K*N+1:end] = λ
        λ = cu(λ)
        λ_large = cu(λ_large)
        λ_large_tmp = cu(zeros(size(λ_large)))
    end

    time0 = time()
    begin
        X_list = []
        λ_list = []
        for k in 1:max_iter
            X_prev = copy(X)
            λ_prev = copy(λ)
            λ_large[1:d+1, 1:K*N] = λ
            λ_large[d+2:end, K*N+1:end] = λ
            @tensor begin
                X[a, b, j] -= α * λ_large[i, j] * FG[a, b, i]
            end
            X = projection_batched(X, d2)
            @tensor begin
                dX = 2 * X - X_prev
                λ_large_tmp[i, j] = β * FG[a, b, i] * dX[b, a, j]
            end
            λ += λ_large_tmp[1:d+1, 1:K*N] + λ_large_tmp[d+2:end, K*N+1:end]
            tmp = copy(eopt)
            λ = reshape(λ, K*N*(d+1))
            CUBLAS.symv!('U', 1., IminusEopt, λ, -β, tmp)
            push!(X_list, copy(X))
            push!(λ_list, copy(λ))
            λ = reshape(tmp, d+1, K*N)
        end
    end
    time1 = time()
    sol_time = time1 - time0
    Xstar = X
    λstar = λ
    GC.gc()
    return Xstar, λstar, X_list, λ_list, sol_time
end

function PDHG_CPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=false, last_X=nothing, last_λ=nothing)
    # PDHG iteration
    if !isnothing(last_λ)
        last_λ = reshape(last_λ, d+1, K*N)
    end
    begin
        d2 = Int((d-1)/2+1)
        if warm_start
            X = last_X
            λ = last_λ
        else
            X = randn(d2, d2, 2K*N) .- 0.5
            λ = zeros(d+1, K*N) .- 0.5
        end

        I_E = Matrix(1.0I, size(Eopt))
        IminusEopt = I_E - Eopt
        dX = zeros(size(X))

        λ_large = zeros(2*(d+1), 2*K*N)
        λ_large[1:d+1, 1:K*N] = λ
        λ_large[d+2:end, K*N+1:end] = λ
        λ_large_tmp = copy(λ_large)
    end

    time0 = time()
    begin
        X_list = []
        λ_list = []
        for k in 1:max_iter
            X_prev = copy(X)
            λ_prev = copy(λ)
            λ_large[1:d+1, 1:K*N] = λ
            λ_large[d+2:end, K*N+1:end] = λ
            for a in axes(X, 1), b in axes(X, 2), j in axes(X, 3), i in axes(FG, 3)
                X[a, b, j] -= α * λ_large[i, j] * FG[a, b, i]
            end
            X = projection(X)
            for a in axes(X, 1), b in axes(X, 2), j in axes(X, 3), i in axes(FG, 3)
                dX = 2 * X - X_prev
                λ_large_tmp[i, j] = β * FG[a, b, i] * dX[b, a, j]  
            end
            λ += λ_large_tmp[1:d+1, 1:K*N] + λ_large_tmp[d+2:end, K*N+1:end]
            tmp = copy(eopt)
            λ = reshape(λ, K*N*(d+1))
            tmp = IminusEopt * λ .- β * tmp
            λ = reshape(tmp, d+1, K*N)
            push!(X_list, copy(X))
            push!(λ_list, copy(λ))
        end
    end
    time1 = time()
    sol_time = time1 - time0
    Xstar = X
    λstar = λ
    return Xstar, λstar, X_list, λ_list, sol_time
end

function calc_sopt(FG, Xstar)
    sopt_large = cu(zeros(2*(d+1), 2*K*N))
    FG = cu(FG)
    Xstar = cu(Xstar)
    @tensor begin
        sopt_large[i, j] = FG[a, b, i] * Xstar[b, a, j]
    end
    sopt = sopt_large[1:d+1, 1:K*N] + sopt_large[d+2:end, K*N+1:end]
    sopt = reshape(sopt, K*N*(d+1))
    sopt = collect(sopt)
    return sopt
end

function calc_cstar(Hopt,ropt,Lopt,gopt,sopt)
    Ropt = vcat(Lopt, Hopt)
    qopt = vcat(gopt+sopt, ropt)
    
    cstar = Ropt \ qopt
    return cstar
end
function calc_λstar(λstar)
    λstar_vec = Array(reshape(-λstar, size(λstar,1)*size(λstar,2)))
    return λstar_vec
end

function calc_νstar(cstar,λstar,Popt,Hopt,Lopt)
    νstar = -inv(Hopt * Hopt') * Hopt * (2Popt * cstar + Lopt' * λstar)
    return νstar
end

function calc_poly_RR(Popt,Hopt,Lopt,ropt,gopt,sopt)
    Ropt = vcat(Lopt, Hopt)
    qopt = vcat(gopt+sopt, ropt)
    dim1_P, dim2_P = size(Popt)
    dim1_R, dim2_R = size(Ropt)
    RRopt = Ropt' * Ropt
    Rqopt = Ropt' * qopt
    dim1_RR, dim2_RR = size(RRopt)
    Z_22 = zeros(dim1_RR, dim1_RR)
    Z_1 = zeros(dim1_P)
    KKT_A = [2Popt RRopt'; RRopt Z_22]
    KKT_b = [Z_1; Rqopt]
    opt_sol = inv(KKT_A) * KKT_b
    cstar = opt_sol[1:dim1_P]
    ϵstar = opt_sol[dim1_P+1:end]
    return cstar, ϵstar, sopt, KKT_A, KKT_b
end

function calc_poly(Popt,Hopt,Lopt,ropt,gopt,sopt)
    Ropt = vcat(Lopt, Hopt)
    qopt = vcat(gopt+sopt, ropt)
    dim1_P, dim2_P = size(Popt)
    dim1_R, dim2_R = size(Ropt)
    Z_22 = zeros(dim1_R, dim1_R)
    Z_1 = zeros(dim1_P)
    KKT_A = [2Popt Ropt'; Ropt Z_22]
    KKT_b = [Z_1; qopt]
    opt_sol = pinv(KKT_A) * KKT_b
    cstar = opt_sol[1:dim1_P]
    ϵstar = opt_sol[dim1_P+1:end]
    λstar = ϵstar[1:K*N*(d+1)]
    νstar = ϵstar[K*N*(d+1)+1:end]
    return cstar, ϵstar, KKT_A, KKT_b, λstar, νstar
end

function calc_traj(T, cstar; dt=0.02)
    cstar = reshape(cstar, d+1, m, N)
    poly_basis = monomials([t], 0:d)
    p = [cstar[:,i,j]' * poly_basis for i in 1:m, j in 1:N]
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    D3p = differentiate.(p,t,3)
    data = []
    total_time = [0.0]
    for j in 1:N 
        if j != N 
            ts = 0:dt:T[j]-dt |> collect
        else
            ts = 0:dt:T[j] |> collect
        end
        push!(total_time, total_time[end] + T[j])
        for i in eachindex(ts)
            ti = ts[i] + total_time[j]
            x1 = p[1,j](t=>ts[i])
            y1 = p[2,j](t=>ts[i])
            vx1 = D1p[1,j](t=>ts[i])
            vy1 = D1p[2,j](t=>ts[i])
            v1 = sqrt(vx1^2+vy1^2)
            ax1 = D2p[1,j](t=>ts[i])
            ay1 = D2p[2,j](t=>ts[i])
            a1 = sqrt(ax1^2+ay1^2)
            jx1 = D3p[1,j](t=>ts[i])
            jy1 = D3p[2,j](t=>ts[i])
            j1 = sqrt(jx1^2+jy1^2)
            x0 = x1 + d1 * vx1 / v1
            y0 = y1 + d1 * vy1 / v1
            vx0 = vx1 - d1 * vy1 * (ay1*vx1-vy1*ax1) / (v1^3)
            vy0 = vy1 + d1 * vx1 * (ay1*vx1-vy1*ax1) / (v1^3)
            ax0 = ax1 - d1 / v1^6 * ( (ay1*(ay1*vx1-vy1*ax1)+vy1*(jy1*vx1-vy1*jx1))*v1^3 - vy1*(ay1*vx1-vy1*ax1)*3/2*v1*(2*vx1*ax1+2*vy1*ay1) )
            ay0 = ay1 + d1 / v1^6 * ( (ax1*(ay1*vx1-vy1*ax1)+vx1*(jy1*vx1-vy1*jx1))*v1^3 - vx1*(ay1*vx1-vy1*ax1)*3/2*v1*(2*vx1*ax1+2*vy1*ay1) )
            θ1 = atan(vy1, vx1)
            θ0 = atan(vy0, vx0)
            v = sqrt(vx0^2+vy0^2)
            a = sqrt(ax0^2+ay0^2)
            ϕ = atan(d0*(ay0*vx0-vy0*ax0), v^3)
            # κ = (ay1*vx1-vy1*ax1)/v1^3
            κ = (ay0*vx0-vy0*ax0)/v^3

            datapoint = TrailerState(ti, x0, y0, vx0, vy0, ax0, ay0, x1, y1, vx1, vy1, ax1, ay1, jx1, jy1, θ0, θ1, v,a, ϕ, κ)
            push!(data, datapoint) 
        end
    end
    return data
end

function QP_diff_solver(FG,Xstar,Popt,Hopt,Lopt,ropt,gopt,sopt)
    #= 
        sopt = M(Xstar), solve QP's KKT equation obatin c, the parameters of polynomials
        ForwardDiff to compute the gradient of optimal solution w.r.t T
    =#
    # compute the optimal solution
    Ropt = vcat(Lopt, Hopt)
    qopt = vcat(gopt+sopt, ropt)
    dim1_P, dim2_P = size(Popt)
    dim1_R, dim2_R = size(Ropt)
    Z_22 = zeros(dim1_R, dim1_R)
    Z_1 = zeros(dim1_P)
    KKT_A = [2Popt Ropt'; Ropt Z_22]
    KKT_b = [Z_1; qopt]
    KKT_A_inv = pinv(KKT_A)
    opt_sol = KKT_A_inv * KKT_b
    cstar = opt_sol[1:dim1_P]
    ϵstar = opt_sol[dim1_P+1:end]

    # compute the gradient of optimal solution w.r.t T
    function KKT_A_fun(T)
        Popt_val, Hopt_val, ropt_val, Lopt_val, gopt_val = calc_PHLR(T)
        Ropt_val = vcat(Lopt_val, Hopt_val)
        dim1_R, dim2_R = size(Ropt_val)
        Z_22 = zeros(dim1_R, dim1_R)
        KKT_A_val = [2Popt_val Ropt_val'; Ropt_val Z_22]
        return KKT_A_val
    end

    KKT_A_grad = ForwardDiff.jacobian(KKT_A_fun, T)
    KKT_A_grad2 = reshape(KKT_A_grad, (size(KKT_A)...,N))
    opt_sol_grad2 = zeros(size(opt_sol)...,N)
    cstar_grad2 = zeros(size(cstar)...,N)
    ϵstar_grad2 = zeros(size(ϵstar)...,N)
    for i in 1:N
        opt_sol_grad2[:,i] = -KKT_A_inv * KKT_A_grad2[:,:,i] * KKT_A_inv * KKT_b
        cstar_grad2[:,i] = opt_sol_grad2[1:dim1_P,i]
        ϵstar_grad2[:,i] = opt_sol_grad2[dim1_P+1:end,i]
    end
    return cstar, ϵstar, cstar_grad2, ϵstar_grad2, qopt
end

function QP_diff_solver_RR(FG,Xstar,Popt,Hopt,Lopt,ropt,gopt,sopt)
    #= 
        sopt = M(Xstar), solve QP's KKT equation obatin c, the parameters of polynomials
        ForwardDiff to compute the gradient of optimal solution w.r.t T
    =#
    # compute the optimal solution
    Ropt = vcat(Lopt, Hopt)
    qopt = vcat(gopt+sopt, ropt)
    dim1_P, dim2_P = size(Popt)
    dim1_R, dim2_R = size(Ropt)
    RRopt = Ropt' * Ropt
    Rqopt = Ropt' * qopt
    dim1_RR, dim2_RR = size(RRopt)
    Z_22 = zeros(dim1_RR, dim1_RR)
    Z_1 = zeros(dim1_P)
    KKT_A = [2Popt RRopt'; RRopt Z_22]
    KKT_b = [Z_1; Rqopt]
    KKT_A_inv = inv(KKT_A)
    opt_sol = KKT_A_inv * KKT_b
    cstar = opt_sol[1:dim1_P]
    ϵstar = opt_sol[dim1_P+1:end]

    # compute the gradient of optimal solution w.r.t T
    function KKT_A_fun(T)
        Popt_val, Hopt_val, ropt_val, Lopt_val, gopt_val = calc_PHLR(T)
        Ropt_val = vcat(Lopt_val, Hopt_val)
        RRopt_val = Ropt_val' * Ropt_val
        dim1_RR, dim2_RR = size(RRopt_val)
        Z_22 = zeros(dim1_RR, dim1_RR)
        KKT_A_val = [2Popt_val RRopt_val'; RRopt_val Z_22]
        return KKT_A_val
    end
    function KKT_b_fun(T)
        Popt_val, Hopt_val, ropt_val, Lopt_val, gopt_val = calc_PHLR(T)
        dim1_P, dim2_P = size(Popt_val)
        Ropt_val = vcat(Lopt_val, Hopt_val)
        Rqopt_val = Ropt_val' * qopt
        Z_1 = zeros(dim1_P)
        KKT_b_val = [Z_1; Rqopt_val]
        return KKT_b_val
    end

    KKT_A_grad = ForwardDiff.jacobian(KKT_A_fun, T)
    KKT_A_grad2 = reshape(KKT_A_grad, (size(KKT_A)...,N))

    KKT_b_grad = ForwardDiff.jacobian(KKT_b_fun, T)
    KKT_b_grad2 = reshape(KKT_b_grad, (size(KKT_b)...,N))

    opt_sol_grad2 = zeros(size(opt_sol)...,N)
    cstar_grad2 = zeros(size(cstar)...,N)
    ϵstar_grad2 = zeros(size(ϵstar)...,N)
    for i in 1:N
        opt_sol_grad2[:,i] = -KKT_A_inv * KKT_A_grad2[:,:,i] * KKT_A_inv * KKT_b + KKT_A_inv * KKT_b_grad2[:,i]
        cstar_grad2[:,i] = opt_sol_grad2[1:dim1_P,i]
        ϵstar_grad2[:,i] = opt_sol_grad2[dim1_P+1:end,i]
    end
    return cstar, ϵstar, cstar_grad2, ϵstar_grad2, qopt
end

function convergence_analysis(FG,T,X_list,λ_list)
    Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_PHLR(T)
    # calculate objective value in each iteration
    obj_list = []
    for i in eachindex(X_list)
        cstar, ϵstar = calc_poly(FG,X_list[i],Popt,Hopt,Lopt,ropt,gopt,Ropt)
        obj = calc_obj(T,cstar,Popt; time_penalty=false)
        push!(obj_list, obj)
    end

    # calculate primal and dual gap in each iteration
    gap_list = []
    primal_gap_list = []
    dual_gap_list = []
    for i in 2:length(X_list)
        gap, primal_gap, dual_gap = calc_pd_gap(X_list[i], X_list[i-1], λ_list[i], λ_list[i-1], α, β, FG)
        push!(gap_list, gap)
        push!(primal_gap_list, primal_gap)
        push!(dual_gap_list, dual_gap)
    end
    return obj_list, gap_list, primal_gap_list, dual_gap_list
end

function calc_obj(T,cstar,Popt; time_penalty=true, Tw=nothing)
    if time_penalty
        if Tw == nothing
            Tw = zeros(size(T))
        end
        obj = Tw' * T + cstar' * Popt * cstar
    else
        obj = cstar' * Popt * cstar
    end
    return obj
end

function calc_Lag(T,Popt, Ropt, qopt,cstar,ϵstar; time_penalty=true, Tw=nothing)
    if time_penalty
        if Tw == nothing
            Tw = zeros(N)
        end
        # Lag = Tw' * T + cstar' * Popt * cstar + 1*ϵstar' * (Ropt * cstar - qopt)
        Lag = Tw' * T + cstar' * Popt * cstar
    else
        # Lag = cstar' * Popt * cstar + 1*ϵstar' * (Ropt * cstar - qopt)
        Lag = cstar' * Popt * cstar
    end
    return Lag
end

function calc_Lag_grad(T,Popt,Ropt,qopt,cstar,ϵstar,cstar_grad2,ϵstar_grad2; time_penalty=true, Tw=nothing)

    Popt_grad = ForwardDiff.jacobian(Popt_fun, T)
    Popt_grad2 = reshape(Popt_grad, (size(Popt)...,N))

    Ropt_grad = ForwardDiff.jacobian(Ropt_fun, T)
    Ropt_grad2 = reshape(Ropt_grad, (size(Ropt)...,N))

    Lag_grad2 = zeros(N)
    for i in 1:N
        if time_penalty
            if Tw == nothing
                Tw = zeros(size(T))
            end
            Lag_grad2[i] = Tw[i] + cstar'*Popt_grad2[:,:,i]*cstar + ϵstar'*(Ropt_grad2[:,:,i]*cstar) #+ (2Popt * cstar + Ropt' * ϵstar)' * cstar_grad2[:,i] + (Ropt * cstar - qopt)' * ϵstar_grad2[:,i]
            # Lag_grad2[i] = Tw[i] + cstar'*Popt_grad2[:,:,i]*cstar + (2Popt * cstar)' * cstar_grad2[:,i]
        else
            Lag_grad2[i] = cstar'*Popt_grad2[:,:,i]*cstar + ϵstar'*(Ropt_grad2[:,:,i]*cstar) + (2Popt * cstar + Ropt' * ϵstar)' * cstar_grad2[:,i] + (Ropt * cstar - qopt)' * ϵstar_grad2[:,i]
            # Lag_grad2[i] = cstar'*Popt_grad2[:,:,i]*cstar + (2Popt * cstar)' * cstar_grad2[:,i]
        end
    end
    return Lag_grad2
end

function calc_Lag_grad_RR(T,Popt,Ropt,qopt,cstar,ϵstar,cstar_grad2,ϵstar_grad2; time_penalty=true, Tw=nothing)

    RRopt = Ropt' * Ropt
    Rqopt = Ropt' * qopt

    function Rqopt_fun(T)
        Popt_val, Hopt_val, ropt_val, Lopt_val, gopt_val = calc_PHLR(T)
        Ropt_val = vcat(Lopt_val, Hopt_val)
        Rqopt_val = Ropt_val' * qopt
        return Rqopt_val
    end

    Popt_grad = ForwardDiff.jacobian(Popt_fun, T)
    Popt_grad2 = reshape(Popt_grad, (size(Popt)...,N))

    RRopt_grad = ForwardDiff.jacobian(RRopt_fun, T)
    RRopt_grad2 = reshape(RRopt_grad, (size(RRopt)...,N))
    Rqopt_grad = ForwardDiff.jacobian(Rqopt_fun, T)
    Rqopt_grad2 = reshape(Rqopt_grad, (size(Rqopt)...,N))

    Lag_grad2 = zeros(N)
    for i in 1:N
        if time_penalty
            if Tw == nothing
                Tw = zeros(size(T))
            end
            Lag_grad2[i] = Tw[i] + cstar'*Popt_grad2[:,:,i]*cstar + ϵstar'*(RRopt_grad2[:,:,i]*cstar - Rqopt_grad2[:,i]) #+ (2Popt * cstar + RRopt' * ϵstar)' * cstar_grad2[:,i] + (RRopt * cstar - Rqopt)' * ϵstar_grad2[:,i]
        else
            # Lag_grad2[i] = cstar'*Popt_grad2[:,:,i]*cstar + ϵstar'*(RRopt_grad2[:,:,i]*cstar - Rqopt_grad2[:,i]) + (2Popt * cstar + RRopt' * ϵstar)' * cstar_grad2[:,i] + (RRopt * cstar - Rqopt)' * ϵstar_grad2[:,i]
            Lag_grad2[i] = cstar'*Popt_grad2[:,:,i]*cstar + (2Popt * cstar)' * cstar_grad2[:,i]
        end
    end
    return Lag_grad2
end

function calc_Lag_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=true, Tw=nothing)
    if time_penalty
        if Tw == nothing
            Tw = zeros(N)
        end
        Lag = Tw' * T + cstar' * Popt * cstar + λstar' * (Lopt * cstar - gopt - sopt) + νstar' * (Hopt * cstar - ropt)
        # Lag = Tw' * T + cstar' * Popt * cstar
    else
        Lag = cstar' * Popt * cstar + λstar' * (Lopt * cstar - gopt - sopt) + νstar' * (Hopt * cstar - ropt)
        # Lag = cstar' * Popt * cstar
    end
    return Lag
end

function calc_Lag_grad_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=true, Tw=nothing)
    Popt_grad = ForwardDiff.jacobian(Popt_fun, T)
    Popt_grad2 = reshape(Popt_grad, (size(Popt)...,N))

    Lopt_grad = ForwardDiff.jacobian(Lopt_fun, T)
    Lopt_grad2 = reshape(Lopt_grad, (size(Lopt)...,N))

    Hopt_grad = ForwardDiff.jacobian(Hopt_fun, T)
    Hopt_grad2 = reshape(Hopt_grad, (size(Hopt)...,N))

    Lag_grad2 = zeros(N)
    for i in 1:N
        if time_penalty
            if Tw == nothing
                Tw = zeros(size(T))
            end
            Lag_grad2[i] = Tw[i] + cstar'*Popt_grad2[:,:,i]*cstar + λstar'*Lopt_grad2[:,:,i]*cstar + νstar'*Hopt_grad2[:,:,i]*cstar
        else
            Lag_grad2[i] = cstar'*Popt_grad2[:,:,i]*cstar + λstar'*Lopt_grad2[:,:,i]*cstar + νstar'*Hopt_grad2[:,:,i]*cstar
        end
    end
    return Lag_grad2
end

function TrajOpt_space(T, α, β, FG; device=:GPU, max_iter=max_iter, warm_start=false, last_X=nothing, last_λ=nothing)
    Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T)
    if device == :GPU 
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)
    elseif device == :CPU
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_CPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)
    end
    sopt = calc_sopt(FG,Xstar)
    cstar = calc_cstar(Hopt,ropt,Lopt,gopt,sopt)
    λstar = calc_λstar(λstar)
    νstar = calc_νstar(cstar,λstar,Popt,Hopt,Lopt)
    obj = calc_obj(T,cstar,Popt; time_penalty=false)
    Lag = calc_Lag_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = calc_Lag_grad_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = Lag_grad2 / norm(Lag_grad2)
    data = calc_traj(T, cstar; dt=0.02)
    return data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar
end

function TrajOpt_space_new(T, α, β, FG; device=:GPU, max_iter=max_iter, warm_start=false, last_X=nothing, last_λ=nothing)
    Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T)
    if device == :GPU 
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)
    elseif device == :CPU
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_CPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)
    end
    sopt = calc_sopt(FG,Xstar)
    cstar = calc_cstar(Hopt,ropt,Lopt,gopt,sopt)
    λstar = calc_λstar(λstar)
    νstar = calc_νstar(cstar,λstar,Popt,Hopt,Lopt)
    obj = calc_obj(T,cstar,Popt; time_penalty=false)
    Lag = calc_Lag_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = calc_Lag_grad_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = Lag_grad2 / norm(Lag_grad2)
    data = calc_traj(T, cstar; dt=0.02)
    return data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar
end

function TrajOpt_space_fd(T, Tw, α, β, FG; device=:GPU, max_iter=max_iter, warm_start=false, last_X=nothing, last_λ=nothing)
    para_time0 = time()
    Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T)
    para_time1 = time()
    para_time = para_time1 - para_time0
    if device == :GPU 
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)
    elseif device == :CPU
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_CPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)
    end
    sopt = calc_sopt(FG,Xstar)
    cstar = calc_cstar(Hopt,ropt,Lopt,gopt,sopt)
    λstar = calc_λstar(λstar)
    νstar = calc_νstar(cstar,λstar,Popt,Hopt,Lopt)
    obj = calc_obj(T,cstar,Popt; time_penalty=false)
    Lag = calc_Lag_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    grad_time0 = time()
    Lag_grad2 = calc_Lag_grad_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=true, Tw=Tw)
    grad_time1 = time()
    grad_time = grad_time1 - grad_time0
    data = calc_traj(T, cstar; dt=0.02)

    # @show obj + Tw' * T
    # @show Lag + Tw' * T
    # kkt_01 = 2Popt * cstar + Lopt' * λstar + Hopt' * νstar
    # kkt_02 = Lopt * cstar - sopt - gopt
    # kkt_03 = Hopt * cstar - ropt
    # lag_λ = λstar' * (Lopt * cstar - gopt - sopt)
    # lag_ν = νstar' * (Hopt * cstar - ropt)
    # @show kkt01_val = norm(kkt_01)
    # @show kkt02_val = norm(kkt_02)
    # @show kkt03_val = norm(kkt_03)
    # @show lag_λ + lag_ν
    # @show Lag - obj
    # @show norm(λstar)
    # @show norm(νstar)
    # @show norm(Lopt * cstar - gopt - sopt)
    # @show norm(Hopt * cstar - ropt)

    return data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, Xstar, λstar, νstar, grad_time, para_time
end

function TrajOpt_space_dP(T, α, β, FG; device=:GPU, max_iter=max_iter, warm_start=false, last_X=nothing, last_λ=nothing, dP=nothing)
    Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para_dP(T, dP)
    # display("text/plain", Lopt0[1:10,1:10]) 
    # display("text/plain", Lopt[1:10,1:10]) 
    # display("text/plain", Popt[1:10,1:10] - Popt0[1:10,1:10]) 
    if device == :GPU 
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)
    elseif device == :CPU
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_CPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)  
    end
    sopt = calc_sopt(FG,Xstar)
    cstar = calc_cstar(Hopt,ropt,Lopt,gopt,sopt)
    λstar = calc_λstar(λstar)
    νstar = calc_νstar(cstar,λstar,Popt,Hopt,Lopt)
    obj = calc_obj(T,cstar,Popt; time_penalty=false)
    Lag = calc_Lag_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = calc_Lag_grad_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = Lag_grad2 / norm(Lag_grad2)
    data = calc_traj(T, cstar; dt=0.02)

    @show 1
    @show obj 
    @show Lag 
    kkt_01 = 2Popt * cstar + Lopt' * λstar + Hopt' * νstar
    kkt_02 = Lopt * cstar - sopt - gopt
    kkt_03 = Hopt * cstar - ropt
    lag_λ = λstar' * (Lopt * cstar - gopt - sopt)
    lag_ν = νstar' * (Hopt * cstar - ropt)
    @show kkt01_val = norm(kkt_01)
    @show kkt02_val = norm(kkt_02)
    @show kkt03_val = norm(kkt_03)
    @show lag_λ + lag_ν
    @show Lag - obj
    @show norm(λstar)
    @show norm(νstar)
    @show norm(Lopt * cstar - gopt - sopt)
    @show norm(Hopt * cstar - ropt)

    return data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar
end

function TrajOpt_space_dL(T, α, β, FG; device=:GPU, max_iter=max_iter, warm_start=false, last_X=nothing, last_λ=nothing, dL=nothing)
    Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para_dL(T, dL)
    # display("text/plain", Lopt0[1:10,1:10]) 
    # display("text/plain", Lopt[1:10,1:10]) 
    # display("text/plain", Lopt[1:10,1:10] - Lopt0[1:10,1:10]) 
    if device == :GPU 
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)
    elseif device == :CPU
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_CPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)  
    end
    sopt = calc_sopt(FG,Xstar)
    cstar = calc_cstar(Hopt,ropt,Lopt,gopt,sopt)
    λstar = calc_λstar(λstar)
    νstar = calc_νstar(cstar,λstar,Popt,Hopt,Lopt)
    obj = calc_obj(T,cstar,Popt; time_penalty=false)
    Lag = calc_Lag_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = calc_Lag_grad_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = Lag_grad2 / norm(Lag_grad2)
    data = calc_traj(T, cstar; dt=0.02)

    @show 1
    @show obj 
    @show Lag 
    kkt_01 = 2Popt * cstar + Lopt' * λstar + Hopt' * νstar
    kkt_02 = Lopt * cstar - sopt - gopt
    kkt_03 = Hopt * cstar - ropt
    lag_λ = λstar' * (Lopt * cstar - gopt - sopt)
    lag_ν = νstar' * (Hopt * cstar - ropt)
    @show kkt01_val = norm(kkt_01)
    @show kkt02_val = norm(kkt_02)
    @show kkt03_val = norm(kkt_03)
    @show lag_λ + lag_ν
    @show Lag - obj
    @show norm(λstar)
    @show norm(νstar)
    @show norm(Lopt * cstar - gopt - sopt)
    @show norm(Hopt * cstar - ropt)

    return data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar
end

function TrajOpt_space_dH(T, α, β, FG; device=:GPU, max_iter=max_iter, warm_start=false, last_X=nothing, last_λ=nothing, dH=nothing)
    Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para_dH(T, dH)
    # display("text/plain", Hopt - Hopt0) 
    if device == :GPU 
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)
    elseif device == :CPU
        Xstar, λstar, X_list, λ_list, sol_time = PDHG_CPU_solver(Eopt, eopt, α, β, FG; max_iter=max_iter, warm_start=warm_start, last_X=last_X, last_λ=last_λ)  
    end
    sopt = calc_sopt(FG,Xstar)
    cstar = calc_cstar(Hopt,ropt,Lopt,gopt,sopt)
    λstar = calc_λstar(λstar)
    νstar = calc_νstar(cstar,λstar,Popt,Hopt,Lopt)
    obj = calc_obj(T,cstar,Popt; time_penalty=false)
    Lag = calc_Lag_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = calc_Lag_grad_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=false)
    Lag_grad2 = Lag_grad2 / norm(Lag_grad2)
    data = calc_traj(T, cstar; dt=0.02)

    @show 1
    @show obj 
    @show Lag 
    kkt_01 = 2Popt * cstar + Lopt' * λstar + Hopt' * νstar
    kkt_02 = Lopt * cstar - sopt - gopt
    kkt_03 = Hopt * cstar - ropt
    lag_λ = λstar' * (Lopt * cstar - gopt - sopt)
    lag_ν = νstar' * (Hopt * cstar - ropt)
    @show kkt01_val = norm(kkt_01)
    @show kkt02_val = norm(kkt_02)
    @show kkt03_val = norm(kkt_03)
    @show lag_λ + lag_ν
    @show Lag - obj
    @show norm(λstar)
    @show norm(νstar)
    @show norm(Lopt * cstar - gopt - sopt)
    @show norm(Hopt * cstar - ropt)

    return data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, λstar, νstar
end


function SpatialTemporalOpt(T0, Tw, α, β, FG; outer_iter=5, inner_iter=500)
    T = T0
    Lag_tol = 1e-3
    Lag = 0
    Lag_opt = Inf
    obj = 0
    k = 1
    k_max = outer_iter
    Xstar = nothing
    λstar = nothing
    cstar = nothing
    ϵstar = nothing
    c_list = []
    ϵ_list = []
    obj_list = []
    Lag_list = []
    Lag_grad_list = []
    T_list = []
    grad_times = []
    para_times = []
    sol_times = []
    STopt_time0 = time()

    para_time0 = time()
    Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T)
    para_time1 = time()
    para_time = para_time1 - para_time0
    push!(para_times, para_time)
    Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=inner_iter, warm_start=false)
    push!(sol_times, sol_time)

    while k <= k_max
        grad_time0 = time()
        cstar, ϵstar, cstar_grad2, ϵstar_grad2, qopt = QP_diff_solver(FG,Xstar,Popt,Hopt,Lopt,ropt,gopt,Ropt)
        obj = calc_obj(T,cstar,Popt; time_penalty=true, Tw=Tw)
        Lag = calc_Lag(T,Popt,Ropt,qopt,cstar,ϵstar; time_penalty=true, Tw=Tw)
        Lag_grad2 = calc_Lag_grad(T,Popt,Ropt,qopt,cstar,ϵstar,cstar_grad2,ϵstar_grad2; time_penalty=true, Tw=Tw)
        Lag_grad2 = Lag_grad2 / norm(Lag_grad2)
        grad_time1 = time()
        grad_time = grad_time1 - grad_time0
        push!(grad_times, grad_time)
        push!(c_list, cstar)
        push!(ϵ_list, ϵstar)
        push!(obj_list, obj)
        push!(Lag_list, Lag)
        push!(Lag_grad_list, Lag_grad2)
        push!(T_list, T)
        @show obj
        @show Lag
        if abs(Lag - Lag_opt) <= Lag_tol
            @show Tstar = T
            break
        else
            if Lag < Lag_opt
                @show Lag_opt = Lag
                Tstar = T
            end
        end
        @show Lag_grad2
        T -= η * Lag_grad2
        Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T)
        k += 1
    end
    STopt_time1 = time()
    STopt_soltime = STopt_time1 - STopt_time0
    return cstar, Xstar, Tstar, c_list, ϵ_list, obj_list, Lag_list, Lag_grad_list, T_list, sol_times, STopt_soltime, grad_times, para_times
end

function BiLevelOpt(T0, Tw, α, β, FG; outer_iter=5, inner_iter=500)
    T = T0
    Lag_tol = 1e-3
    Lag = 0
    Lag_opt = Inf
    obj = 0
    k = 1
    k_max = outer_iter
    cstar = nothing
    Xstar = nothing
    λstar = nothing
    Tstar = nothing
    last_X = nothing
    last_λ = nothing
    c_list = []
    Lag_list = []
    Lag_grad_list = []
    T_list = []
    obj_list = []
    grad_times = []
    para_times = []
    sol_times = []
    bilevel_time0 = time()
    while k <= k_max
        time0 = time()
        para_time0 = time()
        Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T)
        para_time1 = time()
        para_time = para_time1 - para_time0
        push!(para_times, para_time)
        @show k
        if k == 1
            Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=500, warm_start=false)
        else
            Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=inner_iter, warm_start=true, last_X=last_X, last_λ=last_λ)
        end
        push!(sol_times, sol_time)
        last_X = Xstar
        last_λ = λstar
        grad_time0 = time()
        sopt = calc_sopt(FG,Xstar)
        cstar, ϵstar, cstar_grad2, ϵstar_grad2, qopt = QP_diff_solver(FG,Xstar,Popt,Hopt,Lopt,ropt,gopt,sopt)
        obj = calc_obj(T,cstar,Popt; time_penalty=true, Tw=Tw)
        Lag = calc_Lag(T,Popt,Ropt,qopt,cstar,ϵstar; time_penalty=true, Tw=Tw)
        # grad_time0 = time()
        Lag_grad2 = calc_Lag_grad(T,Popt,Ropt,qopt,cstar,ϵstar,cstar_grad2,ϵstar_grad2; time_penalty=true,Tw=Tw)
        Lag_grad2 = Lag_grad2 / norm(Lag_grad2)
        grad_time1 = time()
        grad_time = grad_time1 - grad_time0
        push!(grad_times, grad_time)
        push!(c_list, cstar)
        push!(obj_list, obj)
        push!(Lag_list, Lag)
        push!(Lag_grad_list, Lag_grad2)
        push!(T_list, T)
        @show obj
        @show Lag
        if abs(Lag - Lag_opt) <= Lag_tol
            @show Tstar = T
            # break
        else
            if Lag < Lag_opt
                @show Lag_opt = Lag
                Tstar = T
            end
        end
        @show Lag_grad2
        T -= η * Lag_grad2
        k += 1
        time1 = time()
        @show time1 - time0
    end
    Tstar = T_list[end]
    bilevel_time1 = time()
    bilevel_soltime = bilevel_time1 - bilevel_time0
    return cstar, Xstar, Tstar, c_list, obj_list, Lag_list, Lag_grad_list, T_list, sol_times, bilevel_soltime, grad_times, para_times
end

function BiLevelOpt_new(T0, Tw, α, β, FG; alpha=0.05, outer_iter=5, inner_iter=500)
    T = T0
    Lag_tol = 1e-3
    Lag = 0
    Lag_opt = Inf
    obj = 0
    k = 1
    k_max = outer_iter
    cstar = nothing
    Xstar = nothing
    λstar = nothing
    νstar = nothing
    Tstar = nothing
    sopt = nothing
    last_X = nothing
    last_λ = nothing
    c_list = []
    Lag_list = []
    Lag_grad_list = []
    T_list = []
    obj_list = []
    grad_times = []
    para_times = []
    sol_times = []
    bilevel_time0 = time()
    while k <= k_max
        time0 = time()
        para_time0 = time()
        Eopt, eopt, Popt, Hopt, ropt, Lopt, gopt, Ropt = calc_Para(T)
        para_time1 = time()
        para_time = para_time1 - para_time0
        push!(para_times, para_time)
        @show k
        if k == 1
            Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=500, warm_start=false)
        else
            Xstar, λstar, X_list, λ_list, sol_time = PDHG_GPU_solver(Eopt, eopt, α, β, FG; max_iter=inner_iter, warm_start=true, last_X=last_X, last_λ=last_λ)
        end
        push!(sol_times, sol_time)
        last_X = Xstar
        last_λ = λstar
        grad_time0 = time()
        sopt = calc_sopt(FG,Xstar)
        cstar = calc_cstar(Hopt,ropt,Lopt,gopt,sopt)
        λstar = calc_λstar(λstar)
        νstar = calc_νstar(cstar,λstar,Popt,Hopt,Lopt)

        kkt_01 = 2Popt * cstar + Lopt' * λstar + Hopt' * νstar
        kkt_02 = Lopt * cstar - gopt - sopt
        kkt_03 = Hopt * cstar - ropt
        
        @show T
        @show norm(kkt_01)
        @show norm(kkt_02)
        @show norm(kkt_03)


        obj = calc_obj(T,cstar,Popt; time_penalty=true, Tw=Tw)
        Lag = calc_Lag_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=true, Tw=Tw)
        Lag_grad2 = calc_Lag_grad_new(T,Popt,Hopt,ropt,Lopt,gopt,sopt,cstar,λstar,νstar; time_penalty=true,Tw=Tw)
        grad_AG = Lag_grad2 / norm(Lag_grad2)
        grad_time1 = time()
        grad_time = grad_time1 - grad_time0
        push!(grad_times, grad_time)
        push!(c_list, cstar)
        push!(obj_list, obj)
        push!(Lag_list, Lag)
        push!(Lag_grad_list, grad_AG)
        push!(T_list, T)
        @show obj
        @show Lag
        if abs(Lag - Lag_opt) <= Lag_tol
            @show Tstar = T
            # break
        else
            if Lag < Lag_opt
                @show Lag_opt = Lag
                Tstar = T
            end
        end
        @show Lag_grad2
        T -= alpha * grad_AG
        k += 1
        time1 = time()
        @show time1 - time0
    end
    Tstar = T_list[end]

    bilevel_time1 = time()
    bilevel_soltime = bilevel_time1 - bilevel_time0
    return cstar, Xstar, λstar, νstar, Tstar, sopt, c_list, obj_list, Lag_list, Lag_grad_list, T_list, sol_times, bilevel_soltime, grad_times, para_times
end

function BilevelOpt_FDAG(T0, Tw, α, β, FG; warm_start=false, outer_iter=5, inner_iter=500, alpha=0.05, h=1e-1, Obj_Lag=nothing, FD_AG=nothing)
    function get_gradient_fd(obj0::Float64, Lag0::Float64, Time0::Vector{Float64}, Xstar, λstar, h=1e-1)
        grad_time0 = time()
        grad_obj = zeros(size(Time0))
        grad_Lag = zeros(size(Time0))
        grad = zeros(size(Time0))
        origin_time = Time0
        for i in eachindex(Time0)
            try_time = deepcopy(origin_time)
            try_time[i] += h
            data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, Xstar, λstar, νstar, gradAG_time, para_time = TrajOpt_space_fd(try_time, Tw, α, β, FG;max_iter=inner_iter, warm_start=false, last_X=Xstar, last_λ=λstar)
            grad_obj[i] = (obj - obj0) / h
            grad_Lag[i] = (Lag - Lag0) / h
        end
        if Obj_Lag == :Obj
            grad = grad_obj
        elseif Obj_Lag == :Lag
            grad = grad_Lag
        end
        grad += Tw
        grad_time1 = time()
        grad_time = grad_time1 - grad_time0
        return grad, grad_time
    end
    
    function gradient_descend_fixedstep(Time0, cstar, Xstar, λstar, νstar, obj, objt, Lag, Lag_grad2, gradAG_time, sol_time, para_time; alpha::Float64=0.05, h::Float64=1e-2)
        Times = []
        Objs = []
        Objts = []
        cstars = []
        Xstars = []
        λstars = []
        νstars = []
        gradFD_times = []
        gradAG_times = []
        sol_times = []
        para_times = []
        Lags = []
        Lagts = []
        grads_FD = []
        grads_AG = []
        # kkt01s = []
        # kkt02s = []
        # kkt03s = []
        # lag_λs = []
        # lag_νs = []
        Lagt = Lag + Tw'*Time0
        for i in 1:outer_iter
            @show i

            if FD_AG == :FD
                grad_FD, gradFD_time = get_gradient_fd(obj, Lag, Time0, Xstar, λstar, h)
                grad_FD = grad_FD / norm(grad_FD)
                push!(gradFD_times, gradFD_time)
                push!(grads_FD, grad_FD)
                ΔT = - grad_FD
            elseif FD_AG == :AG
                grad_AG = Lag_grad2 / norm(Lag_grad2)
                ΔT = - grad_AG
                push!(gradAG_times, gradAG_time)
                push!(grads_AG, grad_AG)
            end

            push!(Times, Time0)
            push!(Objs, obj)
            push!(Objts, objt)
            push!(Lags, Lag)
            push!(Lagts, Lagt)
            push!(cstars, cstar)
            push!(Xstars, Xstar)
            push!(λstars, λstar)
            push!(νstars, νstar)
            push!(sol_times, sol_time)
            push!(para_times, para_time)
            

            candid_time = Time0 + alpha * ΔT
            data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, Xstar, λstar, νstar, gradAG_time, para_time = TrajOpt_space_fd(candid_time, Tw, α, β, FG;max_iter=inner_iter, warm_start=warm_start, last_X=Xstar, last_λ=λstar)

            Time0 = candid_time
            objt = obj + Tw'*Time0
            Lagt = Lag + Tw'*Time0

            # push!(kkt01s, kkt01_val)
            # push!(kkt02s, kkt02_val)
            # push!(kkt03s, kkt03_val)
            # push!(lag_λs, lag_λ)
            # push!(lag_νs, lag_ν)
        end
        return Times, cstars, Xstars, λstars, νstars, Objs, Objts, Lags, Lagts, gradFD_times, gradAG_times, grads_FD, grads_AG, sol_times, para_times
    end
    bilevel_time0 = time()
    data, X_list, λ_list, sol_time, obj, Lag, Lag_grad2, cstar, Xstar, λstar, νstar, gradAG_time, para_time = TrajOpt_space_fd(T0, Tw, α, β, FG;max_iter=500, warm_start=false) 
    objt = obj + sum(Tw.*T0)
    Times, cstars, Xstars, λstars, νstars, Objs, Objts, Lags, Lagts, gradFD_times, gradAG_times, grads_FD, grads_AG, sol_times, para_times = gradient_descend_fixedstep(T0, cstar, Xstar, λstar, νstar, obj, objt, Lag, Lag_grad2, gradAG_time, sol_time, para_time; alpha=alpha, h=h)
    bilevel_time1 = time()
    bilevel_soltime = bilevel_time1 - bilevel_time0
    return cstars, Xstars, λstars, νstars, Times, Objs, Objts, Lags, Lagts, bilevel_soltime, gradFD_times, gradAG_times, grads_FD, grads_AG, sol_times, para_times
end

function post_Bilevel(cstar, Tstar)
    data = calc_traj(Tstar, cstar; dt=0.02)
    return data
end


