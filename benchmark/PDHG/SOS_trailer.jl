using DynamicPolynomials
using SumOfSquares
using JuMP
using COPT
using Mosek, MosekTools
using SCS, COSMO, ProxSDP, CSDP
using LinearAlgebra
using Plots
using CSV, DataFrames

@polyvar t 
trajopt_set = TrajOpt_Set(d=7, M=2, N=length(route[:,1])-1)

function trajopt_space(Time::Vector{Float64}, trajopt_set::TrajOpt_Set, cons_init::Cons_Init, limits_list::Vector, obs_list::Vector; solver_name::Symbol)
    pre_time0 = time()
    if solver_name == :COPT
        solver = optimizer_with_attributes(COPT.ConeOptimizer)
    elseif solver_name == :Mosek
        solver = optimizer_with_attributes(Mosek.Optimizer)
    elseif solver_name == :COSMO
        solver = optimizer_with_attributes(COSMO.Optimizer)
    elseif solver_name == :SCS
        solver = optimizer_with_attributes(SCS.Optimizer)
    elseif solver_name == :SDPA
        solver = optimizer_with_attributes(SDPA.Optimizer)
    elseif solver_name == :CSDP
        solver = optimizer_with_attributes(CSDP.Optimizer)
    elseif solver_name == :ProxSDP
        solver = optimizer_with_attributes(ProxSDP.Optimizer)
    end
    model = SOSModel(solver)
    set_silent(model)
    # set_optimizer_attribute(model, "TimeLimit", 0.3)
    # set_optimizer_attribute(model, "MSK_IPAR_INTPNT_MAX_ITERATIONS", iter)
    d = trajopt_set.d 
    M = trajopt_set.M 
    N = trajopt_set.N
    @variable(model, coef[i=1:(d+1),j=1:M,k=1:N])
    poly_basis = monomials([t], 0:d)
    p = [coef[:,i,j]'*poly_basis for i in 1:M, j in 1:N]

    # objective
    order = 3 # order of  derivation
    hilbert = zeros(d+1-order, d+1-order, N)
    # matrix
    size_hilbert = d+1-order
    h = []

    new_coef = 1.0 * model[:coef]
    for i in 1:(d+1)
        if (d+1)-i-order >= 0
            new_coef[i,:,:] = new_coef[i,:,:] * factorial((d+1)-i, (d+1)-i-order)
        else
            new_coef[i,:,:] = new_coef[i,:,:] * 0
        end
    end
    for n in 1:N
        for i in 1:(2*size_hilbert-1)
            for j in 1:size_hilbert, k in 1:size_hilbert
                if j + k == i + 1
                    hilbert[j,k,n] = Time[n]^(2*size_hilbert-i)/(2*size_hilbert-i)                    
                end
            end
        end
        hilbert_L = cholesky(hilbert[:,:,n]).L 
        hc = Matrix(hilbert_L') * new_coef[1:d+1-order,:,n]
        for i in 1:M
            push!(h, hc[:,i])
        end
    end
    L = length(h[1])
    @variable(model, α[1:M*N])
    @objective(model, Min, sum(α))
    mat = Array{Any}(undef, L+1, L+1, M*N)
    for i in 1:M*N
        new_mat = α[i]*Matrix(I,L+1,L+1)
        new_mat[2:L+1,1] = h[i]
        new_mat[1,2:L+1] = h[i]'
        mat[:,:,i] = new_mat
        @constraint(model, new_mat in PSDCone())
    end

    # equality constraint
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    D3p = differentiate.(p,t,3)
    @constraint(model, p[1,1](t=>0.0) == cons_init.pos[1,1])
    @constraint(model, p[2,1](t=>0.0) == cons_init.pos[1,2])
    @constraint(model, p[1,end](t=>Time[end]) == cons_init.pos[2,1])
    @constraint(model, p[2,end](t=>Time[end]) == cons_init.pos[2,2])
    @constraint(model, D1p[1,1](t=>0.0) == cons_init.vel[1,1])
    @constraint(model, D1p[2,1](t=>0.0) == cons_init.vel[1,2])
    @constraint(model, D1p[1,end](t=>Time[end]) == cons_init.vel[2,1])
    @constraint(model, D1p[2,end](t=>Time[end]) == cons_init.vel[2,2])
    # @constraint(model, D2p[1,1](t=>0.0) == cons_init.acc[1,1])
    # @constraint(model, D2p[2,1](t=>0.0) == cons_init.acc[1,2])
    # @constraint(model, D2p[1,end](t=>Time[end]) == cons_init.acc[2,1])
    # @constraint(model, D2p[2,end](t=>Time[end]) == cons_init.acc[2,2])
    # continuous
    for i in 1:N-1
        @constraint(model, p[1,i](t=>Time[i]) == p[1,i+1](t=>0.0))
        @constraint(model, p[2,i](t=>Time[i]) == p[2,i+1](t=>0.0))
        @constraint(model, D1p[1,i](t=>Time[i]) == D1p[1,i+1](t=>0.0))
        @constraint(model, D1p[2,i](t=>Time[i]) == D1p[2,i+1](t=>0.0))
        @constraint(model, D2p[1,i](t=>Time[i]) == D2p[1,i+1](t=>0.0))
        @constraint(model, D2p[2,i](t=>Time[i]) == D2p[2,i+1](t=>0.0))
        @constraint(model, D3p[1,i](t=>Time[i]) == D3p[1,i+1](t=>0.0))
        @constraint(model, D3p[2,i](t=>Time[i]) == D3p[2,i+1](t=>0.0))
    end

    # inequality constraint
    # space
    if d%2 == 0
        Xf = monomials([t], 0:d)
        Xg = monomials([t], 0:d-2)
    else
        Xf = monomials([t], 0:d-1)
        Xg = monomials([t], 0:d-1)
    end
    L1 = maximum([length(obs_list[i].B) for i in eachindex(obs_list)])
    @variable(model, f[1:N,1:L1], Poly(Xf))
    @variable(model, g[1:N,1:L1], Poly(Xg))

    for i in 1:N
        X = [p[1,i], p[2,i], D1p[1,i], D1p[2,i]]
        for j in eachindex(obs_list[i].B)
            @constraint(model, f[i,j] >= 0.0)
            @constraint(model, g[i,j] >= 0.0)
            if d%2 == 0
                @constraint(model, (obs_list[i].B[j] - obs_list[i].A[j,:]'*X) == f[i,j] - (t-0.0)*(t-Time[i])*g[i,j])
            else
                @constraint(model, (obs_list[i].B[j] - obs_list[i].A[j,:]'*X) == (t-0.0)*f[i,j] - (t-Time[i])*g[i,j])
            end
        end
    end

    # dynamic limit
    if d%2 == 0
        Xcf = monomials([t], 0:d-1)
        Xcg = monomials([t], 0:d-3)
    else
        Xcf = monomials([t], 0:d-2)
        Xcg = monomials([t], 0:d-2)
    end
    L2 = maximum([length(limits_list[i].B) for i in eachindex(limits_list)])
    @variable(model, fc[1:N,1:L2], Poly(Xcf))
    @variable(model, gc[1:N,1:L2], Poly(Xcg))

    for i in 1:N
        X = [D1p[1,i], D1p[2,i], D2p[1,i], D2p[2,i], D3p[1,i], D3p[2,i]]
        for j in eachindex(limits_list[i].B)
            @constraint(model, fc[i,j] >= 0.0)
            @constraint(model, gc[i,j] >= 0.0)
            if d%2 == 0
                @constraint(model, (limits_list[i].B[j] - limits_list[i].A[j,:]'*X) == fc[i,j] - (t-0.0)*(t-Time[i])*gc[i,j])
            else
                @constraint(model, (limits_list[i].B[j] - limits_list[i].A[j,:]'*X) == (t-0.0)*fc[i,j] - (t-Time[i])*gc[i,j])
            end
        end
    end
    pre_time1 = time()
    pre_time = pre_time1 - pre_time0
    JuMP.optimize!(model)
    return value.(p), model, pre_time
end

function trajopt_time(Time0::Vector{Float64}, trajopt_set::TrajOpt_Set, cons_init::Cons_Init, limits_list::Vector, obs_list::Vector; solver_name=:COPT, outer_iter=5, tweight=nothing)
    function get_gradient_fd(obj0::Float64, Time0::Vector{Float64}, h=1e-3)
        grad_time0 = time()
        grad = zeros(size(Time0))
        origin_time = Time0
        for i in eachindex(Time0)
            try_time = deepcopy(origin_time)
            try_time[i] += h
            _, mdl = trajopt_space(try_time, trajopt_set, cons_init, limits_list, obs_list; solver_name=solver_name)
            obj = objective_value(mdl)
            grad[i] = (obj - obj0) / h
        end
        grad += tweight 
        grad_time1 = time()
        grad_time = grad_time1 - grad_time0
        return grad, grad_time
    end
    
    function get_gradient_mellinger(obj0::Float64, Time0::Vector{Float64}, h=1e-3)
        grad_time0 = time()
        grad = zeros(size(Time0))
        origin_time = Time0
        m = length(Time0)
        for i in eachindex(Time0)
            gi = -1/(m-1) * ones(m)
            gi[i] = 1
            try_time = deepcopy(origin_time)
            try_time += h * gi
            _, mdl = trajopt_space(try_time, trajopt_set, cons_init, limits_list, obs_list; solver_name=solver_name)
            obj = objective_value(mdl)
            grad[i] = (obj - obj0) / h
        end
        grad += tweight 
        grad_time1 = time()
        grad_time = grad_time1 - grad_time0
        return grad, grad_time
    end

    function gradient_descend_fixedstep(;alpha0::Float64=0.05, h::Float64=1e-1,max_iter::Int64=20, grad_method::Symbol=:fd)
        Times = []
        Objs = []
        Objts = []
        ps = []
        grad_times = []
        grads = []
        for i in 1:max_iter
            @show i
            if grad_method == :fd
                grad, grad_time = get_gradient_fd(obj0, Time0, h)
            elseif grad_method == :mel 
                grad, grad_time = get_gradient_mellinger(obj0, Time0, h)
            else
                println("No this grad method")
            end
            push!(grad_times, grad_time)
            ∇F = grad / norm(grad)
            ΔT = - ∇F
            push!(grads, ∇F)
            
            candid_time = Time0 + alpha0 * ΔT
            objf = obj0
            objft = obj0t
            p, mdl = trajopt_space(candid_time, trajopt_set, cons_init, limits_list, obs_list; solver_name=solver_name)
            objf = objective_value(mdl)
            objft = objf + sum(tweight.*candid_time)
            push!(Times, Time0)
            push!(Objs, obj0)
            push!(Objts, obj0t)
            push!(ps, p)
            @show ∇F
            @show objft
            obj0 = objf
            obj0t = objft
            Time0 = candid_time
        end
        return Times, Objs, Objts, ps, grad_times, grads
    end
    
    function gradient_descend_backtracking(;alpha0::Float64=0.175, h::Float64=1e-3, c::Float64=0.2, tau::Float64=0.2, max_iter::Int64=20, j_iter::Int64=5, grad_method::Symbol=:fd)
        ```
        Parameters
        ----------
        alpha0: initial step length, 0.175 and 0.375 are found to be very good
        h: step size for finding gradient using forward differentiation
        c: the objective decrease parameter 
        tau: the step length shrink parameter
        max_iter: maximum iteration for gradient descent
        j_iter: maximum iteration for finding alpha 
        abs_tol: absolute objective tolerance
        rel_tol: relative objective tolerance
    
        Returns
        -------
        ```
        Times = []
        Objs = []
        Objts = []
        ps = []
        grad_times = []
        grads = []
        for i in 1:max_iter
            @show i
            if grad_method == :fd
                grad, grad_time = get_gradient_fd(obj0, Time0, h)
            elseif grad_method == :mel 
                grad, grad_time = get_gradient_mellinger(obj0, Time0, h)
            else
                println("No this grad method")
            end
            push!(grad_times, grad_time)
            push!(grads, grad)
            ∇F = grad / norm(grad)
            ΔT = - ∇F

            # use a maximum alpha that makes sure time are always positive
            alpha_max = maximum(- Time0 ./ ΔT) - 1e-6
            # @show alpha_max
            if alpha_max > 0
                alpha = min(alpha_max, alpha0)
            else
                alpha = alpha0
            end
            # find alpha
            candid_time = Time0
            objf = obj0
            objft = obj0t
            p = 0
            for j in 1:j_iter
                @show alpha
                candid_time = Time0 + alpha * ΔT
                p, mdl = trajopt_space(candid_time, trajopt_set, cons_init, limits_list, obs_list; solver_name=solver_name)
                objf = objective_value(mdl)
                objft = objf + sum(tweight.*candid_time)
                if objf < 0
                    alpha = tau * alpha
                    continue
                end
                if obj0t - objft >= - alpha * c * ( ∇F'*ΔT ) || obj0t - objft >= 0.1 * obj0 # either backtrack or decrease sufficiently
                    break
                else
                    alpha = tau * alpha
                end
            end
            # obj0 = objf
            # obj0t = objft
            # Time0 = candid_time
            push!(Times, Time0)
            push!(Objs, obj0)
            push!(Objts, obj0t)
            push!(ps, p)
            @show grad
            @show objft
            obj0 = objf
            obj0t = objft
            Time0 = candid_time
        end
        return Times, Objs, Objts, ps, grad_times, grads
    end
    bilevel_time0 = time()
    grad_method = :fd
    _, mdl = trajopt_space(Time0, trajopt_set, cons_init, limits_list, obs_list; solver_name=solver_name)
    obj0 = objective_value(mdl)
    obj0t = obj0 + sum(tweight.*Time0)
    # Times, Objs, Objts, ps, grad_times, grads = gradient_descend_backtracking(max_iter=outer_iter,grad_method=grad_method)
    Times, Objs, Objts, ps, grad_times, grads = gradient_descend_fixedstep(max_iter=outer_iter,grad_method=grad_method)
    bilevel_time1 = time()
    bilevel_soltime = bilevel_time1 - bilevel_time0
    return Times, Objs, Objts, ps, bilevel_soltime, grad_times, grads
end

function process_data(p::Matrix{Polynomial{true, Float64}}, Time::Vector{Float64}, trailer_body::TrailerBody)
    d0, d1, w = trailer_body.trailer_length, trailer_body.link_length, trailer_body.trailer_width 
    # d1 = 1.0
    data = []
    total_time = [0.0]
    dt = 0.01
    D1p = differentiate.(p,t,1)
    D2p = differentiate.(p,t,2)
    D3p = differentiate.(p,t,3)
    L = length(p[1,:])
    for n in 1:L
        if n != L
            ts = 0:dt:Time[n]-dt |> collect
        else
            ts = 0:dt:Time[n] |> collect
        end
        push!(total_time, total_time[end]+Time[n])
        for i in eachindex(ts)
            ti = ts[i] + total_time[n]
            x1 = p[1,n](t=>ts[i])
            y1 = p[2,n](t=>ts[i])
            vx1 = D1p[1,n](t=>ts[i])
            vy1 = D1p[2,n](t=>ts[i])
            v1 = sqrt(vx1^2+vy1^2)
            ax1 = D2p[1,n](t=>ts[i])
            ay1 = D2p[2,n](t=>ts[i])
            a1 = sqrt(ax1^2+ay1^2)
            jx1 = D3p[1,n](t=>ts[i])
            jy1 = D3p[2,n](t=>ts[i])
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
