using LinearAlgebra
using Plots

struct Cons_Corrs
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    D::Matrix{Float64}
end
struct Cons_Limits
    limits_xplus::Cons_Corrs
    limits_yplus::Cons_Corrs
    limits_xminus::Cons_Corrs
    limits_yminus::Cons_Corrs
end
struct Cons_Obs
    obs_xplus::Cons_Corrs
    obs_yplus::Cons_Corrs
    obs_xminus::Cons_Corrs
    obs_yminus::Cons_Corrs
end
struct Cons_Init
    pos::Matrix
    vel::Matrix
    acc::Matrix
end

function gen_env(route::Matrix)
    L = length(route[:,1])
    width = 0.9
    car_width = 0.6
    corridors = zeros(L-1,4)
    obstacles = zeros(L-1,2,4)
    left_side = zeros(L,2)
    right_side = zeros(L,2)
    for i in 1:L-1
        xmin = min(route[i,1] - width, route[i+1,1] - width) 
        xmax = max(route[i,1] + width, route[i+1,1] + width) 
        ymin = min(route[i,2] - width, route[i+1,2] - width) 
        ymax = max(route[i,2] + width, route[i+1,2] + width) 
        corridors[i,1] = xmin + car_width / 2
        corridors[i,2] = xmax - car_width / 2
        corridors[i,3] = ymin + car_width / 2
        corridors[i,4] = ymax - car_width / 2
        if i == 1
            if route[i+1,1] - route[i,1] > 0
                corridors[i,1] -= 1.0
            elseif route[i+1,2] - route[i,2] > 0
                corridors[i,3] -= 1.0
            elseif route[i+1,1] - route[i,1] < 0
                corridors[i,2] += 1.0
            elseif route[i+1,2] - route[i,2] < 0
                corridors[i,4] += 1.0
            end
        end
        if i == L-1
            if route[i+1,1] - route[i,1] > 0
                corridors[i,2] += 1.0 * 2
            elseif route[i+1,2] - route[i,2] > 0
                corridors[i,4] += 1.0 * 2
            elseif route[i+1,1] - route[i,1] < 0
                corridors[i,1] -= 1.0 * 2
            elseif route[i+1,2] - route[i,2] < 0
                corridors[i,3] -= 1.0 * 2
            end
        end
    end
    if route[2,1] - route[1,1] > 0
        right_side[1,:] = route[1,:] + [-width, -width]
        left_side[1,:] = route[1,:] + [-width, +width]
    elseif route[2,2] - route[1,2] > 0
        right_side[1,:] = route[1,:] + [+width, -width]
        left_side[1,:] = route[1,:] + [-width, -width]
    elseif route[2,1] - route[1,1] < 0
        right_side[1,:] = route[1,:] + [+width, +width]
        left_side[1,:] = route[1,:] + [+width, -width]
    elseif route[2,2] - route[1,2] < 0
        right_side[1,:] = route[1,:] + [-width, +width]
        left_side[1,:] = route[1,:] + [+width, +width]
    end

    if route[L,1] - route[L-1,1] > 0
        right_side[L,:] = route[L,:] + [+width, -width]
        left_side[L,:] = route[L,:] + [+width, +width]
    elseif route[L,2] - route[L-1,2] > 0
        right_side[L,:] = route[L,:] + [+width, +width]
        left_side[L,:] = route[L,:] + [-width, +width]
    elseif route[L,1] - route[L-1,1] < 0
        right_side[L,:] = route[L,:] + [-width, +width]
        left_side[L,:] = route[L,:] + [-width, -width]
    elseif route[L,2] - route[L-1,2] < 0
        right_side[L,:] = route[L,:] + [-width, -width]
        left_side[L,:] = route[L,:] + [+width, -width]
    end

    for i in 2:L-1
        vec2 = [sign.(route[i+1,:] - route[i,:])..., 0]
        vec1 = [sign.(route[i,:] - route[i-1,:])..., 0]
        sign_vec = sign(cross(vec2, vec1)[3])
        right_side[i,:] = route[i,:] + width * sign_vec * sign.(route[i+1,:] - route[i,:]) + width * sign_vec * sign.(route[i-1,:] - route[i,:])
        left_side[i,:] = route[i,:] - width * sign_vec * sign.(route[i+1,:] - route[i,:]) - width * sign_vec * sign.(route[i-1,:] - route[i,:])
    end

    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    # obstacles_shape = []
    # left side
    for i in 1:L-1
        if left_side[i+1,1] - left_side[i,1] > 0
            obs_shape = rectangle(left_side[i+1,1] - left_side[i,1], 1.0, left_side[i,1], left_side[i,2])
        elseif left_side[i+1,2] - left_side[i,2] > 0
            obs_shape = rectangle(-1.0, left_side[i+1,2] - left_side[i,2], left_side[i,1], left_side[i,2])
        elseif left_side[i+1,1] - left_side[i,1] < 0
            obs_shape = rectangle(left_side[i+1,1] - left_side[i,1], -1.0, left_side[i,1], left_side[i,2])
        elseif left_side[i+1,2] - left_side[i,2] < 0
            obs_shape = rectangle(1.0, left_side[i+1,2] - left_side[i,2], left_side[i,1], left_side[i,2])
        end
        obstacles[i,1,1] = minimum(obs_shape.x[:])
        obstacles[i,1,2] = maximum(obs_shape.x[:])
        obstacles[i,1,3] = minimum(obs_shape.y[:])
        obstacles[i,1,4] = maximum(obs_shape.y[:])
        # push!(obstacles_shape, obs_shape)
    end
    # right side
    for i in 1:L-1
        if right_side[i+1,1] - right_side[i,1] > 0
            obs_shape = rectangle(right_side[i+1,1] - right_side[i,1], -1.0, right_side[i,1], right_side[i,2])
        elseif right_side[i+1,2] - right_side[i,2] > 0
            obs_shape = rectangle(1.0, right_side[i+1,2] - right_side[i,2], right_side[i,1], right_side[i,2])
        elseif right_side[i+1,1] - right_side[i,1] < 0
            obs_shape = rectangle(right_side[i+1,1] - right_side[i,1], 1.0, right_side[i,1], right_side[i,2])
        elseif right_side[i+1,2] - right_side[i,2] < 0
            obs_shape = rectangle(-1.0, right_side[i+1,2] - right_side[i,2], right_side[i,1], right_side[i,2])
        end
        obstacles[i,2,1] = minimum(obs_shape.x[:])
        obstacles[i,2,2] = maximum(obs_shape.x[:])
        obstacles[i,2,3] = minimum(obs_shape.y[:])
        obstacles[i,2,4] = maximum(obs_shape.y[:])
        # push!(obstacles_shape, obs_shape)
    end
    border = zeros(4)
    border[1] = minimum(route[:,1])-2.5
    border[2] = maximum(route[:,1])+2.5
    border[3] = minimum(route[:,2])-2.5
    border[4] = maximum(route[:,2])+2.5

    return corridors, obstacles, border
end

function gen_cons(route::Matrix, corridors::Matrix, cons_limits::Cons_Limits, cons_obs::Cons_Obs)
    limits_list = []
    obs_list = []
    L = length(route[:,1])
    for i in 1:L-1
        xmin = corridors[i,1]
        xmax = corridors[i,2]
        ymin = corridors[i,3]
        ymax = corridors[i,4]
        if route[i+1,1] - route[i,1] > 0
            push!(limits_list, cons_limits.limits_xplus)
            lx = 3
            ly = 2
            kx = (xmax - xmin) / (2*lx)
            bx = (xmax + xmin) / 2
            ky = (ymax - ymin) / (2*ly)
            by = (ymax + ymin) / 2
            H = zeros(4,4)
            G = zeros(4,1)
            H[1,1] = 1/kx
            H[2,2] = 1/ky
            H[3,3] = 1/kx
            H[4,4] = 1/ky
            G[1,1] = -bx/kx
            G[2,1] = -by/ky
            A = cons_obs.obs_xplus.A * H
            B = cons_obs.obs_xplus.B - cons_obs.obs_xplus.A * G
            C = cons_obs.obs_xplus.C * H 
            D = cons_obs.obs_xplus.C * G + cons_obs.obs_xplus.D
            push!(obs_list, Cons_Corrs(A,B,C,D))
        elseif route[i+1,2] - route[i,2] > 0
            push!(limits_list, cons_limits.limits_yplus)
            lx = 2
            ly = 3
            kx = (xmax - xmin) / (2*lx)
            bx = (xmax + xmin) / 2
            ky = (ymax - ymin) / (2*ly)
            by = (ymax + ymin) / 2
            H = zeros(4,4)
            G = zeros(4,1)
            H[1,1] = 1/kx
            H[2,2] = 1/ky
            H[3,3] = 1/kx
            H[4,4] = 1/ky
            G[1,1] = -bx/kx
            G[2,1] = -by/ky
            A = cons_obs.obs_yplus.A * H
            B = cons_obs.obs_yplus.B - cons_obs.obs_yplus.A * G
            C = cons_obs.obs_yplus.C * H 
            D = cons_obs.obs_yplus.C * G + cons_obs.obs_yplus.D
            push!(obs_list, Cons_Corrs(A,B,C,D))
        elseif route[i+1,1] - route[i,1] < 0
            push!(limits_list, cons_limits.limits_xminus)
            lx = 3
            ly = 2
            kx = (xmax - xmin) / (2*lx)
            bx = (xmax + xmin) / 2
            ky = (ymax - ymin) / (2*ly)
            by = (ymax + ymin) / 2
            H = zeros(4,4)
            G = zeros(4,1)
            H[1,1] = 1/kx
            H[2,2] = 1/ky
            H[3,3] = 1/kx
            H[4,4] = 1/ky
            G[1,1] = -bx/kx
            G[2,1] = -by/ky
            A = cons_obs.obs_xminus.A * H
            B = cons_obs.obs_xminus.B - cons_obs.obs_xminus.A * G
            C = cons_obs.obs_xminus.C * H 
            D = cons_obs.obs_xminus.C * G + cons_obs.obs_xminus.D
            push!(obs_list, Cons_Corrs(A,B,C,D))
        elseif route[i+1,2] - route[i,2] < 0
            push!(limits_list, cons_limits.limits_yminus)
            lx = 2
            ly = 3
            kx = (xmax - xmin) / (2*lx)
            bx = (xmax + xmin) / 2
            ky = (ymax - ymin) / (2*ly)
            by = (ymax + ymin) / 2
            H = zeros(4,4)
            G = zeros(4,1)
            H[1,1] = 1/kx
            H[2,2] = 1/ky
            H[3,3] = 1/kx
            H[4,4] = 1/ky
            G[1,1] = -bx/kx
            G[2,1] = -by/ky
            A = cons_obs.obs_yminus.A * H
            B = cons_obs.obs_yminus.B - cons_obs.obs_yminus.A * G
            C = cons_obs.obs_yminus.C * H 
            D = cons_obs.obs_yminus.C * G + cons_obs.obs_yminus.D
            push!(obs_list, Cons_Corrs(A,B,C,D))
        end
    end
    return limits_list, obs_list
end

function gen_cons_car(route::Matrix, corridors::Matrix, cons_limits::Cons_Limits, cons_obs::Cons_Obs)
    limits_list = []
    obs_list = []
    L = length(route[:,1])
    for i in 1:L-1
        xmin = corridors[i,1]
        xmax = corridors[i,2]
        ymin = corridors[i,3]
        ymax = corridors[i,4]
        if route[i+1,1] - route[i,1] > 0
            push!(limits_list, cons_limits.limits_xplus)
            lx = 3
            ly = 2
            kx = (xmax - xmin) / (2*lx)
            bx = (xmax + xmin) / 2
            ky = (ymax - ymin) / (2*ly)
            by = (ymax + ymin) / 2
            H = zeros(2,2)
            G = zeros(2,1)
            H[1,1] = 1/kx
            H[2,2] = 1/ky
            G[1,1] = -bx/kx
            G[2,1] = -by/ky
            A = cons_obs.obs_xplus.A * H
            B = cons_obs.obs_xplus.B - cons_obs.obs_xplus.A * G
            C = cons_obs.obs_xplus.C * H 
            D = cons_obs.obs_xplus.C * G + cons_obs.obs_xplus.D
            push!(obs_list, Cons_Corrs(A,B,C,D))
        elseif route[i+1,2] - route[i,2] > 0
            push!(limits_list, cons_limits.limits_yplus)
            lx = 2
            ly = 3
            kx = (xmax - xmin) / (2*lx)
            bx = (xmax + xmin) / 2
            ky = (ymax - ymin) / (2*ly)
            by = (ymax + ymin) / 2
            H = zeros(2,2)
            G = zeros(2,1)
            H[1,1] = 1/kx
            H[2,2] = 1/ky
            G[1,1] = -bx/kx
            G[2,1] = -by/ky
            A = cons_obs.obs_yplus.A * H
            B = cons_obs.obs_yplus.B - cons_obs.obs_yplus.A * G
            C = cons_obs.obs_yplus.C * H 
            D = cons_obs.obs_yplus.C * G + cons_obs.obs_yplus.D
            push!(obs_list, Cons_Corrs(A,B,C,D))
        elseif route[i+1,1] - route[i,1] < 0
            push!(limits_list, cons_limits.limits_xminus)
            lx = 3
            ly = 2
            kx = (xmax - xmin) / (2*lx)
            bx = (xmax + xmin) / 2
            ky = (ymax - ymin) / (2*ly)
            by = (ymax + ymin) / 2
            H = zeros(2,2)
            G = zeros(2,1)
            H[1,1] = 1/kx
            H[2,2] = 1/ky
            G[1,1] = -bx/kx
            G[2,1] = -by/ky
            A = cons_obs.obs_xminus.A * H
            B = cons_obs.obs_xminus.B - cons_obs.obs_xminus.A * G
            C = cons_obs.obs_xminus.C * H 
            D = cons_obs.obs_xminus.C * G + cons_obs.obs_xminus.D
            push!(obs_list, Cons_Corrs(A,B,C,D))
        elseif route[i+1,2] - route[i,2] < 0
            push!(limits_list, cons_limits.limits_yminus)
            lx = 2
            ly = 3
            kx = (xmax - xmin) / (2*lx)
            bx = (xmax + xmin) / 2
            ky = (ymax - ymin) / (2*ly)
            by = (ymax + ymin) / 2
            H = zeros(2,2)
            G = zeros(2,1)
            H[1,1] = 1/kx
            H[2,2] = 1/ky
            G[1,1] = -bx/kx
            G[2,1] = -by/ky
            A = cons_obs.obs_yminus.A * H
            B = cons_obs.obs_yminus.B - cons_obs.obs_yminus.A * G
            C = cons_obs.obs_yminus.C * H 
            D = cons_obs.obs_yminus.C * G + cons_obs.obs_yminus.D
            push!(obs_list, Cons_Corrs(A,B,C,D))
        end
    end
    return limits_list, obs_list
end

function gen_init(route::Matrix)
    if route[2,1] - route[1,1] > 0
        pos = [route[1,:]' - [1, 0]'; route[end,:]']
    elseif route[2,2] - route[1,2] > 0
        pos = [route[1,:]' - [0, 1]'; route[end,:]']
    elseif route[2,1] - route[1,1] < 0
        pos = [route[1,:]' + [1, 0]'; route[end,:]']
    elseif route[2,2] - route[1,2] < 0
        pos = [route[1,:]' + [0, 1]'; route[end,:]']
    end

    # if route[end,1] - route[end-1,1] > 0
    #     pos = [route[1,:]'; route[end,:]' + [1, 0]']
    # elseif route[end,1] - route[end-1,1] > 0
    #     pos = [route[1,:]'; route[end,:]' + [0, 1]']
    # elseif route[end,1] - route[end-1,1] < 0
    #     pos = [route[1,:]'; route[end,:]'  - [1, 0]']
    # elseif route[end,1] - route[end-1,1] < 0
    #     pos = [route[1,:]'; route[end,:]' - [0, 1]']
    # end

    vel = [(route[2,:]-route[1,:])'/norm(route[2,:]-route[1,:]); (route[end,:]-route[end-1,:])'/norm(route[end,:]-route[end-1,:])]
    acc = [(route[2,:]-route[1,:])'/norm(route[2,:]-route[1,:]); (route[end,:]-route[end-1,:])'/norm(route[end,:]-route[end-1,:])] * 1.0
    return Cons_Init(pos, vel, acc)
end

function plot_env(route::Matrix, corridors::Matrix, obstacles::Array, cons_init::Cons_Init, border::Array)
    L = length(route[:,1])
    start_x = route[1,1]
    start_y = route[1,2]
    goal_x = route[L,1]
    goal_y = route[L,2]
    # fig_env = plot(aspect_ratio=:equal, grid=false, axis=false, legend=false, size=(1600,900))
    fig_env = plot(aspect_ratio=:equal)
    # fig_env = plot([cons_init.pos[1,1]], [cons_init.pos[1,2]], seriestype=:scatter, aspect_ratio=:equal, markershape=:star5, markersize=3, color=:red, label="")
    # plot!(fig_env, [cons_init.pos[2,1]], [cons_init.pos[2,2]], seriestype=:scatter, aspect_ratio=:equal, markshape=:circle, markersize=3, color=:orange, label="")
    # xlims!(fig_env, border[1], border[2])
    # ylims!(fig_env, border[3], border[4])

    rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
    for i in axes(obstacles,1), j in axes(obstacles,2)
        obs = rectangle(obstacles[i,j,2]-obstacles[i,j,1], obstacles[i,j,4]-obstacles[i,j,3], obstacles[i,j,1], obstacles[i,j,3])
        plot!(fig_env, obs, fillcolor=:gray, label="", opacity=1.0, framestyle=:box, linecolor=:gray)
    end
    return fig_env
end

