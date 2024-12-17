using LinearAlgebra
using Plots
using Random
Random.seed!(123)
include("../visual/visual_trailer.jl")
include("../visual/visual_car.jl")
trailer_body = TrailerBody()

hard_ext_dis1 = trailer_body.trailer_width * 3.5
soft_ext_dis1 = trailer_body.trailer_width * 2.5
ext_dis2 = trailer_body.trailer_width * 1.0
pole_len = trailer_body.trailer_length

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

struct Point2D
    x::Float64
    y::Float64
end

# inside the corridor: a_i * x + b_y * y + c_i >= 0
struct Line
    a::Float64
    b::Float64
    c::Float64
end

struct Corridor
    hard_left_points::Vector{Point2D}
    hard_right_points::Vector{Point2D}
    hard_left_lines::Vector{Line}
    hard_right_lines::Vector{Line}
    soft_left_points::Vector{Point2D}
    soft_right_points::Vector{Point2D}
    soft_left_lines::Vector{Line}
    soft_right_lines::Vector{Line}
    yaw::Float64
end

function gen_lines(route::Matrix{Float64})
    ```
    according to the route, generate boundary lines
    ```
    L = size(route, 1) 
    lines_left_hard = Vector{Line}()
    lines_right_hard = Vector{Line}()
    lines_left_soft = Vector{Line}()
    lines_right_soft = Vector{Line}()
    for i in 1:L
        if i == 1
            x1, x2 = route[i,1], route[i+1,1]
            y1, y2 = route[i,2], route[i+1,2]
            a = x2 - x1;
            b = y2 - y1;
            c =  - x1 * a - y1 * b
            line1 = Line(a, b, c)
            push!(lines_left_hard, line1)
            push!(lines_left_soft, line1)
            push!(lines_right_hard, line1)
            push!(lines_right_soft, line1)
            a_mid = y2 - y1
            b_mid = x1 - x2
            c_mid = x2*y1 - x1*y2
            a_left = a_mid
            b_left = b_mid
            c_left_hard = c_mid + hard_ext_dis1*norm([a_mid, b_mid])
            c_left_soft = c_mid + soft_ext_dis1*norm([a_mid, b_mid])
            a_right = -a_mid
            b_right = -b_mid
            c_right_hard = -c_mid + hard_ext_dis1*norm([a_mid, b_mid])
            c_right_soft = -c_mid + soft_ext_dis1*norm([a_mid, b_mid])
            left_hard = Line(a_left, b_left, c_left_hard)
            left_soft = Line(a_left, b_left, c_left_soft)
            right_hard = Line(a_right, b_right, c_right_hard)
            right_soft = Line(a_right, b_right, c_right_soft)
            push!(lines_left_hard, left_hard)
            push!(lines_left_soft, left_soft)
            push!(lines_right_hard, right_hard)
            push!(lines_right_soft, right_soft)
        elseif i == L
            x1, x2 = route[i-1,1], route[i,1]
            y1, y2 = route[i-1,2], route[i,2]
            a = x2 - x1;
            b = y2 - y1;
            c =  - x2 * a - y2 * b
            lineL = Line(a, b, c)
            push!(lines_left_hard, lineL)
            push!(lines_left_soft, lineL)
            push!(lines_right_hard, lineL)
            push!(lines_right_soft, lineL)
        else
            x1, x2 = route[i,1], route[i+1,1]
            y1, y2 = route[i,2], route[i+1,2]
            a_mid = y2 - y1
            b_mid = x1 - x2
            c_mid = x2*y1 - x1*y2
            a_left = a_mid
            b_left = b_mid
            c_left_hard = c_mid + hard_ext_dis1*norm([a_mid, b_mid])
            c_left_soft = c_mid + soft_ext_dis1*norm([a_mid, b_mid])
            a_right = -a_mid
            b_right = -b_mid
            c_right_hard = -c_mid + hard_ext_dis1*norm([a_mid, b_mid])
            c_right_soft = -c_mid + soft_ext_dis1*norm([a_mid, b_mid])
            left_hard = Line(a_left, b_left, c_left_hard)
            left_soft = Line(a_left, b_left, c_left_soft)
            right_hard = Line(a_right, b_right, c_right_hard)
            right_soft = Line(a_right, b_right, c_right_soft)
            push!(lines_left_hard, left_hard)
            push!(lines_left_soft, left_soft)
            push!(lines_right_hard, right_hard)
            push!(lines_right_soft, right_soft)
        end
    end
    return lines_left_hard, lines_left_soft, lines_right_hard, lines_right_soft
end

function gen_points(lines::Vector{Line})
    points = Vector{Point2D}()
    for i in 1:size(lines,1)-1
        a1 = lines[i].a
        b1 = lines[i].b
        c1 = lines[i].c
        a2 = lines[i+1].a
        b2 = lines[i+1].b
        c2 = lines[i+1].c
        x = (-c1*b2 + c2*b1) / (a1*b2 - a2*b1)
        y = (-a1*c2 + a2*c1) / (a1*b2 - a2*b1)
        push!(points, Point2D(x, y))
    end
    return points
end

function gen_vertice(lines_left_hard::Vector{Line}, lines_left_soft::Vector{Line}, lines_right_hard::Vector{Line}, lines_right_soft::Vector{Line})
    ```
    according to the boundary lines, generate vertice
    ```
    vertice_left_hard = gen_points(lines_left_hard)
    vertice_left_soft = gen_points(lines_left_soft)
    vertice_right_hard = gen_points(lines_right_hard)
    vertice_right_soft = gen_points(lines_right_soft)
    return vertice_left_hard, vertice_left_soft, vertice_right_hard, vertice_right_soft
end

function gen_cor_points(vertice_left::Vector{Point2D}, vertice_right::Vector{Point2D}, left_num::Int64, right_num::Int64)
    if left_num == 0
        left_points = vertice_left
    elseif left_num == 1
        left_mid = Point2D((vertice_left[1].x + vertice_left[2].x)/2, (vertice_left[1].y + vertice_left[2].y)/2)
        a_nor = vertice_left[2].y - vertice_left[1].y
        b_nor = vertice_left[1].x - vertice_left[2].x
        left_add_1 = Point2D(left_mid.x - ext_dis2*a_nor/norm([a_nor, b_nor]), left_mid.y - ext_dis2*b_nor/norm([a_nor, b_nor]))
        left_points = Vector([vertice_left[1], left_add_1, vertice_left[2]])
    elseif left_num == 2
        left_mid1 = Point2D((2*vertice_left[1].x + vertice_left[2].x)/3, (2*vertice_left[1].y + vertice_left[2].y)/3)
        left_mid2 = Point2D((vertice_left[1].x + 2*vertice_left[2].x)/3, (vertice_left[1].y + 2*vertice_left[2].y)/3)
        a_nor = vertice_left[2].y - vertice_left[1].y
        b_nor = vertice_left[1].x - vertice_left[2].x
        left_add_1 = Point2D(left_mid1.x - ext_dis2*a_nor/norm([a_nor, b_nor]), left_mid1.y - ext_dis2*b_nor/norm([a_nor, b_nor]))
        left_add_2 = Point2D(left_mid2.x - ext_dis2*a_nor/norm([a_nor, b_nor]), left_mid2.y - ext_dis2*b_nor/norm([a_nor, b_nor]))
        left_points = Vector([vertice_left[1], left_add_1, left_add_2, vertice_left[2]])
    else
        println("left_num not find")
    end
    if right_num == 0 
        right_points = vertice_right
    elseif right_num == 1
        right_mid = Point2D((vertice_right[1].x + vertice_right[2].x)/2, (vertice_right[1].y + vertice_right[2].y)/2)
        a_nor = vertice_right[2].y - vertice_right[1].y
        b_nor = vertice_right[1].x - vertice_right[2].x
        right_add_1 = Point2D(right_mid.x + ext_dis2*a_nor/norm([a_nor, b_nor]), right_mid.y + ext_dis2*b_nor/norm([a_nor, b_nor]))
        right_points = Vector([vertice_right[1], right_add_1, vertice_right[2]])
    elseif right_num == 2
        right_mid1 = Point2D((2*vertice_right[1].x + vertice_right[2].x)/3, (2*vertice_right[1].y + vertice_right[2].y)/3)
        right_mid2 = Point2D((vertice_right[1].x + 2*vertice_right[2].x)/3, (vertice_right[1].y + 2*vertice_right[2].y)/3)
        a_nor = vertice_right[2].y - vertice_right[1].y
        b_nor = vertice_right[1].x - vertice_right[2].x
        right_add_1 = Point2D(right_mid1.x + ext_dis2*a_nor/norm([a_nor, b_nor]), right_mid1.y + ext_dis2*b_nor/norm([a_nor, b_nor]))
        right_add_2 = Point2D(right_mid2.x + ext_dis2*a_nor/norm([a_nor, b_nor]), right_mid2.y + ext_dis2*b_nor/norm([a_nor, b_nor]))
        right_points = Vector([vertice_right[1], right_add_1, right_add_2, vertice_right[2]])
    else
        println("right_num not find")
    end
    return left_points, right_points
end

function gen_cor_lines(left_points::Vector{Point2D}, right_points::Vector{Point2D})
    left_lines = Vector{Line}()
    right_lines = Vector{Line}()
    for i in 1:size(left_points,1)-1
        a = left_points[i+1].y - left_points[i].y
        b = left_points[i].x - left_points[i+1].x
        c = -left_points[i].x * left_points[i+1].y + left_points[i+1].x * left_points[i].y
        line = Line(a, b, c)
        push!(left_lines, line)
    end
    for i in 1:size(right_points,1)-1
        a = -(right_points[i+1].y - right_points[i].y)
        b = -(right_points[i].x - right_points[i+1].x)
        c = -(-right_points[i].x * right_points[i+1].y + right_points[i+1].x * right_points[i].y)
        line = Line(a, b, c)
        push!(right_lines, line)
    end
    return left_lines, right_lines
end

function gen_convex_corridor(vertice_left_hard::Vector{Point2D}, vertice_left_soft::Vector{Point2D}, vertice_right_hard::Vector{Point2D}, vertice_right_soft::Vector{Point2D}, route::Vector{Point2D}; type::Symbol)
    ```
    generate corridor for each segment
    ```
    if type == :simple 
        yaw = atan(route[2].y - route[1].y, route[2].x - route[1].x)
        left_num = 0
        right_num = 0
        left_points_hard, right_points_hard = gen_cor_points(vertice_left_hard, vertice_right_hard, left_num, right_num)
        left_lines_hard, right_lines_hard = gen_cor_lines(left_points_hard, right_points_hard)
        left_points_soft, right_points_soft = gen_cor_points(vertice_left_soft, vertice_right_soft, left_num, right_num)
        left_lines_soft, right_lines_soft = gen_cor_lines(left_points_soft, right_points_soft)
        corridor = Corridor(left_points_hard, right_points_hard, left_lines_hard, right_lines_hard,left_points_soft, right_points_soft, left_lines_soft, right_lines_soft, yaw)
        return corridor
    elseif type == :complex
        yaw = atan(route[2].y - route[1].y, route[2].x - route[1].x)
        left_num = rand([0,1,2])
        right_num = rand([0,1,2])
        left_points_hard, right_points_hard = gen_cor_points(vertice_left_hard, vertice_right_hard, left_num, right_num)
        left_lines_hard, right_lines_hard = gen_cor_lines(left_points_hard, right_points_hard)
        left_points_soft, right_points_soft = gen_cor_points(vertice_left_soft, vertice_right_soft, left_num, right_num)
        left_lines_soft, right_lines_soft = gen_cor_lines(left_points_soft, right_points_soft)
        corridor = Corridor(left_points_hard, right_points_hard, left_lines_hard, right_lines_hard,left_points_soft, right_points_soft, left_lines_soft, right_lines_soft, yaw)
    else
        println("type not find")
    end
end

function gen_unstructured_env(route::Matrix{Float64}; type::Symbol)
    ```
    - according to vertice, generate corridor for each segment
    - corridor include: left vertice, right vertice, left lines, right lines
    - original lines and vertice are directly generated by routes, thus the corresponding corridors are simple tubes
    - the vertice can be added, and lines can be recomputed to generate complex convex corridors
    - vertice are used for Plots, lines are used for Optimization
    - parameters:
        - route: Matrix, the route of the vehicle
        - type: Symbol, the type of corridor, :simple or :complex
    ```
    lines_left_hard, lines_left_soft, lines_right_hard, lines_right_soft = gen_lines(route)
    vertice_left_hard, vertice_left_soft, vertice_right_hard, vertice_right_soft = gen_vertice(lines_left_hard, lines_left_soft, lines_right_hard, lines_right_soft)
    L = length(route[:,1])
    # @show lines_left
    # @show vertice_left
    corridors = Vector{Corridor}()
    for i = 1:L-1 
        route_P = [Point2D(route[i,1], route[i,2]), Point2D(route[i+1,1], route[i+1,2])]
        # @show vertice_left
        # @show route_P
        corridor = gen_convex_corridor(vertice_left_hard[i:i+1], vertice_left_soft[i:i+1], vertice_right_hard[i:i+1], vertice_right_soft[i:i+1], route_P; type=type)
        push!(corridors, corridor)
    end
    return corridors
end

function gen_unstructured_init(route::Matrix{Float64})
    pos = [route[1,:]'; route[end,:]']
    yaw = [atan(route[2,2] - route[1,2], route[2,1] - route[1,1]); atan(route[end,2] - route[end-1,2], route[end,1] - route[end-1,1])]
    vel = [1.0*cos(yaw[1]) 1.0*sin(yaw[1]); 1.0*cos(yaw[2]) 1.0*sin(yaw[2])]
    acc = [0.1*cos(yaw[1]) 0.1*sin(yaw[1]); 0.1*cos(yaw[2]) 0.1*sin(yaw[2])]
    return Cons_Init(pos, vel, acc)
end

function gen_unstructured_cons_trailer(corridors::Vector{Corridor}, cons_limits::Cons_Limits)
    cons_lim = cons_limits.limits_xplus
    L = length(corridors)
    limits_list = []
    obs_list = []
    for i in 1:L 
        yaw = corridors[i].yaw
        R = [cos(yaw) -sin(yaw); sin(yaw) cos(yaw)]
        RR = [R zeros(2,2) zeros(2,2); zeros(2,2) R zeros(2,2); zeros(2,2) zeros(2,2) R]
        A = zeros(size(cons_lim.A))
        B = zeros(size(cons_lim.B))
        C = zeros(size(cons_lim.C))
        D = zeros(size(cons_lim.D))
        for j in 1:size(A,1)
            A[j,:] = cons_lim.A[j,:]'*RR'
            B[j] = cons_lim.B[j]
        end
        new_lim = Cons_Corrs(A, B, C, D)
        push!(limits_list, new_lim)

        left_lines = corridors[i].soft_left_lines
        right_lines = corridors[i].soft_right_lines
        A = zeros(length(left_lines)+length(right_lines), 4)
        B = zeros(length(left_lines)+length(right_lines), 1)
        for j in 1:(length(left_lines)+length(right_lines))
            if j <= length(left_lines)
                A[j,1] = left_lines[j].a
                A[j,2] = left_lines[j].b
                A[j,3] = left_lines[j].a * pole_len
                A[j,4] = left_lines[j].b * pole_len
                B[j] = left_lines[j].c
            else
                A[j,1] = right_lines[j-length(left_lines)].a 
                A[j,2] = right_lines[j-length(left_lines)].b
                A[j,3] = right_lines[j-length(left_lines)].a * pole_len
                A[j,4] = right_lines[j-length(left_lines)].b * pole_len
                B[j] = right_lines[j-length(left_lines)].c
            end
        end
        new_obs = Cons_Corrs(A, B, zeros(4,4), zeros(4,1))
        push!(obs_list, new_obs)
    end
    return limits_list, obs_list
end

function gen_unstructured_cons_car(corridors::Vector{Corridor}, cons_limits::Cons_Limits)
    cons_lim = cons_limits.limits_xplus
    L = length(corridors)
    limits_list = []
    obs_list = []
    for i in 1:L
        yaw = corridors[i].yaw 
        R = [cos(yaw) -sin(yaw); sin(yaw) cos(yaw)]
        RR = [R zeros(2,2); zeros(2,2) R]
        A = zeros(size(cons_lim.A))
        B = zeros(size(cons_lim.B))
        C = zeros(size(cons_lim.C))
        D = zeros(size(cons_lim.D))
        for j in 1:size(A,1)
            A[j,:] = cons_lim.A[j,:]'*RR'
            B[j] = cons_lim.B[j]
        end
        new_lim = Cons_Corrs(A, B, C, D)
        push!(limits_list, new_lim)

        left_lines = corridors[i].soft_left_lines
        right_lines = corridors[i].soft_right_lines
        A = zeros(length(left_lines)+length(right_lines), 2)
        B = zeros(length(left_lines)+length(right_lines), 1)
        for j in 1:(length(left_lines)+length(right_lines))
            if j <= length(left_lines)
                A[j,1] = left_lines[j].a
                A[j,2] = left_lines[j].b
                B[j] = left_lines[j].c
            else
                A[j,1] = right_lines[j].a 
                A[j,2] = right_lines[j].b
                B[j] = right_lines[j].c
            end
            new_obs = Cons_Corrs(A, B, zeros(2,2), zeros(2,1))
            push!(obs_list, new_obs)
        end
    end
    return limits_list, obs_list
end

##
function plot_unstructured_env(corridors::Vector{Corridor})
    fig = plot(aspect_ratio=:equal)
    for corridor in corridors
        hard_left_points = corridor.hard_left_points
        hard_right_points = corridor.hard_right_points
        soft_left_points = corridor.soft_left_points
        soft_right_points = corridor.soft_right_points
        for i in 1:size(hard_left_points,1)-1
            plot!(fig, [hard_left_points[i].x, hard_left_points[i+1].x], [hard_left_points[i].y, hard_left_points[i+1].y], color=:black, label="", linewidth=2)
            plot!(fig, [soft_left_points[i].x, soft_left_points[i+1].x], [soft_left_points[i].y, soft_left_points[i+1].y], color=:black, label="", linewidth=2, linestyle=:dash)
        end
        for i in 1:size(hard_right_points,1)-1
            plot!(fig, [hard_right_points[i].x, hard_right_points[i+1].x], [hard_right_points[i].y, hard_right_points[i+1].y], color=:black, label="",linewidth=2)
            plot!(fig, [soft_right_points[i].x, soft_right_points[i+1].x], [soft_right_points[i].y, soft_right_points[i+1].y], color=:black, label="", linewidth=2, linestyle=:dash)
        end

    end
    return fig
end

# function plot_boundary_line(fig, corridors::Vector{Corridor})
#     dis = 0.0
#     for corridor in corridors
#         hard_left_lines = corridor.hard_left_lines
#         hard_right_lines = corridor.hard_right_lines
#         hard_left_points = corridor.hard_left_points
#         hard_right_points = corridor.hard_right_points

#         soft_left_lines = corridor.soft_left_lines
#         soft_right_lines = corridor.soft_right_lines
#         soft_left_points = corridor.soft_left_points
#         soft_right_points = corridor.soft_right_points
#         for i in 1:size(hard_left_points,1)-1
#             x_vals = range(hard_left_points[i].x-dis, hard_left_points[i+1].x+dis, length=100)
#             y_vals = (-hard_left_lines[i].c .- hard_left_lines[i].a*x_vals) / hard_left_lines[i].b
#             plot!(fig, x_vals, y_vals, color=:red, label="", linewidth=1)
#             x_vals = range(soft_left_points[i].x-dis, soft_left_points[i+1].x+dis, length=100)
#             y_vals = (-soft_left_lines[i].c .- soft_left_lines[i].a*x_vals) / soft_left_lines[i].b
#             plot!(fig, x_vals, y_vals, color=:blue, label="", linewidth=1, linestyle=:dash)
#         end
#         for i in 1:size(hard_right_points,1)-1
#             x_vals = range(hard_right_points[i].x-dis, hard_right_points[i+1].x+dis, length=100)
#             y_vals = (-hard_right_lines[i].c .- hard_right_lines[i].a*x_vals) / hard_right_lines[i].b
#             plot!(fig, x_vals, y_vals, color=:red, label="", linewidth=1)
#             x_vals = range(soft_right_points[i].x-dis, soft_right_points[i+1].x+dis, length=100)
#             y_vals = (-soft_right_lines[i].c .- soft_right_lines[i].a*x_vals) / soft_right_lines[i].b
#             plot!(fig, x_vals, y_vals, color=:blue, label="", linewidth=1, linestyle=:dash)
#         end
#     end
#     return fig
# end


## Test 
# route = [0.0 0.0;
#         3.0 3.0;
#         6.0 3.0;
#         9.0 5.0] * 1.0

# corridors = gen_unstructured_env(route; type=:complex)

# fig_env = plot_unstructured_env(corridors)
# fig_env = plot_boundary_line(fig_env, corridors)