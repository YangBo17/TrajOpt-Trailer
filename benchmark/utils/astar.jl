using AStarSearch

grid = 0.1
dilate = 0.3
UP = CartesianIndex(-1, 0)
DOWN = CartesianIndex(1, 0)
LEFT = CartesianIndex(0, -1)
RIGHT = CartesianIndex(0, 1)
UP_LEFT = CartesianIndex(-1, -1)
UP_RIGHT = CartesianIndex(-1, 1)
DOWN_LEFT = CartesianIndex(1, -1)
DOWN_RIGHT = CartesianIndex(1, 1)
DIRECTIONS = [UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT]
function diagonal_heuristic(a::CartesianIndex, b::CartesianIndex)
    dx = abs(a[1] - b[1])
    dy = abs(a[2] - b[2])
    D = 1  
    D2 = sqrt(2)  
    return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)
end
# manhattan(a::CartesianIndex, b::CartesianIndex) = sum(abs.((b - a).I))
function mazeneighbours(maze, p)
    res = CartesianIndex[]
    for d in DIRECTIONS
        n = p + d
        if 1 ≤ n[1] ≤ size(maze)[1] && 1 ≤ n[2] ≤ size(maze)[2] && !maze[n]
            push!(res, n)
        end
    end
    return res
end
function solvemaze(maze, start, goal)
    currentmazeneighbours(state) = mazeneighbours(maze, state)
    # Here you can use any of the exported search functions, they all share the same interface, but they won't use the heuristic and the cost
    return astar(currentmazeneighbours, start, goal, heuristic=diagonal_heuristic)
end

function maze2real(border, i, j, grid)
    x_start = border[1]
    y_start = border[4]
    x = x_start + (j-1) * grid
    y = y_start - (i-1) * grid
    return x, y
end

function real2maze(border, x, y, grid)
    x_start = border[1]
    y_start = border[4]
    i = floor(Int, -(y-y_start)/grid + 1)
    j = floor(Int, (x-x_start)/grid + 1)
    return i, j
end

function dilate_obs(obstacles, dilate)
    obs = deepcopy(obstacles)
    for i in axes(obs,1), j in axes(obs,2)
        obs[i,j,1] -= dilate
        obs[i,j,2] += dilate
        obs[i,j,3] -= dilate
        obs[i,j,4] += dilate
    end
    return obs
end

function gen_maze(border, obs, grid)
    x_points = Int(ceil((border[2]-border[1])/grid)) + 1
    y_points = Int(ceil((border[4]-border[3])/grid)) + 1
    maze = zeros(y_points, x_points)
    for index in CartesianIndices(maze)
        i, j = index.I 
        x, y = maze2real(border, i, j, grid)
        for obs_i in axes(obs,1), obs_j in axes(obs,2)
            if x >= obs[obs_i,obs_j,1] && x <= obs[obs_i,obs_j,2] && y >= obs[obs_i,obs_j,3] && y <= obs[obs_i,obs_j,4]
                maze[i,j] = 1
            end
            if obs_i == 1 || obs_i == size(obs,1)
                # if i == 1 && j == 1
                #     @show obs_i
                # end
                obs1_xmin, obs1_xmax, obs1_ymin, obs1_ymax = obs[obs_i,1,:]
                obs2_xmin, obs2_xmax, obs2_ymin, obs2_ymax = obs[obs_i,2,:]
                if abs(obs1_xmin - obs2_xmin) < 1e-6
                    new_xmax = obs1_xmin
                    new_xmin = obs1_xmin - 1.0
                    new_ymin = min(obs1_ymin, obs2_ymin)
                    new_ymax = max(obs1_ymax, obs2_ymax)
                    if x >= new_xmin && x <= new_xmax && y >= new_ymin && y <= new_ymax
                        maze[i,j] = 1
                    end
                elseif abs(obs1_xmax - obs2_xmax) < 1e-6
                    new_xmin = obs1_xmax
                    new_xmax = obs1_xmax + 1.0
                    new_ymin = min(obs1_ymin, obs2_ymin)
                    new_ymax = max(obs1_ymax, obs2_ymax)
                    if x >= new_xmin && x <= new_xmax && y >= new_ymin && y <= new_ymax
                        maze[i,j] = 1
                    end
                elseif abs(obs1_ymin - obs2_ymin) < 1e-6
                    new_ymax = obs1_ymin
                    new_ymin = obs1_ymin - 1.0
                    new_xmin = min(obs1_xmin, obs2_xmin)
                    new_xmax = max(obs1_xmax, obs2_xmax)
                    if x >= new_xmin && x <= new_xmax && y >= new_ymin && y <= new_ymax
                        maze[i,j] = 1
                    end
                elseif abs(obs1_ymax - obs2_ymax) < 1e-6
                    new_ymin = obs1_ymax
                    new_ymax = obs1_ymax + 1.0
                    new_xmin = min(obs1_xmin, obs2_xmin)
                    new_xmax = max(obs1_xmax, obs2_xmax)
                    if x >= new_xmin && x <= new_xmax && y >= new_ymin && y <= new_ymax
                        maze[i,j] = 1
                    end
                end
            end
        end
    end
    maze = maze .== 1
    return maze
end

function get_path(res)
    path = zeros(2,length(res.path))
    for i in eachindex(res.path)
        path[1,i],path[2,i] = maze2real(border,res.path[i][1],res.path[i][2],grid)
    end
    return path
end