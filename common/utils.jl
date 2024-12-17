function clip(x::Real, minx::Real, maxx::Real)::Real
    return max(minx, min(x, maxx))
end

function gen_corridors(route::Matrix, road_unit::Float64) # N x 2
    corridors = [];
    L = length(route[:,1])
    direction = [];
    for i in 2:L
        if (route[i,1] - route[i-1,1] >= 0 && route[i,2] - route[i-1,2] == 0) 
            push!(direction, :right)
        elseif (route[i,1] - route[i-1,1] == 0 && route[i,2] - route[i-1,2] >= 0)
            push!(direction, :up)
        elseif (route[i,1] - route[i-1,1] <= 0 && route[i,2] - route[i-1,2] == 0)
            push!(direction, :left)
        elseif (route[i,1] - route[i-1,1] == 0 && route[i,2] - route[i-1,2] <= 0)
            push!(direction, :down)
        end
    end
    push!(direction, :end)
    indx_a = 1
    for i in 2:L
        if direction[i] != direction[indx_a]
            min_x = minimum([route[i,1]*road_unit-road_unit/2, route[indx_a,1]*road_unit-road_unit/2])
            max_x = maximum([route[i,1]*road_unit+road_unit/2, route[indx_a,1]*road_unit+road_unit/2])
            min_y = minimum([route[i,2]*road_unit-road_unit/2, route[indx_a,2]*road_unit-road_unit/2])
            max_y = maximum([route[i,2]*road_unit+road_unit/2, route[indx_a,2]*road_unit+road_unit/2])
            indx_a = i
            push!(corridors, [min_x,max_x,min_y,max_y])
        end
    end
    L_corr = length(corridors)
    corridors = reshape(vcat(corridors...),(4,L_corr))'
    return corridors
end

