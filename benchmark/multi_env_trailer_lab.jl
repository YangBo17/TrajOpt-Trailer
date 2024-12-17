using Random
using JLD2

println("Initializing")
seed = 1
Random.seed!(seed)
segment = rand(2:3)
@show segment
include("utils/multi_env_trailer.jl")
data_name = nothing
@show route
@show T0
env_file = "benchmark/env/route$seed.jld2"
T_file = "benchmark/env/T$seed.jld2"
data_name = "E$seed"
@save(env_file, route)
@save(T_file, T0)
include("Ours_trailer_multi.jl")
display(fig_traj)
include("Li_trailer_multi.jl")
display(fig_traj)
include("NLP_trailer_multi.jl")
display(fig_traj)
include("Altro_trailer_multi.jl")
display(fig_traj)
print("Initialized")

##
using Random
using JLD2

seed = nothing
segment = nothing
data_name = nothing
max_iter = nothing
for number in 1:30
    seed = number
    Random.seed!(seed)
    if seed >=1 && seed <= 10
        max_iter = 300
        segment = rand(2:4) # 10m ~ 20m
    elseif seed >= 11 && seed <= 20
        max_iter = 300
        segment = rand(4:6) # 20m ~ 30m
    elseif seed >= 21 && seed <= 30
        max_iter = 300
        segment = rand(6:8) # 30m ~ 40m
    end
    include("utils/multi_env_trailer.jl")
    @show route
    @show T0
    env_file = "benchmark/env/route$seed.jld2"
    T_file = "benchmark/env/T$seed.jld2"
    data_name = "E$seed"
    @save(env_file, route)
    @save(T_file, T0)
    include("Ours_trailer_multi.jl")
    display(fig_traj)
    # include("Li_trailer_multi.jl") 
    # display(fig_traj)
    # include("NLP_trailer_multi.jl")
    # display(fig_traj)
    # include("Altro_trailer_multi.jl")
    # display(fig_traj)
end

