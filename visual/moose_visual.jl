##
using Revise
using CSV, DataFrames
includet("visual_car.jl")

moose_or_track = :track
filepath = joinpath(@__DIR__, "../data/sec7.csv")
df = CSV.read(filepath, DataFrame)

moose_X = [0. 1.5 4.5 7.0; 0. 1.5 4.5 7.0];
moose_Y = [-0.1775 -0.1775 -0.5425 -0.5425; 0.1775 0.1775 -0.1575 -0.1575]
W = 0.1775
moose_L = 0.1775
(Y_max, idx) = findmax(moose_Y)
(Y_min, idx) = findmin(moose_Y)
moose_scope = [moose_X[1], moose_X[end]+1.0, Y_min-W-1.0, Y_max+W+1.0]
moose_body = Body(0.4, 0.25)
line_X = [0. 10. 30. 45. 60. 80. 100. 120.];
line_Y = [0. -8. 5. -10. -5. 10. 8. 25.];
track_X = [line_X[1] line_X[2] line_X[3] line_X[4] line_X[5] line_X[6] line_X[7] line_X[8]; line_X[1] line_X[2] line_X[3] line_X[4] line_X[5] line_X[6] line_X[7] line_X[8]];
W = 3.5
track_Y = [line_Y[1]-W line_Y[2]-W line_Y[3]-W line_Y[4]-W line_Y[5]-W line_Y[6]-W line_Y[7]-W line_Y[8]-W; line_Y[1]+W line_Y[2]+W line_Y[3]+W line_Y[4]+W line_Y[5]+W line_Y[6]+W line_Y[7]+W line_Y[8]+W];
track_L = 4.5
(Y_max, idx) = findmax(line_Y)
(Y_min, idx) = findmin(line_Y)
track_scope = [line_X[1]-10, line_X[end]+10, Y_min-W-20.0, Y_max+W+20.0]
track_body = Body()
if moose_or_track == :moose 
    track_X = moose_X
    track_Y = moose_Y
    track_L = moose_L
    track_scope = moose_scope
    track_body = moose_body
end

data_opt = []
for i in 1:length(df.t)
    datapoint = AllState(df.t[i], df.x[i], df.y[i], df.ψ[i], df.v[i], df.δ[i], df.ax[i])
    push!(data_opt, datapoint)
end

fig_env = moose_env(track_X, track_Y, track_scope; L= track_L)
gif_traj = animate_traj(fig_env, data_opt, c=:blue, label="traj", fps=100, interval=15; L=track_L, body=track_body)

## control simulation
function simu(data_opt::Vector{Any}; dt::Float64=0.01)
    x = data_opt[1].x
    y = data_opt[1].y
    v = data_opt[1].v
    θ = data_opt[1].θ
    head_state = State(x, y, θ, v)
    mycar = MyCar(state=head_state, input=Input(0.,0.))
    data_dyn = []
    N = length(data_opt)
    for i in 1:N
        t = mycar.time
        u = Input(data_opt[i].ψ, data_opt[i].at)
        s = mycar.state
        car_step(mycar, u, dt)
        push!(data_dyn, AllState(t, s.x, s.y, s.θ, s.v, u.ψ, u.at))
    end
    return data_dyn
end

data_dyn  = simu(data_opt)
fig_data = visual_data(data_opt, data_dyn)