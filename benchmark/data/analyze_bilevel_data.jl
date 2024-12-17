using JLD2
using CSV, DataFrames
using LaTeXStrings
using PGFPlotsX
pgfplotsx()
#=
- :slateblue
- orchid
- cornflowerblue
- seagreen
- limegreen
- tomato
- mediumvioletred
- dodgerblue
- darkorange
- deepskyblue
- royalblue
=#
t0_color = :purple
t1_color = :seagreen

label_font = 16
tick_font = 14
legend_font = 16

N = 3
include("../PDHG/bilevel_env_trailer.jl")

@load "benchmark/data/Ours2024T0_trailer_traj.jld2" traj
ours24T0_traj = traj
@load "benchmark/data/Ours2024Ts_trailer_traj.jld2" traj
ours24Ts_traj = traj

plot!(fig_env, legend=:topleft, size=(800,600), xlabel=L"x(m)", ylabel=L"y(m)", ylims=(-2.2,+6.2), xlims=(-1.5,9.5),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)

function plot_traj(fig_traj, traj::Traj_trailer; method::String, color::Symbol)
    fig_traj_new = plot(fig_traj, traj.x0, traj.y0, label=method, color=color, linewidth=2)
    plot!(fig_traj_new, traj.x1, traj.y1, label="", color=color, linestyle=:dash, linewidth=2)
    return fig_traj_new
end

fig_traj = plot_traj(fig_env, ours24T0_traj, method="Initial Trajectory", color=t0_color)
fig_traj = plot_traj(fig_traj, ours24Ts_traj, method="Final Trajectory", color=t1_color)

savefig(fig_traj, "benchmark/figs/trailer_bilevel_traj.pdf")

##
function plot_v!(fig_v, traj::Traj_trailer; method::String, color::Symbol)
    fig_v_new = plot!(fig_v, traj.t, traj.v0, label=method, color=color, linewidth=2)
    return fig_v_new
end

function plot_a!(fig_a, traj::Traj_trailer; method::String, color::Symbol)
    dt = sum(diff(traj.t)) / length(diff(traj.t))
    a0 = diff(traj.v0) / dt
    a0 = vcat(a0, a0[end])
    plot!(fig_a, traj.t, a0, label=method, color=color, linewidth=2)
end

function plot_κ!(fig_κ, traj::Traj_trailer; method::String, color::Symbol)
    δ0 = traj.δ0
    plot!(fig_κ, traj.t, tan.(δ0) ./ d0, label=method, color=color, linewidth=2)
end

fig_v = plot(ylabel="v", xlabel="", legend=:topright, legend_columns=4, ylims=(0.8, 1.5),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
plot_v!(fig_v, ours24T0_traj, method="Initial Trajectory", color=t0_color)
plot_v!(fig_v, ours24Ts_traj, method="Final Trajectory", color=t1_color)

fig_a = plot(ylabel="a", xlabel="", legend=false, ylims=(-0.5, 0.5),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
plot_a!(fig_a, ours24T0_traj, method="Initial Trajectory", color=t0_color)
plot_a!(fig_a, ours24Ts_traj, method="Final Trajectory", color=t1_color)

fig_κ = plot(ylabel=L"\kappa", xlabel="t", legend=false, ylims=(-1.8, 1.8),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
plot_κ!(fig_κ, ours24T0_traj, method="Initial Trajectory", color=t0_color)
plot_κ!(fig_κ, ours24Ts_traj, method="Final Trajectory", color=t1_color)

fig_limits = plot(fig_v, fig_a, fig_κ, layout=(3,1), size=(800,600))

savefig(fig_limits, "benchmark/figs/trailer_bilevel_limits.pdf")
