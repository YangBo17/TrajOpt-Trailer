using JLD2, CSV, DataFrames
using Plots
using LaTeXStrings
# using PGFPlotsX
# pgfplotsx()

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
=#
ours_color = :seagreen
li_color = :purple
altro_color = :royalblue
nlp_color = :darkorange

linewidth=1

legend_loc = :topright

label_font = 16
tick_font = 14
legend_font = 16

env_num = 1
route = "route$env_num"
T0 = "T$env_num"

@load "benchmark/env/route$env_num.jld2" route
@load "benchmark/env/T$env_num.jld2" T0

include("../utils/load_env_trailer.jl")

@load "benchmark/data/Ours2024_trailer_traj_E$env_num.jld2" traj
ours24_traj = traj
@load "benchmark/data/Li2022_trailer_traj_E$env_num.jld2" traj 
li22_traj = traj
@load "benchmark/data/Altro2019_trailer_traj_E$env_num.jld2" traj
altro19_traj = traj
@load "benchmark/data/NLP2016_trailer_traj_E$env_num.jld2" traj
nlp16_traj = traj 

plot!(fig_env, legend=:topleft, xlabel=L"x(m)", ylabel=L"y(m)", labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font, aspect_ratio=1)

function plot_traj(fig_traj, traj::Traj_trailer; method::String, color::Symbol)
    fig_traj_new = plot(fig_traj, traj.x0, traj.y0, label=method, color=color, linewidth=2)
    plot!(fig_traj_new, traj.x1, traj.y1, label="", color=color, linestyle=:dash, linewidth=2)
    return fig_traj_new
end

fig_traj = plot_traj(fig_env, ours24_traj, method="Ours", color=ours_color)
fig_traj = plot_traj(fig_traj, li22_traj, method="Li", color=li_color)
fig_traj = plot_traj(fig_traj, altro19_traj, method="Howell", color=altro_color)
fig_traj = plot_traj(fig_traj, nlp16_traj, method="Pardo", color=nlp_color)

savefig(fig_traj, "benchmark/figs/trailer_traj.tex")
@show fig_traj

##
function visual_traj(fig_env, traj::Traj_trailer; method::String, c_traj::Symbol, c_car::Symbol, num::Int64, legend_loc::Symbol)
    N = length(traj.δ0)
    interval_frame = Int64(floor(N / (num+1)))
    index_frame = [1 + i * interval_frame for i in 0:num]
    push!(index_frame, N)
    fig_traj = plot(fig_env, traj.x0, traj.y0, legend = legend_loc, color = c_traj, label=method, linewidth=2)
    plot!(fig_traj, traj.x1, traj.y1, color = c_traj, label="", linestyle=:dash, linewidth=2)
    for i in index_frame
        trailer_s = [traj.x0[i], traj.y0[i], traj.ψ0[i], traj.x1[i], traj.y1[i], traj.ψ1[i], traj.δ0[i]]
        trailer_shapes = depict_trailer(TrailerBody(), trailer_s)
        render_trailer!(fig_traj, trailer_shapes; c=c_car)
    end
    return fig_traj
end
num = 3
fig_traj_ours24 = visual_traj(fig_env, ours24_traj, method="Ours", c_traj=ours_color, c_car=ours_color, num=num, legend_loc=legend_loc)
fig_traj_li22 = visual_traj(fig_env, li22_traj, method="Li", c_traj=li_color, c_car=li_color, num=num, legend_loc=legend_loc)
fig_traj_altro19 = visual_traj(fig_env, altro19_traj, method="Howell", c_traj=altro_color, c_car=altro_color, num=num, legend_loc=legend_loc)
fig_traj_nlp16 = visual_traj(fig_env, nlp16_traj, method="Pardo", c_traj=nlp_color, c_car=nlp_color, num=num, legend_loc=legend_loc)

fig_traj_iso = plot(fig_traj_ours24, fig_traj_li22, fig_traj_altro19, fig_traj_nlp16, layout=(2,2))
savefig(fig_traj_iso, "benchmark/figs/trailer_traj_iso_multi_pgf.tex")

savefig(fig_traj_ours24, "benchmark/Figs/trailer_traj_multi_01_pgf.tex")
savefig(fig_traj_li22, "benchmark/Figs/trailer_traj_multi_02_pgf.tex")
savefig(fig_traj_altro19, "benchmark/Figs/trailer_traj_multi_03_pgf.tex")
savefig(fig_traj_nlp16, "benchmark/Figs/trailer_traj_multi_04_pgf.tex")

@show fig_traj_iso

##
label_font = 16
tick_font = 14
legend_font = 16
function plot_v!(fig_v, traj::Traj_trailer; method::String, color::Symbol)
    fig_v_new = plot!(fig_v, traj.t, traj.v0, label=method, color=color, linewidth=linewidth)
    return fig_v_new
end

function plot_a!(fig_a, traj::Traj_trailer; method::String, color::Symbol)
    if method == "Ours"
        dt = sum(diff(traj.t)) / length(diff(traj.t))
        a0 = diff(traj.v0) / dt
        a0 = vcat(a0, a0[end])
    elseif method == "Howell"
        a0 = vcat(traj.a0, traj.a0[end])
    else
        a0 = traj.a0
    end
    plot!(fig_a, traj.t, a0, label=method, color=color, linewidth=linewidth)
end

function plot_κ!(fig_κ, traj::Traj_trailer; method::String, color::Symbol)
    if method == "Howell"
        δ0 = vcat(traj.δ0, traj.δ0[end])
    else
        δ0 = traj.δ0
    end
    plot!(fig_κ, traj.t, tan.(δ0) ./ d0, label=method, color=color, linewidth=linewidth)
end

fig_v = plot(ylabel=L"v", xlabel="", legend=:topright, legend_columns=4, ylims=(0.0, 2.3),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
plot_v!(fig_v, ours24_traj, method="Ours", color=ours_color)
plot_v!(fig_v, li22_traj, method="Li", color=li_color)
plot_v!(fig_v, altro19_traj, method="Howell", color=altro_color)
plot_v!(fig_v, nlp16_traj, method="Pardo", color=nlp_color)

fig_a = plot(ylabel=L"a", xlabel="", legend=false, ylims=(-1.0, 1.0),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
plot_a!(fig_a, ours24_traj, method="Ours", color=ours_color)
plot_a!(fig_a, li22_traj, method="Li", color=li_color)
plot_a!(fig_a, altro19_traj, method="Howell", color=altro_color)
plot_a!(fig_a, nlp16_traj, method="Pardo", color=nlp_color)

fig_κ = plot(ylabel=L"\kappa", xlabel=L"t(s)", legend=false, ylims=(-2.3, 2.3),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
plot_κ!(fig_κ, ours24_traj, method="Ours", color=ours_color)
plot_κ!(fig_κ, li22_traj, method="Li", color=li_color)
plot_κ!(fig_κ, altro19_traj, method="Howell", color=altro_color)
plot_κ!(fig_κ, nlp16_traj, method="Pardo", color=nlp_color)

fig_limits = plot(fig_v, fig_a, fig_κ, layout=(3,1))

savefig(fig_limits, "benchmark/Figs/trailer_limits_multi_pgf.tex")
@show fig_limits

##
 function calc_energy(traj; method::String)
    dt = sum(diff(traj.t)) / length(diff(traj.t))
    if method == "Ours"
        a0 = diff(traj.v0) / dt
        a0 = vcat(a0, a0[end])
        δ0 = traj.δ0
    elseif method == "Howell"
        a0 = vcat(traj.a0, traj.a0[end])
        δ0 = vcat(traj.δ0, traj.δ0[end])
    else
        a0 = traj.a0
        δ0 = traj.δ0
    end
    Energy = (sum(a0.^2) + sum(δ0.^2)) * dt
    Energy = round(Energy, digits=3)
    return Energy
end

function calc_length(traj; method::String)
    Length = 0.0
    for i in 2:length(traj.t)
        Length += sqrt((traj.x0[i] - traj.x0[i-1])^2 + (traj.y0[i] - traj.y0[i-1])^2)
    end
    Length = round(Length, digits=3)
    return Length
end

function calc_ave_curvature(traj; method::String)
    dt = sum(diff(traj.t)) / length(diff(traj.t))
    if method == "Howell"
        δ0 = vcat(traj.δ0, traj.δ0[end])
    else
        δ0 = traj.δ0
    end
    Curvature = sum(abs.(tan.(δ0) / d0 * dt)) / traj.t[end]
    Curvature = round(Curvature, digits=3)
    return Curvature
end

function calc_max_curvature(traj; method::String)
    if method == "Howell"
        δ0 = vcat(traj.δ0, traj.δ0[end])
    else
        δ0 = traj.δ0
    end
    Curvature = maximum(abs.(tan.(δ0) / d0))
    Curvature = round(Curvature, digits=3)
    return Curvature
end

function calc_jerk(traj; method::String)
    dt = sum(diff(traj.t)) / length(diff(traj.t))
    if method == "Ours"
        a0 = diff(traj.v0) / dt
        a0 = vcat(a0, a0[end])
    elseif method == "Howell"
        a0 = vcat(traj.a0, traj.a0[end])
    else
        a0 = traj.a0
    end
    Jerk = sum(abs.(diff(a0) / dt))
    Jerk = round(Jerk, digits=3)
    return Jerk
end

energy_ours24 = calc_energy(ours24_traj; method="Ours")
energy_li22 = calc_energy(li22_traj; method="Li")
energy_altro19 = calc_energy(altro19_traj; method="Howell")
energy_nlp16 = calc_energy(nlp16_traj; method="Pardo")

length_ours24 = calc_length(ours24_traj; method="Ours")
length_li22 = calc_length(li22_traj; method="Li")
length_altro19 = calc_length(altro19_traj; method="Howell")
length_nlp16 = calc_length(nlp16_traj; method="Pardo")

max_curvature_ours24 = calc_max_curvature(ours24_traj; method="Ours")
max_curvature_li22 = calc_max_curvature(li22_traj; method="Li")
max_curvature_altro19 = calc_max_curvature(altro19_traj; method="Howell")
max_curvature_nlp16 = calc_max_curvature(nlp16_traj; method="Pardo")

# ave_curvature_ours24 = calc_ave_curvature(ours24_traj; method="Ours")
# ave_curvature_li22 = calc_ave_curvature(li22_traj; method="Li")
# ave_curvature_altro19 = calc_ave_curvature(altro19_traj; method="Howell")
# ave_curvature_nlp16 = calc_ave_curvature(nlp16_traj; method="Pardo")

# jerk_ours24 = calc_jerk(ours24_traj; method="Ours")
# jerk_li22 = calc_jerk(li22_traj; method="Li")
# jerk_altro19 = calc_jerk(altro19_traj; method="Howell")
# jerk_nlp16 = calc_jerk(nlp16_traj; method="Pardo")



