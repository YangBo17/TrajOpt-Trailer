using JLD2, CSV, DataFrames
using Plots
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
- crimson
=#
ours_color = :seagreen
eira_color = :crimson
altro_color = :royalblue
nlp_color = :darkorange

include("../utils/env_car2.jl")
linewidth = 1
scale = 9.0
carbody = BodyReal()

legend_loc = :topright

label_font = 16
tick_font = 14
legend_font = 16


@load "benchmark/data/Ours2024_car_traj.jld2" traj
ours24_traj = traj
@load "benchmark/data/Eira2022_car_traj.jld2" traj 
eiras22_traj = traj
@load "benchmark/data/Altro2019_car_traj.jld2" traj
altro19_traj = traj
@load "benchmark/data/NLP2016_car_traj.jld2" traj
nlp16_traj = traj 

function scale_env(route, corridors, obstacles, cons_init, border)
    route = route * scale
    corridors = corridors * scale
    obstacles = obstacles * scale
    cons_init_pos = cons_init.pos * scale
    cons_init = Cons_Init(cons_init.pos, cons_init.vel, cons_init.acc)
    border = border * scale
    fig_env = plot_env(route, corridors, obstacles, cons_init, border)
    return fig_env
end

function scale_traj(traj)
    t = traj.t * sqrt(scale)
    x = traj.x * scale
    y = traj.y * scale
    ψ = traj.ψ
    v = traj.v * sqrt(scale)
    δ = traj.δ
    a = traj.a 
    return Traj_car(t,x,y,ψ,v,δ,a)
end

fig_env = scale_env(route, corridors, obstacles, cons_init, border)
plot_car_trans!(fig_env, car_line, cons_line * scale)
plot!(fig_env, aspect_ratio = 1)
plot!(fig_env, ylims=(-23,+23))

ours24_traj = scale_traj(ours24_traj)
eiras22_traj = scale_traj(eiras22_traj)
altro19_traj = scale_traj(altro19_traj)
nlp16_traj = scale_traj(nlp16_traj)

fig_env_stack = plot(fig_env, legend=:top, xlabel=L"x(m)", ylabel=L"y(m)", labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font, legend_columns=4, aspect_ratio=1)
function plot_traj(fig_traj, traj::Traj_car; method::String, color::Symbol)
    fig_traj_new = plot(fig_traj, traj.x, traj.y, label=method, color=color, linewidth=2, aspect_ratio=1)
    return fig_traj_new
end

fig_traj = plot_traj(fig_env_stack, ours24_traj, method="Ours", color=ours_color)
fig_traj = plot_traj(fig_traj, eiras22_traj, method="Eiras", color=eira_color)
fig_traj = plot_traj(fig_traj, altro19_traj, method="Howell", color=altro_color)
fig_traj = plot_traj(fig_traj, nlp16_traj, method="Pardo", color=nlp_color)

savefig(fig_traj, "benchmark/figs/car_traj.tex")
@show fig_traj

##
fig_env_iso = plot(fig_env, legend=:top, xlabel=L"x(m)", ylabel=L"y(m)", labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font, aspect_ratio=1)
function visual_traj(fig_env, traj::Traj_car; method::String, c_traj::Symbol, c_car::Symbol, num::Int64, legend_loc::Symbol)
    N = length(traj.δ)
    interval_frame = Int64(floor(N / (num+1)))
    index_frame = [1 + i * interval_frame for i in 0:num]
    push!(index_frame, N)
    fig_traj = plot(fig_env, traj.x, traj.y, legend = legend_loc, color = c_traj, label=method, linewidth=2, aspect_ratio=1)
    for i in index_frame
        car_s = [traj.x[i], traj.y[i], traj.ψ[i], traj.δ[i]]
        car_shapes = depict_car(carbody, car_s)
        render_car!(fig_traj, car_shapes; c=c_car)
    end
    return fig_traj
end
num = 5
fig_traj_ours24 = visual_traj(fig_env_iso, ours24_traj, method="Ours", c_traj=ours_color, c_car=ours_color, num=num, legend_loc=legend_loc)
fig_traj_eiras22 = visual_traj(fig_env_iso, eiras22_traj, method="Eiras", c_traj=eira_color, c_car=eira_color, num=num, legend_loc=legend_loc)
fig_traj_altro19 = visual_traj(fig_env_iso, altro19_traj, method="Howell", c_traj=altro_color, c_car=altro_color, num=num, legend_loc=legend_loc)
fig_traj_nlp16 = visual_traj(fig_env_iso, nlp16_traj, method="Pardo", c_traj=nlp_color, c_car=nlp_color, num=num, legend_loc=legend_loc)

fig_traj_iso = plot(fig_traj_ours24, fig_traj_eiras22, fig_traj_altro19, fig_traj_nlp16, layout=(2,2))

savefig(fig_traj_iso, "benchmark/figs/car_traj_iso.tex")

savefig(fig_traj_ours24, "benchmark/Figs/car_traj_01_pgf.tex")
savefig(fig_traj_eiras22, "benchmark/Figs/car_traj_02_pgf.tex")
savefig(fig_traj_altro19, "benchmark/Figs/car_traj_03_pgf.tex")
savefig(fig_traj_nlp16, "benchmark/Figs/car_traj_04_pgf.tex")
@show fig_traj_iso

##
label_font = 16
tick_font = 14
legend_font = 16
function plot_v!(fig_v, traj::Traj_car; method::String, color::Symbol)
    fig_v_new = plot!(fig_v, traj.t, traj.v, label=method, color=color, linewidth=linewidth)
    return fig_v_new
end

function plot_a!(fig_a, traj::Traj_car; method::String, color::Symbol)
    if method == "Ours"
        dt = sum(diff(traj.t)) / length(diff(traj.t))
        a = diff(traj.v) / dt  
        a = vcat(a, a[end])
    elseif method == "Howell"
        a = vcat(traj.a, traj.a[end])
    else
        a = traj.a
    end
    plot!(fig_a, traj.t, a, label=method, color=color, linewidth=linewidth)
end

function plot_κ!(fig_κ, traj::Traj_car; method::String, color::Symbol)
    if method == "Howell"
        δ = vcat(traj.δ, traj.δ[end])
    else
        δ = traj.δ
    end
    plot!(fig_κ, traj.t, tan.(δ) ./ carbody.L, label=method, color=color, linewidth=linewidth)
end

fig_v = plot(ylabel=L"v", xlabel="", legend=false, ylims=(0.9*sqrt(scale), 2.2*sqrt(scale)),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
hline!(fig_v, [0.5*sqrt(scale), 2.0*sqrt(scale)], label="", color=:gray, linewidth=linewidth)
plot_v!(fig_v, ours24_traj, method="Ours", color=ours_color)
plot_v!(fig_v, eiras22_traj, method="Eiras", color=eira_color)
plot_v!(fig_v, altro19_traj, method="Howell", color=altro_color)
plot_v!(fig_v, nlp16_traj, method="Pardo", color=nlp_color)

fig_a = plot(ylabel=L"a", xlabel="", legend=:topright, legend_columns=4, ylims=(-1.0, 1.8),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
hline!(fig_a, [-2.0, +2.0], label="", color=:gray, linewidth=linewidth)
plot_a!(fig_a, ours24_traj, method="Ours", color=ours_color)
plot_a!(fig_a, eiras22_traj, method="Eiras", color=eira_color)
plot_a!(fig_a, altro19_traj, method="Howell", color=altro_color)
plot_a!(fig_a, nlp16_traj, method="Pardo", color=nlp_color)

fig_κ = plot(ylabel=L"\kappa", xlabel=L"t(s)", legend=false, ylims=(-1.2/scale, 1.2/scale),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
hline!(fig_κ, [-2.0/scale, +2.0/scale], label="", color=:gray, linewidth=linewidth)
plot_κ!(fig_κ, ours24_traj, method="Ours", color=ours_color)
plot_κ!(fig_κ, eiras22_traj, method="Eiras", color=eira_color)
plot_κ!(fig_κ, altro19_traj, method="Howell", color=altro_color)
plot_κ!(fig_κ, nlp16_traj, method="Pardo", color=nlp_color)

fig_limits = plot(fig_v, fig_a, fig_κ, layout=(3,1))

savefig(fig_limits, "benchmark/Figs/car_limits_pgf.tex")
@show fig_limits

##
function calc_energy(traj; method::String)
    dt = sum(diff(traj.t)) / length(diff(traj.t))
    if method == "Ours"
        a = diff(traj.v) / dt
        a = vcat(a, a[end])
        δ = traj.δ
    elseif method == "Howell"
        a = vcat(traj.a, traj.a[end])
        δ = vcat(traj.δ, traj.δ[end])
    else
        a = traj.a
        δ = traj.δ
    end
    Energy = (sum(a.^2) + sum(δ.^2)) * dt
    Energy = round(Energy, digits=3)
    return Energy
end

function calc_length(traj; method::String)
    Length = 0.0
    for i in 2:length(traj.t)
        Length += sqrt((traj.x[i] - traj.x[i-1])^2 + (traj.y[i] - traj.y[i-1])^2)
    end
    Length = round(Length, digits=3)
    return Length
end

function calc_ave_curvature(traj; method::String)
    dt = sum(diff(traj.t)) / length(diff(traj.t))
    if method == "Howell"
        δ = vcat(traj.δ, traj.δ[end])
    else
        δ = traj.δ
    end
    Curvature = sum(abs.(tan.(δ) / carbody.L * dt)) / traj.t[end]
    Curvature = round(Curvature, digits=3)
    return Curvature
end

function calc_max_curvature(traj; method::String)
    if method == "Howell"
        δ = vcat(traj.δ, traj.δ[end])
    else
        δ = traj.δ
    end
    Curvature = maximum(abs.(tan.(δ) / carbody.L))
    Curvature = round(Curvature, digits=3)
    return Curvature
end

function calc_jerk(traj; method::String)
    dt = sum(diff(traj.t)) / length(diff(traj.t))
    if method == "Ours"
        a = diff(traj.v) / dt
        a = vcat(a, a[end])
    elseif method == "Howell"
        a = vcat(traj.a, traj.a[end])
    else
        a = traj.a
    end
    Jerk = sum(abs.(diff(a) / dt))
    Jerk = round(Jerk, digits=3)
    return Jerk
end

energy_ours24 = calc_energy(ours24_traj; method="Ours")
energy_eiras22 = calc_energy(eiras22_traj; method="Eira")
energy_altro19 = calc_energy(altro19_traj; method="Howell")
energy_nlp16 = calc_energy(nlp16_traj; method="Pardo")

length_ours24 = calc_length(ours24_traj; method="Ours")
length_eiras22 = calc_length(eiras22_traj; method="Eira")
length_altro19 = calc_length(altro19_traj; method="Howell")
length_nlp16 = calc_length(nlp16_traj; method="Pardo")

max_curvature_ours24 = calc_max_curvature(ours24_traj; method="Ours")
max_curvature_eiras22 = calc_max_curvature(eiras22_traj; method="Eira")
max_curvature_altro19 = calc_max_curvature(altro19_traj; method="Howell")
max_curvature_nlp16 = calc_max_curvature(nlp16_traj; method="Pardo")

# ave_curvature_ours24 = calc_ave_curvature(ours24_traj; method="Ours")
# ave_curvature_eiras22 = calc_ave_curvature(eiras22_traj; method="Eira")
# ave_curvature_altro19 = calc_ave_curvature(altro19_traj; method="Howell")
# ave_curvature_nlp16 = calc_ave_curvature(nlp16_traj; method="Pardo")

# jerk_ours24 = calc_jerk(ours24_traj; method="Ours") 
# jerk_eiras22 = calc_jerk(eiras22_traj; method="Eira")
# jerk_altro19 = calc_jerk(altro19_traj; method="Howell")
# jerk_nlp16 = calc_jerk(nlp16_traj; method="Pardo")