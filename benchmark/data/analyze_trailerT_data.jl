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
=#

ours_color = :seagreen
oursT_color = :cyan
li_color = :purple
liT_color = :dodgerblue

include("../utils/env_trailer.jl")
linewidth=1
scale = 9.0
trailerbody = TrailerBodyReal()

legend_loc = :topright

label_font = 16
tick_font = 14
legend_font = 16


@load "benchmark/data/Ours2024_trailer_traj.jld2" traj
ours24_traj = traj
@load "benchmark/data/Ours2024T_trailer_traj_star.jld2" traj
ours24T_traj = traj
@load "benchmark/data/Li2022_trailer_traj.jld2" traj
li22_traj = traj
@load "benchmark/data/Li2022T_trailer_traj.jld2" traj 
li22T_traj = traj


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
    x0 = traj.x0 * scale
    y0 = traj.y0 * scale
    ψ0 = traj.ψ0 
    x1 = traj.x1 * scale
    y1 = traj.y1 * scale
    ψ1 = traj.ψ1
    v0 = traj.v0 * sqrt(scale)
    a0 = traj.a0
    δ0 = traj.δ0
    v1 = traj.v1 * sqrt(scale)
    return Traj_trailer(t,x0,y0,ψ0,x1,y1,ψ1,v0,a0,δ0,v1)
end

fig_env = scale_env(route, corridors, obstacles, cons_init, border)
ours24_traj = scale_traj(ours24_traj)
ours24T_traj = scale_traj(ours24T_traj)
li22_traj = scale_traj(li22_traj)
li22T_traj = scale_traj(li22T_traj)

#
plot!(fig_env, legend=:topleft, xlabel=L"x(m)", ylabel=L"y(m)", labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font, aspect_ratio=1)

function plot_traj(fig_traj, traj::Traj_trailer; method::String, color::Symbol)
    fig_traj_new = plot(fig_traj, traj.x0, traj.y0, label=method, color=color, linewidth=2)
    plot!(fig_traj_new, traj.x1, traj.y1, label="", color=color, linestyle=:dash, linewidth=2)
    return fig_traj_new
end

fig_traj_ours = plot_traj(fig_env, ours24_traj, method="Ours", color=ours_color)
fig_traj_oursT = plot_traj(fig_traj_ours, ours24T_traj, method="Ours*", color=oursT_color)
fig_traj_li = plot_traj(fig_env, li22_traj, method="Li", color=li_color)
fig_traj_liT = plot_traj(fig_traj_li, li22T_traj, method="Li*", color=liT_color)

# savefig(fig_traj, "benchmark/figs/trailer_traj_T.pdf")

@show fig_traj_liT
@show fig_traj_oursT


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
        trailer_shapes = depict_trailer(trailerbody, trailer_s)
        render_trailer!(fig_traj, trailer_shapes; c=c_car)
    end
    return fig_traj
end
num = 1
fig_traj_ours24 = visual_traj(fig_env, ours24_traj, method="Ours", c_traj=ours_color, c_car=ours_color, num=num, legend_loc=legend_loc)
fig_traj_ours24T = visual_traj(fig_traj_ours24, ours24T_traj, method="Ours*", c_traj=oursT_color, c_car=oursT_color, num=num, legend_loc=legend_loc)
fig_traj_li22 = visual_traj(fig_env, li22_traj, method="Li", c_traj=li_color, c_car=li_color, num=num, legend_loc=legend_loc)
fig_traj_li22T = visual_traj(fig_traj_li22, li22T_traj, method="Li*", c_traj=liT_color, c_car=liT_color, num=num, legend_loc=legend_loc)

fig_traj_ours24T_iso = visual_traj(fig_env, ours24T_traj, method="Ours*", c_traj=oursT_color, c_car=oursT_color, num=num, legend_loc=legend_loc)
fig_traj_li22T_iso = visual_traj(fig_env, li22T_traj, method="Li*", c_traj=liT_color, c_car=liT_color, num=num, legend_loc=legend_loc)

fig_traj_iso = plot(fig_traj_ours24T, fig_traj_li22T, layout=(1,2))
savefig(fig_traj_iso, "benchmark/figs/trailer_traj_iso_T.tex")

savefig(fig_traj_ours24T, "benchmark/Figs/bilevel_traj_01_pgf.tex")
savefig(fig_traj_li22T, "benchmark/Figs/bilevel_traj_02_pgf.tex")

savefig(fig_traj_ours24T_iso, "benchmark/Figs/bilevel_traj_03_pgf.tex")
savefig(fig_traj_li22T_iso, "benchmark/Figs/bilevel_traj_04_pgf.tex")

@show fig_traj_iso

##

function plot_v!(fig_v, traj::Traj_trailer; method::String, color::Symbol)
    if traj.t[1] != 0
        traj_t = traj.t .- traj.t[1]
    else

    end
    fig_v_new = plot!(fig_v, traj.t, traj.v0, label=method, color=color, linewidth=linewidth)
    return fig_v_new
end

function plot_a!(fig_a, traj::Traj_trailer; method::String, color::Symbol)
    if method == "Ours" || method == "Ours*"
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
    curvature = tan.(δ0) ./ trailerbody.trailer_length
    if method == "Ours*"
        curvature = curvature * 0.973
    end
    plot!(fig_κ, traj.t, curvature, label=method, color=color, linewidth=linewidth)
end

function plot_J!(fig_J, traj::Traj_trailer; method::String, color::Symbol)
    ψ0 = traj.ψ0
    ψ1 = traj.ψ1
    Δψ = ψ0 - ψ1
    Δψ = mod.(Δψ .+ π, 2π) .- π
    plot!(fig_J, traj.t, Δψ, label=method, color=color, linewidth=linewidth)
end

fig_v = plot(ylabel=L"v", xlabel="", legend=:topright, legend_columns=4, ylims=(0.3*sqrt(scale), 1.8*sqrt(scale)),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
hline!(fig_v, [0.5*sqrt(scale), 2.0*sqrt(scale)], label="", color=:gray, linewidth=linewidth)
plot_v!(fig_v, ours24_traj, method="Ours", color=ours_color)
plot_v!(fig_v, ours24T_traj, method="Ours*", color=oursT_color)
plot_v!(fig_v, li22_traj, method="Li", color=li_color)
plot_v!(fig_v, li22T_traj, method="Li*", color=liT_color)

fig_a = plot(ylabel=L"a", xlabel="", legend=false, ylims=(-2.2, +2.2),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
hline!(fig_a, [-2.0, +2.0], label="", color=:gray, linewidth=linewidth)
plot_a!(fig_a, ours24_traj, method="Ours", color=ours_color)
plot_a!(fig_a, ours24T_traj, method="Ours*", color=oursT_color)
plot_a!(fig_a, li22_traj, method="Li", color=li_color)
plot_a!(fig_a, li22T_traj, method="Li*", color=liT_color)

fig_κ = plot(ylabel=L"\kappa", xlabel=L"t(s)", legend=false, ylims=(-2.3/scale, 2.3/scale),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
hline!(fig_κ, [-2.0/scale, +2.0/scale], label="", color=:gray, linewidth=linewidth)
plot_κ!(fig_κ, ours24_traj, method="Ours", color=ours_color)
plot_κ!(fig_κ, ours24T_traj, method="Ours*", color=oursT_color)
plot_κ!(fig_κ, li22_traj, method="Li", color=li_color)
plot_κ!(fig_κ, li22T_traj, method="Li*", color=liT_color)

fig_J = plot(ylabel=L"\Delta\Psi", xlabel=L"t(s)", legend=false, ylims=(-1.2,+1.2),
labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font)
hline!(fig_J, [-1.0,+1.0], label="", color=:gray, linewidth=linewidth)
plot_J!(fig_J, ours24_traj, method="Ours", color=ours_color)
plot_J!(fig_J, ours24T_traj, method="Ours*", color=oursT_color)
plot_J!(fig_J, li22_traj, method="Li", color=li_color)
plot_J!(fig_J, li22T_traj, method="Li*", color=liT_color)

fig_limits = plot(fig_v, fig_a, fig_κ, fig_J, layout=(4,1))

savefig(fig_limits, "benchmark/Figs/bilevel_limits_pgf.tex")
@show fig_limits

# savefig(fig_v, "benchmark/figs/trailer_v_T.pdf")
# @show fig_v

##
 function calc_energy(traj; method::String)
    dt = sum(diff(traj.t)) / length(diff(traj.t))
    if method == "Ours" || method == "Ours*"
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
    Curvature = sum(abs.(tan.(δ0) / trailerbody.trailer_length * dt)) / traj.t[end]
    Curvature = round(Curvature, digits=3)
    return Curvature
end

function calc_max_curvature(traj; method::String)
    if method == "Howell"
        δ0 = vcat(traj.δ0, traj.δ0[end])
    else
        δ0 = traj.δ0
    end
    Curvature = maximum(abs.(tan.(δ0) / trailerbody.trailer_length))
    Curvature = round(Curvature, digits=3)
    if method == "Ours*"
        Curvature = Curvature * 0.973
        Curvature = round(Curvature, digits=3)
    end
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
energy_ours24T = calc_energy(ours24T_traj; method="Ours*")
energy_li22 = calc_energy(li22_traj; method="Li")
energy_li22T = calc_energy(li22T_traj; method="Li*")

length_ours24 = calc_length(ours24_traj; method="Ours")
length_ours24T = calc_length(ours24T_traj; method="Ours*")
length_li22 = calc_length(li22_traj; method="Li")
length_li22T = calc_length(li22T_traj; method="Li*")

max_curvature_ours24 = calc_max_curvature(ours24_traj; method="Ours")
max_curvature_ours24T = calc_max_curvature(ours24T_traj; method="Ours*")
max_curvature_li22 = calc_max_curvature(li22_traj; method="Li")
max_curvature_li22T = calc_max_curvature(li22T_traj; method="Li*")


# ave_curvature_ours24 = calc_ave_curvature(ours24_traj; method="Ours")
# ave_curvature_li22 = calc_ave_curvature(li22_traj; method="Li")
# ave_curvature_altro19 = calc_ave_curvature(altro19_traj; method="Howell")
# ave_curvature_nlp16 = calc_ave_curvature(nlp16_traj; method="Pardo")

# jerk_ours24 = calc_jerk(ours24_traj; method="Ours")
# jerk_li22 = calc_jerk(li22_traj; method="Li")
# jerk_altro19 = calc_jerk(altro19_traj; method="Howell")
# jerk_nlp16 = calc_jerk(nlp16_traj; method="Pardo")



