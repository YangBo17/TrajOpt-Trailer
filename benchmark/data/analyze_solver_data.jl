using CSV, DataFrames
using Plots
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
copt_color = :purple
mosek_color = :royalblue

label_font = 16
tick_font = 14
legend_font = 16

df = CSV.read("benchmark/data/scalability_study.csv", DataFrame)

Cones = Int.(df[1,2:end] |> collect)
Ours_data = df[2,2:end] |> collect
COPT_data = df[3,2:end] |> collect
Mosek_data = df[4,2:end] |> collect

fig = plot(xlabel="Number of PSD Cones", ylabel="Computation Time (s)", yscale=:log10, labelfontsize=label_font, tickfontsize=tick_font, legendfontsize=legend_font, legend=:topleft)
plot!(fig, Cones, Ours_data, label="Ours", linewidth=2, color=ours_color, marker=:circle, markersize=5)
plot!(fig, Cones, COPT_data, label="COPT", linewidth=2, color=copt_color, marker=:utriangle, markersize=5)
plot!(fig, Cones, Mosek_data, label="Mosek", linewidth=2, color=mosek_color, marker=:diamond, markersize=5)

savefig(fig, "benchmark/Figs/benchmark_solver_pgf.tex")