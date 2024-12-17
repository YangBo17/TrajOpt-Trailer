using CSV, DataFrames
using Plots, StatsPlots
using Statistics
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

label_font = 16
tick_font = 14
legend_font = 16

digits = 3

df = CSV.read("benchmark/data/sol_time_trailer.csv", DataFrame)

Ours_soltime = Float64[]
Li_soltime = Float64[]
NLP_soltime = Float64[]
Altro_soltime = Float64[]
for i in 1:120
    if i%4 == 1
        push!(Ours_soltime, df[i,2])
    elseif i%4 == 2
        push!(Li_soltime, df[i,2])
    elseif i%4 == 3
        push!(NLP_soltime, df[i,2])
    elseif i%4 == 0
        push!(Altro_soltime, df[i,2])
    end
end


@show Ours_soltime
@show Li_soltime
@show NLP_soltime
@show Altro_soltime

Ours_soltime_case1 = Ours_soltime[1:10]
Ours_soltime_case2 = Ours_soltime[11:20]
Ours_soltime_case3 = Ours_soltime[21:30]

Li_soltime_case1 = Li_soltime[1:10]
Li_soltime_case2 = Li_soltime[11:20]
Li_soltime_case3 = Li_soltime[21:30]

NLP_soltime_case1 = NLP_soltime[1:10]
NLP_soltime_case2 = NLP_soltime[11:20]
NLP_soltime_case3 = NLP_soltime[21:30]

Altro_soltime_case1 = Altro_soltime[1:10]
Altro_soltime_case2 = Altro_soltime[11:20]
Altro_soltime_case3 = Altro_soltime[21:30]

#
Ours_median_case1 = round(median(Ours_soltime_case1), digits=digits)
Ours_median_case2 = round(median(Ours_soltime_case2), digits=digits)
Ours_median_case3 = round(median(Ours_soltime_case3), digits=digits)

Ours_maximum_case1 = round(maximum(Ours_soltime_case1), digits=digits)
Ours_maximum_case2 = round(maximum(Ours_soltime_case2), digits=digits)
Ours_maximum_case3 = round(maximum(Ours_soltime_case3), digits=digits)

# Ours_quantile_case1 = round(quantile(Ours_soltime_case1, 0.9), digits=digits)
# Ours_quantile_case2 = round(quantile(Ours_soltime_case2, 0.9), digits=digits)
# Ours_quantile_case3 = round(quantile(Ours_soltime_case3, 0.9), digits=digits)

#
Li_median_case1 = round(median(Li_soltime_case1), digits=digits)
Li_median_case2 = round(median(Li_soltime_case2), digits=digits)
Li_median_case3 = round(median(Li_soltime_case3), digits=digits)

Li_maximum_case1 = round(maximum(Li_soltime_case1), digits=digits)
Li_maximum_case2 = round(maximum(Li_soltime_case2), digits=digits)
Li_maximum_case3 = round(maximum(Li_soltime_case3), digits=digits)

# Li_quantile_case1 = round(quantile(Li_soltime_case1, 0.9), digits=digits)
# Li_quantile_case2 = round(quantile(Li_soltime_case2, 0.9), digits=digits)
# Li_quantile_case3 = round(quantile(Li_soltime_case3, 0.9), digits=digits)

#
Altro_median_case1 = round(median(Altro_soltime_case1), digits=digits)
Altro_median_case2 = round(median(Altro_soltime_case2), digits=digits)
Altro_median_case3 = round(median(Altro_soltime_case3), digits=digits)

Altro_maximum_case1 = round(maximum(Altro_soltime_case1), digits=digits)
Altro_maximum_case2 = round(maximum(Altro_soltime_case2), digits=digits)
Altro_maximum_case3 = round(maximum(Altro_soltime_case3), digits=digits)

# Altro_quantile_case1 = round(quantile(Altro_soltime_case1, 0.9), digits=digits)
# Altro_quantile_case2 = round(quantile(Altro_soltime_case2, 0.9), digits=digits)
# Altro_quantile_case3 = round(quantile(Altro_soltime_case3, 0.9), digits=digits)

#
NLP_median_case1 = round(median(NLP_soltime_case1), digits=digits)
NLP_median_case2 = round(median(NLP_soltime_case2), digits=digits)
NLP_median_case3 = round(median(NLP_soltime_case3), digits=digits)

NLP_maximum_case1 = round(maximum(NLP_soltime_case1), digits=digits)
NLP_maximum_case2 = round(maximum(NLP_soltime_case2), digits=digits)
NLP_maximum_case3 = round(maximum(NLP_soltime_case3), digits=digits)

# NLP_quantile_case1 = round(quantile(NLP_soltime_case1, 0.9), digits=digits)
# NLP_quantile_case2 = round(quantile(NLP_soltime_case2, 0.9), digits=digits)
# NLP_quantile_case3 = round(quantile(NLP_soltime_case3, 0.9), digits=digits)



##

fig = plot(yscale=:log10)
boxplot!(fig, Ours_soltime_case1, label="Ours", color=ours_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)
boxplot!(fig, Li_soltime_case1, label="Li", color=li_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)
boxplot!(fig, Altro_soltime_case1, label="Howell", color=altro_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)
boxplot!(fig, NLP_soltime_case1, label="Pardo", color=nlp_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)

boxplot!(fig, Ours_soltime_case2, label="Ours", color=ours_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)
boxplot!(fig, Li_soltime_case2, label="Li", color=li_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)
boxplot!(fig, Altro_soltime_case2, label="Howell", color=altro_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)
boxplot!(fig, NLP_soltime_case2, label="Pardo", color=nlp_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)

boxplot!(fig, Ours_soltime_case3, label="Ours", color=ours_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)
boxplot!(fig, Li_soltime_case3, label="Li", color=li_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)
boxplot!(fig, Altro_soltime_case3, label="Howell", color=altro_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)
boxplot!(fig, NLP_soltime_case3, label="Pardo", color=nlp_color, boxalpha=0.5, whiskeralpha=0.5, markerstrokewidth=0, legend=false)




