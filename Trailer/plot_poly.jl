using Plots
using DynamicPolynomials
@polyvar t 

p1 = 0.05t^4-0.4t^3-1.2t^2-2.0t+1
p2 = 1+0.5t^2+0.8t^3+0.6t^4+t^8
plot([i for i in 0:0.1:10], [p1(t=>i)/sqrt(p2(t=>i)) for i in 0:0.1:10])