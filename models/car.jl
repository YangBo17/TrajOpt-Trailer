"""
Differential Flat Vehicle Model
Based on the Han 2022 paper: https://arxiv.org/abs/2208.13160
"""

using DifferentialEquations
using Parameters

# x = (x, y, θ, v), u = (ψ, a_t), (a_n, κ) is extended state (no derivative)
struct State
    x::Float64
    y::Float64
    θ::Float64
end

struct AllState
    t::Float64
    x::Float64
    y::Float64
    θ::Float64
    v::Float64
    ψ::Float64
end

struct StateDerivative
    ẋ::Float64
    ẏ::Float64
    θ̇::Float64
end

struct Input
    ψ::Float64
    v::Float64
end

@with_kw struct Body
    L::Float64 = 0.353
    W::Float64 = 0.264
end

@with_kw struct BodyReal
    L::Float64 = 0.353 * 9
    W::Float64 = 0.264 * 9
end

@with_kw mutable struct MyCar
    body::Body = Body()
    state::State
    input::Input
    time::Float64 = 0.
end

struct DataPoint 
    s::State
    u::Input
end

function to_array(s::State)
    return [s.x, s.y, s.θ]
end

function to_array(st::StateDerivative)
    return [st.ẋ, st.ẏ, st.θ̇]
end

function to_array(u::Input)
    return [u.ψ, u.v]
end

function dynamics(s::Array, body::Body, u::Array)
    θ = s[3]
    v = s[4]
    ψ = u[1]
    v = u[2]
    ẋ = v * cos(θ)
    ẏ = v * sin(θ)
    θ̇ = 1/body.L * v * tan(ψ)
    return StateDerivative(ẋ, ẏ, θ̇)
end

function car_step(mycar::MyCar, u::Input, dt::Float64)
    s = mycar.state
    # p = mycar.body
    mycar.input = u
    mycar.time += dt
    tspan = (0.0, dt)
    function ode_dyn!(ds::Array, s::Array, u::Array, t)
        st_deriv = dynamics(s, mycar.body, u)
        ds[1:end] = to_array(st_deriv)
    end
    prob = ODEProblem(ode_dyn!, to_array(s), tspan, to_array(u))
    sol = DifferentialEquations.solve(prob)
    x = sol(dt)[1]
    y = sol(dt)[2]
    θ = sol(dt)[3]
    new_state = State(x,y,θ)
    mycar.state = new_state
end
