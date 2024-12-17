##
using Revise
using DynamicPolynomials
using SumOfSquares
using JuMP, COPT
using LinearAlgebra
using Plots
using YAML

# settings of trailer
vehicle = YAML.load_file("vehicle.yml")
traielr_length = vehicle["trailer"]["length"]*vehicle["trailer"]["scale"]
trailer_width = vehicle["trailer"]["width"]*vehicle["trailer"]["scale"]
link_length = vehicle["trailer"]["link"]
car_length = trailer_length + link_length


