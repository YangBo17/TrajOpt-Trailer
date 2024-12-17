## env
using JLD2
includet("../../common/common.jl")
@load "data/cons_limits2.jld2" cons_limits
includet("../../visual/visual_trailer.jl")
trailer_body = TrailerBodyReal()

# parameters
d0 = trailer_body.trailer_length
d1 = trailer_body.link_length
W = trailer_body.trailer_width

# trailer env
route_kargo = [-23341.2 7173.53;
                                -23323.6 7167.39;
                                -23306.7 7256.59;
                                -23298.5 7140.4;
                                -23279.9 7134.07;
                                -23263.4 7142.07;
                                -23256.2 7160.89;
                                -23256.6 7179.06;
                                -23285.4 7171.03;
                                -23307.6 7183.35;
                                -23335.5 7193.31] / 10.0

