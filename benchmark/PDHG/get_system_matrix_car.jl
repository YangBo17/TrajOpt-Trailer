function get_system_matrix()
    Ac = [
    0 1 0 0 0 0;
    0 0 1 0 0 0;
    0 0 0 0 0 0;
    0 0 0 0 1 0;
    0 0 0 0 0 1;
    0 0 0 0 0 0
    ]
    
    Bc = [
    0 0;
    0 0;
    1 0;
    0 0;
    0 0;
    0 1
    ]
    
    P = [Bc[:, 1] Ac*Bc[:, 1] Ac^2*Bc[:,1] Bc[:, 2] Ac*Bc[:, 2] Ac^2*Bc[:, 2]]
    S = P^(-1)
    S = [S[3,:]' ; S[3,:]'*Ac; S[3,:]'*Ac^2; S[6,:]' ; S[6,:]'*Ac; S[6,:]'*Ac^2]
    
    G = S * Ac * (S)^(-1)
    H = S * Bc
    for i in eachindex(G)
        if abs(G[i])<10^(-10)
            G[i]=0
        end
    end
    for i in eachindex(H)
        if abs(H[i])<10^(-10)
            H[i]=0
        end
    end
    
    return Ac,Bc,G,S,H
end