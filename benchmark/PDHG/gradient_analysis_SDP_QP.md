# SDP Problem
$$
\begin{aligned}
   \min_{c,X} \quad&  c^\top P c \\
   \text{s.t.} \quad&  H c = r \\
   & L c - M(X) = g \\
   & X \in \mathbb{S}_+
\end{aligned}
$$
## SDP Lagrangian
$$
  \mathcal{L}(c,X,\lambda,\nu,Y) =  c^\top P c + \lambda^\top (Lc - M(X) - g) + \nu^\top (H c - r) - tr(YX) \\
$$

## SDP KKT Conditions
$$
\begin{aligned}
  2 P c + L^\top \lambda + H^\top \nu &= 0 \\
  -\nabla_X(M(X))^\top \lambda - Y &= 0 \\
  Lc - M(X) - g &= 0 \\
  Hc - r &= 0 \\
  tr(YX) &= 0 \\
  X &\in \mathbb{S}_+ \\
Y &\in \mathbb{S}_+
\end{aligned}
$$


# QP Problem
$X$ is fixed, $s=M(X)$
$$
\begin{aligned}
  \min_c \quad& c^\top P c \\
  \text{s.t.} \quad& L c - s = g \\
  & H c = r
\end{aligned}
$$

## QP Lagrangian
$$
  \mathcal{L}(c,\lambda,\nu) = c^\top P c + \lambda^\top (Lc - s - g) + \nu^\top (H c - r) 
$$

## QP KKT Conditions
$$
\begin{aligned}
  2Pc + L^\top \lambda + H^\top \nu &= 0 \\
  Lc - s - g &= 0 \\
    Hc - r &= 0
\end{aligned}
$$

# Verify the following KKT conditions
## SDP
$$
\begin{aligned}
  2 P c + L^\top \lambda + H^\top \nu &= 0 \\
  Lc - M(X) - g &= 0 \\
  Hc - r &= 0 \\
\end{aligned}
$$
## QP
$$
\begin{aligned}
  2 P c + L^\top \lambda + H^\top \nu &= 0 \\
  Lc - s - g &= 0 \\
  Hc - r &= 0 \\
\end{aligned}
$$

