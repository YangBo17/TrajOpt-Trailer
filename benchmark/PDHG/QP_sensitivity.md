# QP
## Primal Problem (P)
$$
\begin{aligned}
\min_{x} \quad & \frac{1}{2} x^T Q x + c^T x \\
\text{s.t.} \quad & A x = b \\
& G x \leq h 
\end{aligned}
$$
Lagrangian:
$$
L(x,u,v) = \frac{1}{2} x^T Q x + c^T x + u^T (A x - b) + v^T (G x - h)
$$

## Dual Problem (D)
$$
\begin{aligned}
\max_{u,v} \quad & -\frac{1}{2} (c + A^\top u + G^\top v)^T Q^{-1} (c + A^\top u + G^\top v) - u^\top b - v^\top h \\
\text{s.t.} \quad & u \geq 0
\end{aligned}
$$

# QP with slack variable
## Primal Problem (P)
$$
\begin{aligned}
\min_{x,s} \quad & \frac{1}{2}x^\top Q x + c^\top x \\
\text{s.t.} \quad & A x = b \\
& G x + s = h \\
& s >= 0 
\end{aligned}
$$

## Lagrangian
$$
L(x,s,u,v,\lambda) = \frac{1}{2}x^\top Q x + c^\top x + u^\top (Ax - b) + v^\top (Gx + s - h) + \lambda^T s
$$
- $A$的灵敏度计算
$$
\frac{\partial L}{\partial A} = u  x^\top
$$
### KKT 
$$
\begin{aligned}
\frac{\partial L}{\partial x} = Qx + c + A^\top u + G^\top v &= 0 \\
\frac{\partial L}{\partial s} = v + \lambda &= 0 \\
Ax &= b \\
Gx + s &= h \\
\lambda^T s &= 0 \\
s &>= 0 \\
\lambda &>= 0 
\end{aligned}
$$

$$
\begin{aligned}
\frac{\partial L}{\partial A} &= Qx\frac{\partial x}{\partial A} + c \frac{\partial x}{\partial A} + A^Tu\frac{\partial x}{\partial A} + G^\top v \frac{\partial x}{\partial A} + (v + \lambda)\frac{\partial s}{\partial A} \\
& \quad + (Ax-b)\frac{\partial u}{\partial A} + (Gx+s-h)\frac{\partial v}{\partial A} + s\frac{\partial \lambda}{\partial A} + ux^T \frac{\partial A}{\partial A} \\
&= ux^T
\end{aligned} 
$$

$$
\begin{aligned}
\frac{\partial L}{\partial G} &= Qx\frac{\partial x}{\partial G} + c \frac{\partial x}{\partial G} + A^Tu\frac{\partial x}{\partial G} + G^\top v \frac{\partial x}{\partial G} + (v + \lambda)\frac{\partial s}{\partial G} \\
& \quad + (Ax-b)\frac{\partial u}{\partial G} + (Gx+s-h)\frac{\partial v}{\partial G} + s\frac{\partial \lambda}{\partial G} + vx^T \frac{\partial G}{\partial G} \\
&= vx^T
\end{aligned} 
$$
