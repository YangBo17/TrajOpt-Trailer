# LP
## Primal Problem (P)
$$
\begin{aligned}
\min_x \quad & z=c^T x \\
\text{s.t.} \quad & Ax = b \\
& x \geq 0
\end{aligned}
$$

## Dual Probelm (D)
$$
\begin{aligned}
\max_y \quad & b^T y \\
\text{s.t.} \quad & A^T y + s = c \\
& s \geq 0
\end{aligned}
$$


## Gradient
$$
\frac{\partial z}{\partial a_{i,j}} = -y_i  x_j
$$


# SDP
## Primal Problem (P)
$$
\begin{aligned}
\min_{X \in \mathbb{S}_n^+} \quad & z = tr(CX) \\
\text{s.t.} \quad & tr(A_kX) = b_k, \; k = 1,\cdots,m \\
& X \geq 0 
\end{aligned}
$$
where $C \in \mathbb{S}_n, A_k \in \mathbb{S}_n$

## Dual Probelm (D)
$$
\begin{aligned}
\max_{y_k} \quad & \sum_{k=1}^m b_k y_k \\
\text{s.t.} \quad& \sum_{k=1}^m A_k y_k + S = C \\
& S \geq 0
\end{aligned}
$$
where $S \in \mathbb{S}_n$

## Conjecture
$$
\frac{\partial z}{A_k} = -y_k  X
$$

- **已在LP和SDP问题中验证上述的灵敏度计算成立**
- **已在【具有与TrajOPt相同形式的随即生成的SDP问题】中验证上述对于L和H矩阵的灵敏度计算成立**

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

## 计算目标函数对于A的灵敏度
$$
\frac{\partial z}{\partial a_{i,j}} = -u_i x_j
$$

- **以验证上述灵敏度的结论在QP问题上成立**
- **对于不等式约束的灵敏度，只有active的约束才可计算，不active的约束不可计算**


# 理解
## Conic Progamming (P)

$$
\begin{aligned}
    \min_x \quad & c^T x \\
    \text{s.t.} \quad & Ax + s= b  \\
    & s \geq \mathcal{K}
\end{aligned}
$$

## Conic Progamming (D)
$$
\begin{aligned}
    \max_y \quad & b^T y \\
    \text{s.t.} \quad & A^T y + c = 0 \\
    & c \in \mathcal{K}^*
\end{aligned}
$$

## Lagrangian of Conic Programming
$$
    \mathcal{L}(x,s,y) = c^T x + y^T (Ax + s - b)
$$
- 拉格朗日函数对A求导
$$
    \frac{\partial \mathcal{L}}{\partial A} = y x^T
$$
- 这里没有负号，是因为拉格朗日函数也可以是另一种形式
$$
    \mathcal{L}(x,s,y) = c^T x + y^T (b - Ax - s) 
$$
$$
    \frac{\partial \mathcal{L}}{\partial A} = -y x^T
$$
由此可以推广到其他任何参数的灵敏度

## General Progamming Problem
- 对于一般的优化问题
$$
\begin{aligned}
 \min_x \quad & f(x;\theta) \\
 \text{s.t.} \quad & h(x;\theta) = 0 \\
    & g(x;\theta) \leq 0
\end{aligned}
$$
- 仅考虑active的约束，则简化为仅有等式约束的情况
$$
\begin{aligned}
 \min_x \quad & f(x;\theta) \\
 \text{s.t.} \quad & h(x;\theta) = 0 
\end{aligned}
$$

## Lagragian
$$
\mathcal{L}(x,u,v;\theta) = f(x;\theta) + u^T h(x;\theta) + v^T g(x;\theta)
$$
- 拉格朗日函数对$\theta$求导
$$
\frac{\partial \mathcal{L}(x,u,v;\theta)}{\partial \theta} = \frac{\partial f(x;\theta)}{\partial \theta} + u^T \frac{\partial h(x;\theta)}{\partial \theta} + v^T \frac{\partial g(x;\theta)}{\partial \theta}
$$

## TRO2021
- TRO2021: Fast UAV Trajectory Optimization Using Bilevel Optimization With Analytical Gradients
  - 该文中计算梯度的方式为，这里$y$是问题参数，$\lambda,\nu$是对偶变量，我将原文公式直接写出，而没有匹配我上述的符号系统
  $$
  \nabla_y J^*(y) = \nabla_y J + \sum_{i=1}^m \lambda_i(y) \nabla_y g_i + \sum_{j=1}^p \nu_j(y) \nabla_y h_j
  $$
  - 其实，这就是我上面导出的那个灵敏度分析的公式
- 这个灵敏度分析的公式是普适性的，对于NLP也成立，实际上，这个公式也被称为NLP的灵敏度公式

## Important Fact
- 然而，这个公式考虑的因素其实是简化了的，因为最优解$(x,u,v)$实际上也是和参数$\theta$有关的，都是在特定的$\theta$下求解出来的，因此也应当视为参数的函数，即$(x(\theta),u(\theta),v(\theta))$
$$
\mathcal{L}(x(\theta),u(\theta),v(\theta);\theta) = f(x(\theta);\theta) + u(\theta)^T h(x(\theta);\theta) + v(\theta)^T g(x(\theta);\theta)
$$
- 该拉格朗日函数对$\theta$求导
$$
\begin{aligned}
\frac{\partial \mathcal{L}(x(\theta),u(\theta),v(\theta);\theta)}{\partial \theta} &= \frac{\partial f(x(\theta);\theta)}{\partial \theta} + \frac{\partial [u(\theta)^\top h(x(\theta);\theta)]}{\partial \theta} + \frac{\partial [v(\theta)^T g(x(\theta);\theta)]}{\partial \theta} \\
&= \frac{\partial f(x(\theta);\theta)}{\partial \theta} + u(\theta)^\top \frac{\partial h(x(\theta);\theta)}{\partial \theta} + \frac{\partial u(\theta)}{\partial \theta}^\top h(x(\theta);\theta) + \\
& \quad v(\theta)^\top \frac{\partial g(x(\theta);\theta)}{\partial \theta} + \frac{\partial v(\theta)}{\partial \theta}^\top g(x(\theta);\theta)
\end{aligned}
$$
- NLP的灵敏度公式用我的符号系统表示是
$$
\begin{aligned}
\frac{\partial \mathcal{L}(x,u,v;\theta)}{\partial \theta} &= \frac{\partial f(x;\theta)}{\partial \theta} + \frac{\partial [u^\top h(x;\theta)]}{\partial \theta} + \frac{\partial [v^T g(x;\theta)]}{\partial \theta} \\
&= \frac{\partial f(x;\theta)}{\partial \theta} + u^\top \frac{\partial h(x;\theta)}{\partial \theta} + v^\top \frac{\partial g(x;\theta)}{\partial \theta} 
\end{aligned}
$$
- 我们的方法是考虑了参数$\theta$对原对偶最优解$(x,u,v)$的影响，然后再导出参数$\theta$对于拉格朗日函数$\mathcal{L}$的影响
- NLP灵敏度公式是直接考虑参数$\theta$对拉格朗日函数$\mathcal{L}$的影响
- 两种不同的方法的本质区别类似于不同阶次的泰勒展开？我们考虑了更高阶的影响？
- 如果采用NLP的公式，那么似乎会显得我们相比于TRO2021那篇文章在解析梯度方面没有新的贡献
- 【思考：Sensitivity Analysis & Differentiable Optimization】


***
# TRO2021引用的NLP灵敏度分析的文献
- A.【博士论文】K. Jittorntrum, “Sequential algorithms in nonlinear programming,” Ph.D. dissertation, Australian Nat. Univ., Canberra, Australia, 1978.
- B.【Anthony V. Fiacco，灵敏度分析领域的大佬】Introduction to Sensitivity and Stability Analysis in Nonlinear Programming

## 概述 A.
- Mathematical Programming Problem (MPP)
- Jacobian uniqueness condition
- Theorem 2.3.1
  - 具有strict complementary的条件
  - 只考虑$x(\theta)$，没考虑$u(\theta),v(\theta)$
  - 且认为$\frac{d h}{d \theta}=0, \frac{d g}{d \theta}=0$，基于non-degenerate的假设
- Theorem 2.3.2
  - 没有strict complementary的条件
  - 将Jacobian uniqueness condition替换为regular solution condition
- Theorem 2.3.3 (TRO2021引用的定理)
  - 相比于Theorem 2.3.1移除了strict complementary的条件

## 概述 B.
- 第三章，定理3.2.2 and 3.4.1
  - 通过对KKT条件求导得到一阶甚至更高阶的导数

## 懂了
- 我的那个梯度计算公式通过KKT条件简化后就是NLP的灵敏度公式


***
***
***




## KKT Conditions
$$
\begin{aligned}
tr(A_k X) &= b_k, k=1,\cdots,m \\
\sum_{k=1}^m A_k y_k + S &= C \\
X &\geq 0 \\
S &\geq 0 \\
X S &= 0
\end{aligned}
$$

# Our Problem
## SDP Problem1
$$
\begin{aligned}
\min_{c,X} \quad & c^T P c \\
\text{s.t.} \quad & H c = r \\
& L c - g = M(X) \\
& X \geq 0
\end{aligned}
$$
## SDP Problem3
$$
\begin{aligned}
\min_{c,s} \quad & c^T P c \\
\text{s.t.} \quad & H c = r \\
& L c = g + s \\
& s \in \mathcal{K}
\end{aligned}
$$
## Lagragian
$$
\begin{aligned}
\mathcal{L}(c,s,d_1,d_2,\lambda) &= c^\top P c + d_1^\top (L c - s - g) + d_2^\top (H c - r) + \lambda^\top s \\
\frac{\partial \mathcal{L}}{\partial c} &= 2Pc + L^\top d_1 + H^\top d_2 = 0 \\
\frac{\partial \mathcal{L}}{\partial s} &= -d_1 + \lambda = 0
\end{aligned}
$$
即为
$$
\begin{pmatrix}
  L^T & H^T \\
  0 & -I \\
\end{pmatrix}\begin{pmatrix} d_1 \\ d_2 \end{pmatrix} =
\begin{pmatrix} -2Pc \\ -\lambda \end{pmatrix}
$$
甚至只考虑第一个等式
$$
\begin{pmatrix}
  L^T & H^T 
\end{pmatrix}\begin{pmatrix} d_1 \\ d_2 \end{pmatrix} =
-2Pc
$$

## SDP Problem2
$$
\begin{aligned}
\min_{c,s,X} \quad & c^T P c \\
\text{s.t.} \quad & H c = r \\
& L c - g = s \\
& M(X) = s \\
& \quad \quad \text{$\lambda$ is the corresponding dual varaible}\\
& X \geq 0
\end{aligned} 
$$

## Lagragian
$$
L(c,s,d_1,d_2,\lambda,X,Z) = c^TPc + d_1^T (Lc-s-g) + d_2^T (Hc-r) + \lambda^T (M(X)-s) - Tr(ZX)
$$
## KKT Conditions for only Equality Constraints
$$
\begin{aligned}
\begin{pmatrix}2P & 0 & L^\top & H^\top \\
             0 & 0 & -I & 0 \\
             L & -I & 0 & 0 \\
             H & 0 & 0 & 0
\end{pmatrix} \begin{pmatrix} c \\ s \\ d_1 \\d_2 \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \\ g \\ r \end{pmatrix}
\end{aligned}
$$

## Schur Complement
$$
\begin{aligned}
\min_{c,X} \quad & \rho \\
\text{s.t.} \quad & \begin{bmatrix} \rho & c^T \\ c & P^{-1}\end{bmatrix} \geq 0 \\
& \alpha \geq 0 \\
& H c = r \\
& L c - g = M(X) \\
& X \geq 0
\end{aligned}
$$

## QP KKT
$$
\begin{aligned}
\frac{\partial\begin{pmatrix}c^\star \\ \epsilon^\star\end{pmatrix}}{\partial T} &= 
\frac{\partial \begin{pmatrix}2P(T) & R(T)^\top \\ R(T) & 0 \end{pmatrix}^{-1}\begin{pmatrix} 0 \\ q(T) \end{pmatrix}}{\partial T} \\
& = \frac{\partial \begin{pmatrix}2P(T) & R(T)^\top \\ R(T) & 0 \end{pmatrix}^{-1}}{\partial T}\begin{pmatrix} 0 \\ q(T) \end{pmatrix} + \begin{pmatrix}2P(T) & R(T)^\top \\ R(T) & 0 \end{pmatrix}^{-1} \frac{\partial \begin{pmatrix} 0 \\ q(T) \end{pmatrix}}{\partial T}\\
&=-\begin{pmatrix}2P(T) & R(T)^\top \\ R(T) & 0 \end{pmatrix}^{-1} \frac{\partial \begin{pmatrix}2P(T) & R(T)^\top \\ R(T) & 0 \end{pmatrix}}{\partial T} \begin{pmatrix}2P(T) & R(T)^\top \\ R(T) & 0 \end{pmatrix}^{-1} \begin{pmatrix} 0 \\ q(T) \end{pmatrix} \\
& \quad + \begin{pmatrix}2P(T) & R(T)^\top \\ R(T) & 0 \end{pmatrix}^{-1} \frac{\partial \begin{pmatrix} 0 \\ q(T) \end{pmatrix}}{\partial T}
\end{aligned}
$$