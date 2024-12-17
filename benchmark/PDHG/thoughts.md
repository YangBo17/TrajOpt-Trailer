# Matrix Inverse Lemma
$$
P = \begin{bmatrix}
A & B \\
C & D
\end{bmatrix}^{-1} = \begin{bmatrix}A^{-1} + A^{-1}B(D-CA^{-1}B)^{-1}CA^{-1} & -A^{-1}B(D-CA^{-1}B)^{-1} \\
-(D-CA^{-1}B)^{-1}CA^{-1} & (D-CA^{-1}B)^{-1}
\end{bmatrix}
$$
# Our PDHG Matrix
$$
E = I + \frac{1}{\beta} \begin{bmatrix}0 \\ I \\ 0\end{bmatrix}^\top 
\begin{bmatrix}2P & L^\top & H^\top \\ 
                L & -\frac{1}{\beta}I & 0 \\
                H & 0 & 0
\end{bmatrix}^{-1}
\begin{bmatrix}
0 \\ I \\ 0
\end{bmatrix} \\
e = \frac{1}{\beta} \begin{bmatrix}0 \\ I \\ 0\end{bmatrix}^\top 
\begin{bmatrix}2P & L^\top & H^\top \\ 
                L & -\frac{1}{\beta}I & 0 \\
                H & 0 & 0
\end{bmatrix}^{-1}
\begin{bmatrix}
0 \\ g \\ r
\end{bmatrix}
$$
# Convert into 2x2 block matrix
$$
E = I + \frac{1}{\beta} \begin{bmatrix}I \\ 0 \\ 0\end{bmatrix}^\top 
\begin{bmatrix}-\frac{1}{\beta}I & L & 0 \\ 
                L^\top & 2P & H^\top \\
                0 & H & 0
\end{bmatrix}^{-1}
\begin{bmatrix}
I \\ 0 \\ 0
\end{bmatrix} \\
e = \frac{1}{\beta} \begin{bmatrix}I \\ 0 \\ 0\end{bmatrix}^\top 
\begin{bmatrix}-\frac{1}{\beta}I & L & 0 \\ 
                L^\top & 2P & H^\top \\
                0 & H & 0
\end{bmatrix}^{-1}
\begin{bmatrix}
g \\ 0 \\ r
\end{bmatrix}
$$
# Corresponding matrix
$$
A = -\frac{1}{\beta}I \\
B = \begin{bmatrix} L & 0 \end{bmatrix} \\
C = \begin{bmatrix} L^\top \\ 0 \end{bmatrix} \\
D = \begin{bmatrix} 2P & H^\top \\ H & 0 \end{bmatrix}
$$
$$
E = I + \frac{1}{\beta} (A^{-1}+A^{-1}B(D-CA^{-1}B)^{-1}CA^{-1}) \\
e = \frac{1}{\beta} [(A^{-1}+A^{-1}B(D-CA^{-1}B)^{-1}CA^{-1}) \cdot g + (-A^{-1}B(D-CA^{-1}B)^{-1}) \cdot \begin{bmatrix}0 \\ r \end{bmatrix}]
$$
## Important middle matrix
$$
A^{-1} = -\beta I \\
K = (D-CA^{-1}B)^{-1} = (D + \beta C B) ^{-1} 
$$
## Final calculation of E and e 
$$
\begin{aligned}
E &= I + \frac{1}{\beta} (-\beta I + \beta^2 \cdot B K C ) \\
&= I + (-I + \beta BKC) \\
&= \beta BKC \\
e &= \frac{1}{\beta} [(-\beta I + \beta^2 \cdot B K C ) \cdot g + (\beta \cdot B K) \cdot \begin{bmatrix}0 \\ r \end{bmatrix}] \\
&= (-I + \beta \cdot BKC) \cdot g + BK \cdot \begin{bmatrix}0 \\ r \end{bmatrix}
\end{aligned}
$$

## TODO 0601

1. 【不需要】IRIS: 给trailer生成新的避障corridor,采用基于质点的方法，这样应该也可以优化我们的算法求解出来的轨迹，不置于拐弯比较奇怪
2. 【已解决，把车的场景换了之后，双层优化可以继续采用3段转弯的场景而不致于重复，求解时间也通过采用Matrix Inverse Lemma得到提升】现在双层优化的例子要么过于简单，要么复杂到解不出好的结果，但是复杂的求解时间慢
3. 【已解决，通过设计新的实验场景】对于复杂的汽车的场景，例如引入了3辆其他车辆，使得轨迹变为9段，添加所有的约束后，SOS无解，我们的求解器还没有对这种场景求解(可能可以求解出来)
4. 需要一个证明约束严格满足的例子
5. 需要一个CPU和GPU之间benchmark的表格？
6. 需要把收敛的gap曲线画出来？
7. 【已解决】目前MLP和Altro求解出来的轨迹有点奇怪，NLP终点扭曲严重，Altro控制能量太低了

## TODO 0602
1. 前面文字部分的修改
2. 仿真文字及部分图像的修改
3. bilevel time的进一步提升，把必要的计算都挪到GPU
4. 【大师兄说不加了，因为别人的收敛过程我们也没有】收敛的gap曲线绘制？
5. 【大师兄说不加了，因为我们的Solver在CPU上比别人的慢】CPU和GPU的benchmark？
6. 约束严格满足的例子？

## TODO 0603
1. 仿真数据的优化，仿真文字的修改，仿真图像的修改
2. 加入更多的求解器的数据
3. related works引入ddp,ilqr,cilqr，来自滴滴的建议
4. bilevel time的进一步提升，把必要的计算都挪到GPU
5. 写一下discussion？
6. 倒车的图放在文章第一页
7. 实验数据绘制的轨迹曲线
8. 更详细的硬件平台介绍
9. 问题参数的表格，比如约束极限的数值，以及优化目标的权重，时间间隔或离散点数量