# Benchmark
TRO: Eira 2022 TRO
RAL: Li 2022 RAL
Altro: Howell 2019 IROS
## 分析
benchmark的步骤为：问题建模+优化求解
- 问题建模：【目标+约束】的构建，构建成某种形式的优化问题
    - SDP
    - NLP
- 优化求解：各种求解器
    - SDP：Ours, COPT, Mosek, COSMO， SCS etc. ProxSDP.jl与SumOfSquares.jl不兼容
    - NLP：Altro(问题建模+优化求解), iLQR, Ipopt, etc.
如何benchmark？
- 针对问题建模：对比SDP问题和NLP问题(Altro(IROS), Ipopt(RAL+TRO))
- 针对优化求解：对比我们的SDP求解器(PDHG)和其他SDP求解器(COPT,Mosek,etc)


下面仔细分析问题建模的处理
### 目标函数
- Ours目标函数是【多项式高阶导数的平方求积分】，等价于平坦化后线性系统的【控制能量cost】
- TRO目标函数是【终点cost+轨迹横向偏移总量cost+控制能量cost】
- RAL目标函数是【时间长度】，求解最短时间轨迹
- Altro可以自定义【各种非线性的目标函数】(也可以是【时间长度】)，但是一般来说是一个【LQR形式的cost=状态cost+控制能量cost】

需要统一目标函数才能进行对比，都统一成控制能量的cost


### 避障约束
- Ours避障约束是将其构建成【安全走廊】
- TRO避障约束时【大M法则+椭圆法则】，MILP时大M法则，NLP阶段时椭圆法则
    - 实际上是将避障约束划分成了：车道线约束（简单线性）+其他车辆约束，其他车辆约束采用椭圆法则
- RAL避障约束是【安全走廊+三角形法则】，LIOS是安全走廊，TRMO是三角形法则
    - 在TRMO阶段，采用了trust-region进行减枝
- Altro可以自定义【各种避障约束】，可以采用椭圆法则，也可以采用三角形法则
    - 阅读Altro的IROS2019的论文，可以用多个膨胀后的圆约束来近似所有的避障约束
    - 在汽车问题上，采用【椭圆法则】来构建避障约束?
        - 用膨胀的圆来近似其他车辆的轮廓，ego-car视为质点
    - 在卡车问题上，采用【三角形法则】来构建避障约束？
        - 但是如果不采用Trust-region进行减枝，没用的约束会很多
        - 因此还是采用【椭圆法则】？
        - 用多个膨胀后的圆约束来表示环境的障碍物，ego-trailer视为2个质点，前车质点的避障约束需要自定义

## 时间分配优化
我们的方法中有分段，所以有时间分配；
其他方法中没有分段，因此没有时间分配，不过可以优化总的时间长度
- Ours+PDHG: 分段，可以优化时间分配和总时间
- TRO：固定的时间长度
- RAL：可以优化总的时间长度，（将dt或者T设为优化变量）
- Altro：可以优化总的时间长度（但是没有找到具体的形式），但一般是固定的时间

## 卡车
- Ours + COPT, Mosek, etc.
- Ours + PDHG
- RAL
- Altro

## 汽车
- Ours + COPT, Mosek, etc.
- Ours + PDHG
- TRO
- Altro


## benchmark的指标

- Altro2019IROS：在多个机器人系统上进行benchmark，将ALTRO与IPOPT,SNOPT,案例-ILQR进行对比
  - 约束违反程度随着求解时间或者迭代次数的变化
  - 多种系统的轨迹问题上的求解时间对比
  - 实验：没有实物实验

- Eira2022TRO: 在4个经典的城市道路驾驶场景上进行benchmark，每个场景有1000个实例，对比不同的初始化策略，以及与纯NLP的方法进行对比
  - 初始化策略：4个启发式的初始化策略，3个MILP初始化策略的变体
    - Convergence rate
    - Cost Optimality, 与NLP的进行对比
    - Runtime，与NLP的进行对比
  - NLP Baseline Comparison:
    - Solved rate
    - Progress at 8s, 即8s时的行驶距离
    - Speed Tracking, 越接近8m/s越好
    - Absolute Jerk Value, 越小越好
    - 火焰图对比，横轴是指标，纵轴的百分比
  - 实验：没有实物实验

- Li2022RAL: 主要是2个场景case
  - 与其他规划器进行对比
    - This work
    - EHA：扩展混合A*算法
    - PCOC：逐渐约束的最优控制方法
    - CSOC：结合采样搜索和最优控制的方法
  - 不同初始化策略的对比，即该工作初始化策略的各种变体
    - Cost
    - Runtime
  - 求解时间随着trailer数量的变化
  - 时间惩罚的权重以及$\Delta s$对于求解时间的影响
  - 实验：没有实物实验

- Sun2021TRO： 随机生成了200个问题（或场景）
  - 有限差分梯度FD与分析梯度AG的对比
  - 与Joint Optimization的对比（NLP由SNOPT求解），同时优化空间参数和时间参数，不分层
  - 与Direct Collocation的对比（NLP由SNOPT求解），采用的是质点模型，时间也被视为决策变量。【由于formulation不一样，因此最终的目标函数值无法直接对比，之对比了成功率】
  - Scalability Study: 【不同规模】问题上的【不同求解器】的求解时间的对比，如Sqopt和Gurobi
  - 次梯度的影响
  - 实验：
    - 和Gao等人的工作（启发式的时间分配）的对比，轨迹的长度，飞行时长，以及急动度Jerk
    - 时间的权重对于求解时间的影响， 飞行时长， 计算时间，急动度Jerk
    - 跟踪动态的目标

- Wang2022TRO：
  - 大规模无约束最小化控制能量的问题的求解，对比了Proposed, Mellinger, Burke, Bry Dense和Bry Sparse的方法
    - 各种方法的求解时间随着问题规模的变化曲线
  - 有约束的优化问题的求解： Proposed*, Patterson*, Gao*, Deits*, Deits,Tordesillas*, Mellinger, Sun*， 
    - 轨迹形状的profile
    - 求解时间
    - 成功率，需要大量场景
    - 相对的控制能量，在固定时间的情况下计算和对比
    - 飞行时间
    - 轨迹状态的profile：速度和加速度随时间的变化，以及约束满足的情况
  - 实物实验1：大型地下停车场实验
    - 测试不同的最大速度约束条件下的轨迹，证明充分利用$v_{max},a_{max}$的可行空间
  - SE3空间的轨迹生成，不再将无人机视为质点，而是视为一个刚体，考虑了姿态的约束，在狭窄的空间进行轨迹优化，在不同宽度的狭窄通道上生成极限运动轨迹，并绘制其形状的profile和状态的profile，以及约束满足的情况，以及计算时间
  - 实物实验2：窗户飞行实验，绘制规划出来的轨迹曲线，以及实物实验的幻影图
    - 交互场景
    - 2个连续窗户场景
    - 3个连续窗户场景
 - 实验的描述：
    - 硬件条件：无人机+计算平台等
    - 约束数值：最大速度、最大加速度等


# 我们怎么Benchmark

## Numerical Simulations
- 计算平台： CPU + GPU
- 程序语言： Julia, 其他工作也用Julia复现或者调用其开源的代码

### Trailer
- 无其他车辆，只有固定的道路边界的障碍物
- 既Benchmark Formulation，又Benchmark Solver
- Formulation Benchmark：约束满足（避障+动力极限） + 轨迹质量（控制能量+平整度+空间长度+时间长度） + 求解时间（计算时长）
  - **Figure**: Geometrical Profile轨迹的几何形状，障碍物规避约束的可视化，【Formulation】（相同的SDP Formulation,不同SDP Solver求解出来的Geometrical Profile和Trajectory Profile基本一致），曲线图
  - **Figure**: Trajectory Profile轨迹的状态变化，速度，加速度，曲率约束的可视化，【Formulation】
  - **Figure-Table**: Trajectory Quality + Solve Time，轨迹质量的对比，【Formulation】，曲线图
  ，由于不同的Formulation，目标函数其实并不一致，因此轨迹质量的指标为：（控制能量的积分+曲率绝对值的积分+轨迹的空间长度+轨迹的时间长度+求解时间），曲线图或者柱状图
- Solver Benchmark：规模扩展性（不同规模问题的求解时间） + 时间分配参数的优化（分析梯度+数值梯度）
  - **Figure-Table**: Scalability Study, 不同规模问题上不同Solver的求解时间的对比，【Solver】，曲线图或这表格
  - **Figure-Table**: Analytical Gradient, 有限差分FD与分析梯度AG的对比（Runtime+Cost+最终的时间分配？），【Solver】，只选1种其他SDP求解器与我们的求解器进行对比即可，我们的SDP求解器是AG，其他SDP求解器是FD，表格或者曲线图，曲线图表示每次outer loop的时间分配参数的变化情况


### Car
- 既有固定的道路边界的障碍物，又有其他车辆的障碍物，因此我们的方法需要划分出更多的区段，其他车辆等障碍物的引入可以在论文衔接段强调一下
- Solver的Benchmark已经在Trailer上进行了，因此只需要在Formulation上进行benchmark，即与其他Car的TrajOpt论文的Formulation进行Benchmark
- Formulation Benchmark：约束满足（避障+动力极限） + 轨迹质量（控制能量+平整度+空间长度+时间长度） + 求解时间（计算时长）
  - **Figure**: Geometrical Profile轨迹的几何形状，障碍物规避约束的可视化，【Formulation】（相同的SDP Formulation,不同SDP Solver求解出来的Geometrical Profile和Trajectory Profile基本一致），曲线图
  - **Figure**: Trajectory Profile轨迹的状态变化，速度，加速度，曲率约束的可视化，【Formulation】
  - **Figure-Table**: Trajectory Quality + Solve Time，轨迹质量的对比，【Formulation】，曲线图
  ，由于不同的Formulation，目标函数其实并不一致，因此轨迹质量的指标为：（控制能量的积分+曲率绝对值的积分+轨迹的空间长度+轨迹的时间长度+求解时间），曲线图或者柱状图

## Physical Experiments
- 计算平台： NVIDIA Jetson Nano
- 程序语言： Julia
- Planner: Our TrajOpt
- Controller: Pure-Pursuit or MPC

### Trailer
- 固定的边界障碍物
- **Figure**: 实物实验的幻影图或者多帧平铺成大图并标上序号
- **Figure**: 规划出来的轨迹曲线图
- **Text**: 求解时间的数值以及规划轨迹的时间及空间长度，体现在线规划

### Car
- 移动的其他障碍物