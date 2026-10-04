# FDGA*方法

## A. 方法概述

本文提出一种快速行进方向引导A*算法（Fast-Marching Direction-Guided A*，FDGA*），用于复杂城市环境中的无人机三维航线规划。该方法以欧式符号距离场（Euclidean Signed Distance Field，ESDF）描述障碍物的空间分布及净空距离，并将ESDF转换为快速行进法（Fast Marching Method，FMM）的传播速度场。随后，以目标点为波源求解全局到达时间场，从中提取到达时间单调下降的参考路径，并通过空间膨胀构建引导搜索走廊。在该走廊内，FDGA*以栅格位置和入射方向共同构成搜索状态，将航段长度、障碍物净空风险和离散曲率纳入统一代价函数，同时施加最小安全距离与最大转角约束，从而获得安全且具有良好几何连续性的三维航线。

FMM在FDGA*中不直接输出最终航线，而是用于生成障碍感知的全局引导信息；最终航线由方向感知A*在受限搜索走廊内获得。该框架利用FMM减少大规模三维空间中的无效扩展，并利用方向状态和几何约束改善传统栅格搜索路径频繁折转的问题。算法主要包括局部规划空间构建、基于ESDF的FMM到达时间场求解、FMM引导走廊生成和方向感知约束搜索四个阶段。

---

## B. 局部三维规划空间

设无人机起点与目标点分别为$\mathbf p_s$和$\mathbf p_g$。为控制城市级三维栅格中的计算规模，本文以起终点的轴对齐包围盒为基础构建局部规划区域，并沿三个坐标轴向外扩展距离$p_b$，随后将扩展区域裁剪至ESDF的有效空间范围内。本文取$p_b=60\ \mathrm m$。

局部规划栅格间距根据起终点的欧式距离自适应确定，并被限制在$5$～$15\ \mathrm m$。该设计能够在较短航线中保留较细的空间结构，同时在长距离任务中控制三维状态数量。

上述局部规划空间及自适应栅格间距定义如下：

$$
\mathbf p_s=(x_s,y_s,z_s)^\mathrm T,\qquad
\mathbf p_g=(x_g,y_g,z_g)^\mathrm T,
\tag{1}
$$

$$
\Omega_L=
\operatorname{Clip}\left[
\operatorname{AABB}(\mathbf p_s,\mathbf p_g)\oplus p_b,
\Omega_D
\right],
\tag{2}
$$

$$
\Delta=
\operatorname{clip}\left(
\frac{\|\mathbf p_s-\mathbf p_g\|_2}{250},
5,15
\right),
\tag{3}
$$

其中，$\Omega_L$为局部规划空间，$\Omega_D$为ESDF的有效空间范围，$\oplus p_b$表示包围盒沿各方向扩展$p_b$，$\Delta$为局部栅格间距，且$\operatorname{clip}(x,a,b)=\min[\max(x,a),b]$。

---

## C. 基于ESDF的FMM全局引导

### 1) ESDF速度场

ESDF值用于刻画空间位置与最近障碍物之间的欧式距离。自由空间、障碍物边界和障碍物内部分别对应正值、零值和负值。为使FMM传播同时反映路径长度与障碍物净空，本文将ESDF映射为归一化传播速度场。

对于障碍物内部或ESDF无效的位置，将传播速度设置为接近零的正常数，以阻止波前通过。对于自由空间，当净空距离小于预设安全距离时，传播速度随净空距离线性增大；当净空距离达到安全距离后，传播速度饱和为1。因此，FMM波前在远离障碍物的区域传播较快，而在靠近障碍物的区域传播较慢，使后续到达时间场具备障碍物风险感知能力。本文设置最小安全距离$d_{\mathrm{safe}}=5\ \mathrm m$，并采用$\varepsilon_o=10^{-6}$和$\varepsilon_f=10^{-3}$保证数值稳定性。

ESDF符号关系与传播速度场定义如下：

$$
D(\mathbf p)
\begin{cases}
>0, & \mathbf p\ \text{位于自由空间},\\
=0, & \mathbf p\ \text{位于障碍物边界},\\
<0, & \mathbf p\ \text{位于障碍物内部},
\end{cases}
\tag{4}
$$

$$
F(c)=
\begin{cases}
\varepsilon_o,
& D(\mathbf p_c)<0\ \text{或}\ D(\mathbf p_c)\ \text{无效},\\[2mm]
\displaystyle
\max\left\{
\min\left[
\frac{D(\mathbf p_c)}{d_{\mathrm{safe}}},1
\right],
\varepsilon_f
\right\},
& D(\mathbf p_c)\ge 0,
\end{cases}
\tag{5}
$$

其中，$D(\mathbf p)$为位置$\mathbf p$处的ESDF值，$c$为局部栅格，$\mathbf p_c$为栅格中心位置，$F(c)$为相应的FMM传播速度。

### 2) 到达时间场求解

本文以目标栅格$c_g$为波源，利用FMM反向计算局部规划空间中的到达时间场。由于传播速度由ESDF构建，所得到达时间不仅反映位置与目标之间的空间距离，还编码了障碍物分布和净空风险。在低净空区域，较低的传播速度会产生较大的累计到达时间，从而使全局引导方向倾向于远离障碍物。

在数值求解过程中，FMM采用三维六邻域进行波前传播。对于待更新栅格，算法从三个坐标轴方向的已接受邻居中分别选取最小到达时间，并采用一阶迎风格式求解候选值。所有栅格按照到达时间从小到大依次由Far状态转为Trial状态，最终转为Accepted状态。当起点栅格$c_s$的到达时间为无穷大时，说明当前局部规划空间内不存在连接起点与目标点的可行通路，算法返回规划失败。

到达时间场满足Eikonal方程：

$$
F(\mathbf p)\|\nabla T(\mathbf p)\|=1,
\qquad T(\mathbf p_g)=0,
\tag{6}
$$

其中，$T(\mathbf p)$为位置$\mathbf p$至目标点的累计到达时间。将待更新栅格三个坐标轴方向上的有限邻居值按升序排列为$a_1\le a_2\le\cdots\le a_k$，$k\le3$，其一阶迎风离散形式为

$$
\sum_{i=1}^{k}[T(c)-a_i]^2
=
\left(\frac{\Delta}{F(c)}\right)^2.
\tag{7}
$$

为简化表达，定义

$$
A=\sum_{i=1}^{k}a_i,\qquad
B=\sum_{i=1}^{k}a_i^2-
\left(\frac{\Delta}{F(c)}\right)^2,
\tag{8}
$$

则候选到达时间为

$$
T(c)=\frac{A+\sqrt{A^2-kB}}{k}.
\tag{9}
$$

---

## D. FMM引导搜索走廊

获得到达时间场后，本文从起点栅格开始，在三维26邻域中反复选择到达时间最小且满足严格下降条件的相邻栅格，直至到达目标栅格，从而得到一条FMM参考路径。该过程采用邻域到达时间下降近似沿到达时间场的负梯度方向回溯，无需显式计算连续梯度。严格下降条件用于避免数值误差导致的循环或停滞；当不存在满足条件的相邻栅格时，算法判定参考路径提取失败。

FMM参考路径主要用于提供全局可行方向。为避免将后续A*严格限制在单一离散路径上，本文以参考路径上的每个栅格为中心进行离散球形膨胀，将膨胀区域的并集作为FDGA*的有效搜索走廊。本文取走廊膨胀半径$r_c=5$个局部栅格。该走廊既保留了对参考路径进行局部优化的空间，又能排除与全局可行方向关联较弱的区域，从而减少无效节点扩展。

参考路径的离散下降规则、严格下降条件和走廊定义如下：

$$
c_{k+1}=
\underset{c'\in\mathcal N_{26}(c_k)}{\arg\min}\ T(c'),
\tag{10}
$$

$$
T(c_{k+1})<T(c_k)-\varepsilon_T,
\qquad \varepsilon_T=10^{-6},
\tag{11}
$$

$$
\mathcal P_F=\{c_s,c_1,\ldots,c_g\},
\tag{12}
$$

$$
\mathcal C=
\bigcup_{c\in\mathcal P_F}
\left\{
c+\mathbf o\ \middle|\
o_x^2+o_y^2+o_z^2\le r_c^2
\right\},
\tag{13}
$$

其中，$\mathcal N_{26}(c_k)$为栅格$c_k$的三维26邻域，$\mathcal P_F$为FMM参考路径，$\mathbf o=(o_x,o_y,o_z)$为离散栅格偏移，$\mathcal C$为膨胀后得到的引导搜索走廊。

---

## E. 方向感知A*搜索

### 1) 扩展状态

传统A*通常仅以栅格位置作为搜索状态，无法区分以不同方向到达同一位置时产生的后续转向差异。为此，FDGA*采用“栅格位置—入射方向”联合状态。三维26邻域中的每个非零偏移对应一种离散入射方向，起点则使用特殊的无方向标签。

同一栅格以不同方向到达时被视为不同搜索状态，并分别维护累计代价与父状态。该表示方式可避免仅保留同一位置的最低代价标签而过早舍弃具有更优后续转向条件的路径，同时使算法能够在搜索过程中显式计算相邻航段的转角和曲率代价。

方向感知状态定义如下：

$$
s=(c,q),
\tag{14}
$$

$$
c\in\mathcal C,\qquad
q\in\{0,1,\ldots,25\}\cup\{q_{\varnothing}\},
\tag{15}
$$

$$
g(c,q_1)\neq g(c,q_2),
\qquad q_1\neq q_2,
\tag{16}
$$

其中，$c$为走廊内的三维栅格，$q$为进入该栅格时的离散方向，$q_{\varnothing}$为起点的无方向标签，$g(c,q)$为对应状态的累计代价。

### 2) 航段安全约束与净空风险

候选状态扩展采用三维26邻域。为避免仅检查候选栅格中心而导致连接航段穿越障碍物，FDGA*沿每条候选航段进行均匀ESDF采样。采样间隔根据局部栅格间距和最小安全距离共同确定，并设置$0.5\ \mathrm m$的下限。只有当航段上包括两个端点在内的全部采样点均满足最小安全距离要求时，该状态转移才被接受。

在硬安全约束之外，本文还根据航段沿线ESDF倒数的均值构造净空风险代价。该代价随着航段与障碍物距离的减小而增大，使FDGA*能够在均满足最低安全要求的候选航段之间进一步偏向安全余量更大的区域。计算中采用$\varepsilon_d=0.1\ \mathrm m$作为数值保护参数。

候选航段长度、采样间隔、采样数量和采样位置分别定义为

$$
L_{ij}=\|\mathbf p_j-\mathbf p_i\|_2,
\tag{17}
$$

$$
\delta_e=
\max\left\{
0.5,
\min\left(0.25\Delta,0.5d_{\mathrm{safe}}\right)
\right\},
\tag{18}
$$

$$
N_{ij}=\max\left[
1,
\left\lceil\frac{L_{ij}}{\delta_e}\right\rceil
\right],
\tag{19}
$$

$$
\mathbf p_{ij}^{k}
=
\mathbf p_i+
\frac{k}{N_{ij}}(\mathbf p_j-\mathbf p_i),
\quad k=0,\ldots,N_{ij}.
\tag{20}
$$

航段安全约束为

$$
D(\mathbf p_{ij}^{k})\ge d_{\mathrm{safe}},
\qquad \forall k=0,\ldots,N_{ij},
\tag{21}
$$

净空风险代价定义为

$$
R_{ij}=
\frac{L_{ij}}{N_{ij}+1}
\sum_{k=0}^{N_{ij}}
\frac{1}{\max[D(\mathbf p_{ij}^{k}),\varepsilon_d]}.
\tag{22}
$$

### 3) 最大转角约束与曲率代价

FDGA*根据当前状态记录的入射方向和候选航段方向计算相邻航段的夹角，并在状态扩展阶段施加最大转角硬约束。转角超过阈值的候选状态被直接舍弃。本文设置最大允许转角为$60^\circ$，以减少原始规划航线中的剧烈方向突变。

对于满足最大转角约束的候选状态，算法进一步根据相邻航段转角及其平均支撑长度计算离散曲率代价。该代价可视为局部曲率平方积分的离散近似，使算法在多个安全可行的状态转移中优先选择方向变化更平缓的方案。需要指出的是，该曲率项用于描述折线路径的局部几何变化，不涉及无人机的速度、加速度或姿态动力学。

设相邻航段向量分别为

$$
\mathbf v_{i-1}=\mathbf p_i-\mathbf p_{i-1},\qquad
\mathbf v_i=\mathbf p_{i+1}-\mathbf p_i,
\tag{23}
$$

则相邻航段转角及其约束定义为

$$
\theta_i=
\arccos\left\{
\operatorname{clip}\left[
\frac{\mathbf v_{i-1}^{\mathrm T}\mathbf v_i}
{\|\mathbf v_{i-1}\|_2\|\mathbf v_i\|_2},
-1,1
\right]
\right\},
\tag{24}
$$

$$
\theta_i\le\theta_{\max},
\qquad \theta_{\max}=60^\circ.
\tag{25}
$$

离散曲率代价及其曲率平方积分近似表示为

$$
K_i=
\frac{\theta_i^2}
{\frac{1}{2}(L_{i-1}+L_i)},
\tag{26}
$$

$$
\int\kappa^2(l)\,\mathrm dl
\approx
\frac{\theta_i^2}{L_{\mathrm{support}}},
\qquad
L_{\mathrm{support}}=\frac{L_{i-1}+L_i}{2}.
\tag{27}
$$

### 4) 综合代价与FMM启发函数

FDGA*的状态转移代价由航段长度、净空风险和离散曲率三部分构成。航段长度用于控制总航程，净空风险用于提高路径与障碍物之间的安全余量，曲率代价用于抑制相邻航段的方向突变。本文分别设置三项权重为1.0、0.2和0.1。

在启发搜索阶段，FDGA*直接采用FMM到达时间作为障碍感知启发信息。与仅依赖当前节点至目标点欧式距离的传统启发函数相比，FMM到达时间包含由ESDF速度场编码的全局障碍物分布和净空信息，可引导搜索朝综合通行代价较低的区域扩展。启发权重设置为1.0。

由于FMM到达时间对应的传播代价与FDGA*完整状态转移代价并不完全相同，本文不基于该启发函数的可采纳性或一致性声明理论上的全局最优性，而将其用于提高搜索的方向性与效率。

状态累计代价、权重设置、启发函数及优先队列评价函数分别定义为

$$
g(s_j)=g(s_i)
+w_L L_{ij}
+w_D R_{ij}
+w_K K_i,
\tag{28}
$$

$$
w_L=1.0,\qquad
w_D=0.2,\qquad
w_K=0.1,
\tag{29}
$$

$$
h(s_j)=T(c_j),
\tag{30}
$$

$$
f(s_j)=g(s_j)+w_H T(c_j),
\qquad w_H=1.0.
\tag{31}
$$

---

## F. FDGA*核心伪代码

上述过程按照实际执行逻辑划分为三个相互衔接的算法。算法1给出FDGA*总体流程，包括局部规划空间构建、FMM全局引导和方向感知A*搜索。算法2描述ESDF速度场、FMM到达时间场以及引导走廊的构建过程。算法3进一步给出走廊内的方向感知A*搜索、航段安全检查和代价更新过程。该拆分方式能够避免单段伪代码过长，并清晰区分全局引导与局部约束搜索两个核心阶段。

**Algorithm 1  Overall Procedure of FDGA* for 3-D Path Planning**

---

**Input:** $\mathbf p_s$, $\mathbf p_g$, $D$, $d_{\mathrm{safe}}$, $r_c$, $\theta_{\max}$, $w_L$, $w_D$, $w_K$, $w_H$

**Output:** $\mathcal P$

---

1　$\Omega_L\leftarrow\operatorname{LocalDomain}(\mathbf p_s,\mathbf p_g,D)$

2　$\Delta\leftarrow\operatorname{clip}(\|\mathbf p_s-\mathbf p_g\|_2/250,5,15)$

3　$(c_s,c_g)\leftarrow\operatorname{WorldToGrid}(\mathbf p_s,\mathbf p_g,\Omega_L,\Delta)$

4　$(T,\mathcal C)\leftarrow\operatorname{BuildGuidanceCorridor}(D,\Omega_L,c_s,c_g,\Delta,d_{\mathrm{safe}},r_c)$

5　**if** $(T,\mathcal C)=\varnothing$ **then return** $\varnothing$

6　$\mathcal P\leftarrow\operatorname{DirectionGuidedAStar}(D,T,\mathcal C,c_s,c_g,d_{\mathrm{safe}},\theta_{\max},w_L,w_D,w_K,w_H)$

7　**if** $\mathcal P=\varnothing$ **then return** $\varnothing$

8　$(\mathcal P_1,\mathcal P_{|\mathcal P|})\leftarrow(\mathbf p_s,\mathbf p_g)$

9　**return** $\mathcal P$

---

**Algorithm 2  ESDF-Based FMM Guidance Corridor Construction**

---

**Input:** $D$, $\Omega_L$, $c_s$, $c_g$, $\Delta$, $d_{\mathrm{safe}}$, $r_c$

**Output:** $T$, $\mathcal C$

---

1　**for** $c\in\Omega_L$ **do**

2　　　$d\leftarrow D(\mathbf p_c)$

3　　　**if** $d<0\lor\neg\operatorname{finite}(d)$ **then**

4　　　　　$F(c)\leftarrow10^{-6}$

5　　　**else**

6　　　　　$F(c)\leftarrow\max\{\min[d/d_{\mathrm{safe}},1],10^{-3}\}$

7　　　**end if**

8　**end for**

9　$T\leftarrow\operatorname{FMM}(F,c_g,\Delta)$

10　**if** $\neg\operatorname{finite}(T(c_s))$ **then return** $\varnothing$

11　$c\leftarrow c_s$, $\mathcal P_F\leftarrow\{c_s\}$

12　**while** $c\neq c_g$ **do**

13　　　$c_{\mathrm{next}}\leftarrow\arg\min_{c'\in\mathcal N_{26}(c)}T(c')$

14　　　**if** $T(c_{\mathrm{next}})\geq T(c)-10^{-6}$ **then return** $\varnothing$

15　　　$\mathcal P_F\leftarrow\mathcal P_F\cup\{c_{\mathrm{next}}\}$

16　　　$c\leftarrow c_{\mathrm{next}}$

17　**end while**

18　$\mathcal C\leftarrow\mathcal P_F\oplus\mathcal B(r_c)$

19　**return** $(T,\mathcal C)$

---

**Algorithm 3  Direction-Guided A* Search Within the FMM Corridor**

---

**Input:** $D$, $T$, $\mathcal C$, $c_s$, $c_g$, $d_{\mathrm{safe}}$, $\theta_{\max}$, $w_L$, $w_D$, $w_K$, $w_H$

**Output:** $\mathcal P$

---

1　$s_{\mathrm{start}}\leftarrow(c_s,q_{\varnothing})$, $g(s_{\mathrm{start}})\leftarrow0$

2　$\operatorname{parent}(s_{\mathrm{start}})\leftarrow\mathrm{NULL}$

3　$\mathrm{OPEN}\leftarrow\{(s_{\mathrm{start}},w_HT(c_s))\}$

4　**while** $\mathrm{OPEN}\neq\varnothing$ **do**

5　　　$s\leftarrow\operatorname{ExtractMin}(\mathrm{OPEN})$

6　　　**if** $\operatorname{cell}(s)=c_g$ **then**

7　　　　　$s_{\mathrm{goal}}\leftarrow s$; **break**

8　　　**end if**

9　　　**for** $q_{\mathrm{next}}\in\mathcal N_{26}$ **do**

10　　　　　$c_{\mathrm{next}}\leftarrow\operatorname{cell}(s)+q_{\mathrm{next}}$

11　　　　　**if** $c_{\mathrm{next}}\notin\mathcal C$ **then continue**

12　　　　　$e_{ij}\leftarrow(\mathbf p_i,\mathbf p_j)$

13　　　　　$\mathcal S_{ij}\leftarrow\operatorname{Sample}(e_{ij},\delta_e)$

14　　　　　**if** $\exists\mathbf p\in\mathcal S_{ij}:D(\mathbf p)<d_{\mathrm{safe}}\lor\neg\operatorname{finite}(D(\mathbf p))$ **then continue**

15　　　　　$R_{ij}\leftarrow\dfrac{L_{ij}}{N_{ij}+1}\displaystyle\sum_{k=0}^{N_{ij}}\dfrac{1}{\max[D(\mathbf p_{ij}^{k}),\varepsilon_d]}$

16　　　　　**if** $q(s)\neq q_{\varnothing}$ **then**

17　　　　　　　$\theta_i\leftarrow\arccos\!\left(\operatorname{clip}\!\left[\dfrac{\mathbf v_{i-1}^{\mathrm T}\mathbf v_i}{\|\mathbf v_{i-1}\|_2\|\mathbf v_i\|_2},-1,1\right]\right)$

18　　　　　　　**if** $\theta_i>\theta_{\max}$ **then continue**

19　　　　　　　$K_i\leftarrow\theta_i^2/[0.5(L_{i-1}+L_i)]$

20　　　　　**else**

21　　　　　　　$K_i\leftarrow0$

22　　　　　**end if**

23　　　　　$s_{\mathrm{next}}\leftarrow(c_{\mathrm{next}},q_{\mathrm{next}})$

24　　　　　$g_{\mathrm{new}}\leftarrow g(s)+w_LL_{ij}+w_DR_{ij}+w_KK_i$

25　　　　　**if** $g_{\mathrm{new}}<g(s_{\mathrm{next}})$ **then**

26　　　　　　　$g(s_{\mathrm{next}})\leftarrow g_{\mathrm{new}}$

27　　　　　　　$\operatorname{parent}(s_{\mathrm{next}})\leftarrow s$

28　　　　　　　$f(s_{\mathrm{next}})\leftarrow g_{\mathrm{new}}+w_HT(c_{\mathrm{next}})$

29　　　　　　　$\operatorname{Insert}(\mathrm{OPEN},s_{\mathrm{next}},f(s_{\mathrm{next}}))$

30　　　　　**end if**

31　　　**end for**

32　**end while**

33　**if** $s_{\mathrm{goal}}=\varnothing$ **then return** $\varnothing$

34　$\mathcal P\leftarrow\operatorname{Reverse}(\operatorname{Backtrack}(s_{\mathrm{goal}},s_{\mathrm{start}}))$

35　**return** $\mathcal P$

---

## G. ESDF与FMM的协同机制

FDGA*中ESDF与FMM的协同作用体现在全局引导与局部约束两个层面。首先，ESDF被映射为FMM传播速度场，使全局到达时间在低净空区域快速增大，进而引导参考路径和搜索走廊远离障碍物。其次，在方向感知A*搜索阶段，ESDF被用于候选航段的连续采样检查与净空风险计算，从而同时提供安全距离硬约束和障碍风险软惩罚。FMM到达时间场则同时用于提取参考路径、生成受限搜索走廊和构造A*启发函数。

通过上述机制，FDGA*以ESDF统一表征障碍物净空和飞行风险，以FMM到达时间场提供全局方向引导，并通过参考路径膨胀控制三维搜索规模。在走廊内部，算法利用“位置—入射方向”联合状态描述航段方向变化，在搜索阶段直接施加最小安全距离和最大转角约束，并通过净空风险与离散曲率代价平衡航程、安全性和几何连续性。因此，FDGA*并非FMM与A*的简单串联，而是以ESDF为纽带，将FMM全局传播与方向感知A*局部约束搜索耦合为统一的三维航线规划框架。

其协同关系概括如下：

$$
\mathrm{ESDF}
\longrightarrow
\begin{cases}
\text{FMM速度场与全局到达时间},\\
\text{航段净空约束与风险代价}
\end{cases}
\longrightarrow
\text{方向感知A*约束搜索}.
\tag{32}
$$
