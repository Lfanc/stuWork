# 作业 1：ResNet 中权重层 $W_l$ 的梯度推导

## 一、记号与基本关系

残差块（省略 BN 与最后的非线性，不影响结构）：

$$
x_{l+1}=x_l+F(x_l,W_l),\qquad l=0,1,\dots,L-1
$$

$\mathcal{L}$ 表示损失，$L$ 表示总层数。对 $x_{l+1}$ 的两条依赖关系分别求偏导：

$$
\frac{\partial x_{l+1}}{\partial x_l}=I+\frac{\partial F(x_l,W_l)}{\partial x_l}\;:=\;I+J_l
$$

$$
\frac{\partial x_{l+1}}{\partial W_l}=\frac{\partial F(x_l,W_l)}{\partial W_l}
$$

其中 $J_l$ 是残差分支对输入的雅可比矩阵。因为 $W_l$ 只出现在 $F(x_l,\cdot)$ 中，所以第二个式子是"纯局部"的。

## 二、深层梯度：梯度的"高速公路"

$$
\frac{\partial \mathcal{L}}{\partial x_l}=\left(I+J_l\right)^{\mathrm{T}}\frac{\partial \mathcal{L}}{\partial x_{l+1}}
$$

逐层展开到 $x_L$：

$$
\frac{\partial \mathcal{L}}{\partial x_l}=\left[\prod_{i=l}^{L-1}\left(I+J_i\right)^{\mathrm{T}}\right]\frac{\partial \mathcal{L}}{\partial x_L}
$$

把连乘展开（共 $2^{L-l}$ 项）：

$$
\frac{\partial \mathcal{L}}{\partial x_l}=\frac{\partial \mathcal{L}}{\partial x_L}+\sum_{i=l}^{L-1}J_i^{\mathrm{T}}\frac{\partial \mathcal{L}}{\partial x_L}+\sum_{l\le i<j\le L-1}\left(J_iJ_j\right)^{\mathrm{T}}\frac{\partial \mathcal{L}}{\partial x_L}+\cdots+\left(J_{L-1}\cdots J_l\right)^{\mathrm{T}}\frac{\partial \mathcal{L}}{\partial x_L}
$$

**关键点**：第一项 $\partial \mathcal{L}/\partial x_L$ 不含任何 $J_i$，系数恒为 $1$。即使所有中间层的雅可比都极小（$\|J_i\|\ll 1$），深层梯度仍能原样传回第 $l$ 层——这就是"$+1$ 保证梯度无衰减地回传"。

对比普通网络（纯连乘，$\|J_i\|<1$ 时指数衰减）：

$$
\frac{\partial \mathcal{L}}{\partial x_l}=\left(\prod_{i=l}^{L-1}J_i\right)^{\mathrm{T}}\frac{\partial \mathcal{L}}{\partial x_L}
$$

## 三、权重梯度 $\partial \mathcal{L}/\partial W_l$

由链式法则（$x_l$ 的取值与 $W_l$ 无关，故只有 $F$ 这一条路径）：

$$
\frac{\partial \mathcal{L}}{\partial W_l}=\left(\frac{\partial F(x_l,W_l)}{\partial W_l}\right)^{\mathrm{T}}\frac{\partial \mathcal{L}}{\partial x_{l+1}}
$$

代入第二节结果，得到完整表达式：

$$
\frac{\partial \mathcal{L}}{\partial W_l}=\left(\frac{\partial F_l}{\partial W_l}\right)^{\mathrm{T}}\left[\prod_{i=l+1}^{L-1}\left(I+J_i\right)^{\mathrm{T}}\right]\frac{\partial \mathcal{L}}{\partial x_L}
$$

把它展开，可清楚看出"直达项 + 修正项"两部分：

$$
\frac{\partial \mathcal{L}}{\partial W_l}=\left(\frac{\partial F_l}{\partial W_l}\right)^{\mathrm{T}}\frac{\partial \mathcal{L}}{\partial x_L}+\left(\frac{\partial F_l}{\partial W_l}\right)^{\mathrm{T}}\sum_{i=l+1}^{L-1}J_i^{\mathrm{T}}\frac{\partial \mathcal{L}}{\partial x_L}+\cdots
$$

其中第一项 $\left(\partial F_l/\partial W_l\right)^{\mathrm{T}}\partial \mathcal{L}/\partial x_L$ 就是**直达项**：它不经过 $l$ 与 $L$ 之间的任何一层。

## 四、两者的关联与结论

1. **形式上的关联**：$\partial \mathcal{L}/\partial W_l$ 等于局部雅可比 $\partial F_l/\partial W_l$ 左乘深层梯度 $\partial \mathcal{L}/\partial x_{l+1}$。这一步与普通网络完全相同，**区别全部藏在 $\partial \mathcal{L}/\partial x_{l+1}$ 里**——它比普通网络多了一个系数恒为 $1$ 的恒等项。
2. **不存在梯度消失的直连通道**：直达项的系数恒为 $1$，与中间隔了多少层、那些层的权重多小、有没有 BN/ReLU 都无关。因此最浅层的权重也能收到未衰减的梯度信号。
3. **恒等映射成为"默认解"**：主干的系数是 $1$，残差分支只需学一个小的修正量 $F$。把 $F$ 学成 $0$ 比让一堆非线性层直接拟合恒等映射容易得多，这正是"网络加深后训练误差反而更低"的原因。
4. **另一面**：求和项随层数增加而变多，梯度范数可能变大（相当于 $2^{L-l}$ 条不同深度路径的叠加），所以 ResNet 更容易出现"梯度过强"而非过弱的问题，实践中常配合 BN 与权重衰减来稳定。
