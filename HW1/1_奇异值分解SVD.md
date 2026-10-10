# 线性代数作业：奇异值分解（SVD）

## 一、定义

设 $A\in\mathbb R^{m\times n}$，$\operatorname{rank}(A)=r$。则存在正交矩阵 $U\in\mathbb R^{m\times m}$、$V\in\mathbb R^{n\times n}$ 和对角矩阵 $\Sigma\in\mathbb R^{m\times n}$，使

$$A=U\Sigma V^{\mathsf T},\qquad \Sigma=\operatorname{diag}(\sigma_1,\dots,\sigma_r,0,\dots,0),\quad \sigma_1\ge\cdots\ge\sigma_r>0$$

其中

- $\sigma_i$ 称为 $A$ 的**奇异值**；
- $U=(\mathbf u_1,\dots,\mathbf u_m)$ 的列称为**左奇异向量**；
- $V=(\mathbf v_1,\dots,\mathbf v_n)$ 的列称为**右奇异向量**。

写成分量形式：

$$A=\sum_{i=1}^{r}\sigma_i\,\mathbf u_i\mathbf v_i^{\mathsf T}$$

## 二、推导过程

**第 1 步：构造对称方阵。**
$A^{\mathsf T}A$ 是 $n\times n$ 对称半正定矩阵（因为 $\mathbf x^{\mathsf T}A^{\mathsf T}A\mathbf x=\lVert A\mathbf x\rVert^2\ge0$）。由对称矩阵的谱定理，存在正交矩阵 $V$ 使

$$A^{\mathsf T}A=V\Lambda V^{\mathsf T},\qquad \Lambda=\operatorname{diag}(\lambda_1,\dots,\lambda_n),\quad \lambda_1\ge\cdots\ge\lambda_n\ge0$$

**第 2 步：定义奇异值。**

$$\sigma_i=\sqrt{\lambda_i},\qquad \sigma_1\ge\cdots\ge\sigma_r>0=\sigma_{r+1}=\cdots=\sigma_n$$

其中 $r=\operatorname{rank}(A)$（非零特征值的个数）。

**第 3 步：构造左奇异向量。** 对 $i=1,\dots,r$ 令

$$\mathbf u_i=\frac{A\mathbf v_i}{\sigma_i}=\frac{A\mathbf v_i}{\lVert A\mathbf v_i\rVert}$$

验证它是 $AA^{\mathsf T}$ 的单位特征向量：

$$AA^{\mathsf T}\mathbf u_i=\frac{1}{\sigma_i}AA^{\mathsf T}A\mathbf v_i=\frac{1}{\sigma_i}A(\lambda_i\mathbf v_i)=\frac{\lambda_i}{\sigma_i}A\mathbf v_i=\sigma_i^2\mathbf u_i\ \checkmark$$

验证两两正交（$i\ne j$ 时 $\mathbf v_i^{\mathsf T}\mathbf v_j=0$）：

$$\mathbf u_i^{\mathsf T}\mathbf u_j=\frac{(A\mathbf v_i)^{\mathsf T}(A\mathbf v_j)}{\sigma_i\sigma_j}=\frac{\mathbf v_i^{\mathsf T}A^{\mathsf T}A\mathbf v_j}{\sigma_i\sigma_j}=\frac{\lambda_j\mathbf v_i^{\mathsf T}\mathbf v_j}{\sigma_i\sigma_j}=0\ \checkmark$$

**第 4 步：补全基。** 若 $r<m$，把 $\{\mathbf u_1,\dots,\mathbf u_r\}$ 用 Gram–Schmidt 补成 $\mathbb R^m$ 的标准正交基；$\{\mathbf v_i\}$ 同理补全。

**第 5 步：验证分解。** 只要验证两边对基 $\{\mathbf v_j\}$ 作用相同：

- $j\le r$ 时 $(U\Sigma V^{\mathsf T})\mathbf v_j=U\Sigma\mathbf e_j=\sigma_j\mathbf u_j=A\mathbf v_j$；
- $j>r$ 时 $\sigma_j=0$，左边为 $\mathbf 0$；而 $\lVert A\mathbf v_j\rVert^2=\mathbf v_j^{\mathsf T}A^{\mathsf T}A\mathbf v_j=0$，故 $A\mathbf v_j=\mathbf 0$，右边也是 $\mathbf 0$。

所以 $U\Sigma V^{\mathsf T}=A$。

**结论**：$A$ 的右奇异向量是 $A^{\mathsf T}A$ 的特征向量，左奇异向量是 $AA^{\mathsf T}$ 的特征向量，奇异值是二者非零特征值的平方根。

## 三、例子

### 例 1：$A=\begin{bmatrix}3&2\\2&3\end{bmatrix}$

**求 $A^{\mathsf T}A$：**

$$A^{\mathsf T}A=\begin{bmatrix}3&2\\2&3\end{bmatrix}\begin{bmatrix}3&2\\2&3\end{bmatrix}=\begin{bmatrix}13&12\\12&13\end{bmatrix}$$

**求特征值：**

$$\det(A^{\mathsf T}A-\lambda I)=(13-\lambda)^2-144=0\ \Rightarrow\ 13-\lambda=\pm12\ \Rightarrow\ \lambda_1=25,\ \lambda_2=1$$

**奇异值：** $\sigma_1=5,\ \sigma_2=1$，故 $\Sigma=\begin{bmatrix}5&0\\0&1\end{bmatrix}$，$\operatorname{rank}(A)=2$。

**右奇异向量：**

$\lambda_1=25$：$\begin{bmatrix}-12&12\\12&-12\end{bmatrix}\mathbf v=\mathbf 0\Rightarrow v_1=v_2$，取 $\mathbf v_1=\frac{1}{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}$。

$\lambda_2=1$：$\begin{bmatrix}12&12\\12&12\end{bmatrix}\mathbf v=\mathbf 0\Rightarrow v_1=-v_2$，取 $\mathbf v_2=\frac{1}{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix}$。

**左奇异向量：**

$$A\mathbf v_1=\frac{1}{\sqrt2}\begin{bmatrix}5\\5\end{bmatrix}\ \Rightarrow\ \mathbf u_1=\frac{A\mathbf v_1}{5}=\frac{1}{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}$$

$$A\mathbf v_2=\frac{1}{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix}\ \Rightarrow\ \mathbf u_2=\frac{A\mathbf v_2}{1}=\frac{1}{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix}$$

**分解结果：**

$$A=\begin{bmatrix}3&2\\2&3\end{bmatrix}=\frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}\begin{bmatrix}5&0\\0&1\end{bmatrix}\frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$$

**验证（外积展开）：**

$$5\mathbf u_1\mathbf v_1^{\mathsf T}+\mathbf u_2\mathbf v_2^{\mathsf T}=\begin{bmatrix}2.5&2.5\\2.5&2.5\end{bmatrix}+\begin{bmatrix}0.5&-0.5\\-0.5&0.5\end{bmatrix}=\begin{bmatrix}3&2\\2&3\end{bmatrix}=A\ \checkmark$$

**性质核对：** $\lVert A\rVert_2=\sigma_1=5$；$\lVert A\rVert_F=\sqrt{25+1}=\sqrt{26}\approx5.099$（与 $\sqrt{9+4+4+9}$ 一致）；$\lvert\det A\rvert=5=\sigma_1\sigma_2$；条件数 $\kappa_2=5/1=5$。

**秩 1 逼近**：$A_1=5\mathbf u_1\mathbf v_1^{\mathsf T}=\begin{bmatrix}2.5&2.5\\2.5&2.5\end{bmatrix}$，误差 $\lVert A-A_1\rVert_2=\sigma_2=1$。

（注：本例 $A$ 对称，所以 $U=V$，SVD 退化为特征值分解。）

### 例 2：$A=\begin{bmatrix}3&2\\2&3\\1&1\end{bmatrix}$（3×2 长方阵）

$$A^{\mathsf T}A=\begin{bmatrix}3&2&1\\2&3&1\end{bmatrix}\begin{bmatrix}3&2\\2&3\\1&1\end{bmatrix}=\begin{bmatrix}14&13\\13&14\end{bmatrix}$$

$$(14-\lambda)^2-169=0\ \Rightarrow\ \lambda_1=27,\ \lambda_2=1\ \Rightarrow\ \sigma_1=3\sqrt3\approx5.196,\ \sigma_2=1$$

右奇异向量：$\mathbf v_1=\frac{1}{\sqrt2}\begin{bmatrix}1\\1\end{bmatrix}$，$\mathbf v_2=\frac{1}{\sqrt2}\begin{bmatrix}1\\-1\end{bmatrix}$。

左奇异向量：

$$\mathbf u_1=\frac{A\mathbf v_1}{3\sqrt3}=\frac{1}{3\sqrt6}\begin{bmatrix}5\\5\\2\end{bmatrix},\qquad
\mathbf u_2=\frac{A\mathbf v_2}{1}=\frac{1}{\sqrt2}\begin{bmatrix}1\\-1\\0\end{bmatrix}$$

由于 $m=3>r=2$，需补第三个向量，取 $\mathbf u_3=\mathbf u_1\times\mathbf u_2=\frac{1}{3\sqrt3}\begin{bmatrix}1\\1\\-5\end{bmatrix}$，可验证 $\lVert\mathbf u_3\rVert=1$ 且与 $\mathbf u_1,\mathbf u_2$ 正交。

**完全型 SVD：**

$$A=\begin{bmatrix}\frac{5}{3\sqrt6}&\frac{1}{\sqrt2}&\frac{1}{3\sqrt3}\\[3pt]\frac{5}{3\sqrt6}&-\frac{1}{\sqrt2}&\frac{1}{3\sqrt3}\\[3pt]\frac{2}{3\sqrt6}&0&-\frac{5}{3\sqrt3}\end{bmatrix}\begin{bmatrix}\sqrt{27}&0\\0&1\\0&0\end{bmatrix}\frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}$$

**经济型 SVD**（丢掉 $\sigma=0$ 的行列）：$A=U_2\Sigma_2V^{\mathsf T}$，其中 $U_2$ 取 $U$ 的前两列，$\Sigma_2=\operatorname{diag}(\sqrt{27},1)$。

**验证：**

$$\sqrt{27}\,\mathbf u_1\mathbf v_1^{\mathsf T}+\mathbf u_2\mathbf v_2^{\mathsf T}=\begin{bmatrix}2.5&2.5\\2.5&2.5\\1&1\end{bmatrix}+\begin{bmatrix}0.5&-0.5\\-0.5&0.5\\0&0\end{bmatrix}=\begin{bmatrix}3&2\\2&3\\1&1\end{bmatrix}=A\ \checkmark$$

秩 1 逼近误差 $\lVert A-A_1\rVert_2=\sigma_2=1$。

## 四、几点说明

1. 任何矩阵（不论方阵、是否满秩、是否可对角化）都存在 SVD。
2. 零奇异值的个数 $=n-r$，等于零空间维数，因此 SVD 可用来数值稳定地判断矩阵的秩。
3. 截断 SVD $A_k=\sum_{i\le k}\sigma_i\mathbf u_i\mathbf v_i^{\mathsf T}$ 是所有秩不超过 $k$ 的矩阵中在谱范数与 Frobenius 范数下**同时最优**的逼近（Eckart–Young 定理），这是图像压缩、PCA 降维的理论依据。
4. 数值计算中不显式构造 $A^{\mathsf T}A$，因为条件数会被平方（$\kappa(A^{\mathsf T}A)=\kappa(A)^2$），实际采用双对角化 + QR 迭代。

## 附：验证代码

```python
import numpy as np

# 例1：2×2 方阵
A = np.array([[3., 2.], [2., 3.]])
U, s, Vt = np.linalg.svd(A)
print("奇异值:", s)                                          # [5. 1.]
print("重构误差:", np.linalg.norm(A - U @ np.diag(s) @ Vt))     # 约 1e-16
print("秩1逼近误差:", np.linalg.norm(A - s[0] * np.outer(U[:, 0], Vt[0]), 2))  # 1.0，即 sigma2
print("条件数:", s[0] / s[1])                                # 5.0

# 例2：3×2 长方阵（经济型分解）
C = np.array([[3., 2.], [2., 3.], [1., 1.]])
Uc, sc, Vtc = np.linalg.svd(C, full_matrices=False)
print("奇异值:", sc)                                         # [5.19615242 1.]
print("u1 =", C @ Vtc[0] / sc[0])                            # [0.680414 0.680414 0.272166]
print("u2 =", C @ Vtc[1] / sc[1])                            # [0.707107 -0.707107 0.]
```
