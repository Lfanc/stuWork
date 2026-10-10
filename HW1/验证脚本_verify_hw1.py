# -*- coding: utf-8 -*-
"""HW1 三份作业文档的数值验证脚本（奇异值分解 / 贝叶斯过程 / 交叉熵）

运行方式（需要 numpy）：
    python 验证脚本_verify_hw1.py

脚本覆盖三份文档中出现的全部数值，可用于复核手算结果。
"""

import numpy as np

# 固定随机种子，保证蒙特卡洛结果可复现
rng = np.random.default_rng(20240501)


def 标题(文字):
    """打印分节标题，便于阅读输出"""
    print("\n" + "=" * 60)
    print(文字)
    print("=" * 60)


# ============================================================
# 第一部分：奇异值分解的两个例子
# ============================================================
标题("例1：A = [[3,2],[2,3]] 的奇异值分解")

A = np.array([[3.0, 2.0], [2.0, 3.0]])
print("A^T A =", (A.T @ A).tolist())          # [[13, 12], [12, 13]]

# 对 A^T A 做特征分解，求特征值与特征向量
特征值, V = np.linalg.eigh(A.T @ A)
顺序 = np.argsort(-特征值)                     # 特征值从大到小排序
特征值, V = 特征值[顺序], V[:, 顺序]
print("A^T A 的特征值 =", 特征值)              # [25. 1.]
print("奇异值 =", np.sqrt(特征值))             # [5. 1.]
print("右奇异向量 V =\n", V)

# 用 numpy 的 SVD 交叉验证
U, s, Vt = np.linalg.svd(A)
print("numpy 给出的奇异值 =", s)               # [5. 1.]
print("重构误差 =", np.linalg.norm(A - U @ np.diag(s) @ Vt))   # 约 1e-16
# 秩1逼近的误差应当等于 sigma2
print("秩1逼近误差 =", np.linalg.norm(A - s[0] * np.outer(U[:, 0], Vt[0]), 2))
print("条件数 =", s[0] / s[1])                 # 5.0
print("Frobenius 范数 =", np.linalg.norm(A, "fro"), " sqrt(25+1) =", np.sqrt(26))
print("行列式 =", np.linalg.det(A), " sigma1*sigma2 =", s.prod())

标题("例2：C = [[3,2],[2,3],[1,1]]（3×2 长方阵）")

C = np.array([[3.0, 2.0], [2.0, 3.0], [1.0, 1.0]])
print("C^T C =", (C.T @ C).tolist())          # [[14, 13], [13, 14]]

特征值, V2 = np.linalg.eigh(C.T @ C)
顺序 = np.argsort(-特征值)
特征值, V2 = 特征值[顺序], V2[:, 顺序]
# 统一符号：让每列绝对值最大的分量为正，保证输出稳定
for j in range(V2.shape[1]):
    if V2[np.argmax(np.abs(V2[:, j])), j] < 0:
        V2[:, j] = -V2[:, j]

奇异值 = np.sqrt(特征值)
print("特征值 =", 特征值, " 奇异值 =", 奇异值)  # [27. 1.] -> [5.19615242 1.]
print("右奇异向量 V =\n", V2)

# 左奇异向量由 u_i = C v_i / sigma_i 得到
U2 = C @ V2 / 奇异值
print("左奇异向量（前两个）=\n", U2)
print("u1 =", C @ V2[:, 0] / 奇异值[0])        # [0.680414 0.680414 0.272166]
print("u2 =", C @ V2[:, 1] / 奇异值[1])        # [0.707107 -0.707107 0.]
print("两个左奇异向量是否标准正交：", np.allclose(U2.T @ U2, np.eye(2)))

# 补全第三个左奇异向量：用前两个的叉积
u3 = np.cross(U2[:, 0], U2[:, 1])
print("u3 =", u3, " 模长 =", np.linalg.norm(u3))

# 经济型 SVD 的重构验证
Uc, sc, Vtc = np.linalg.svd(C, full_matrices=False)
print("重构误差 =", np.linalg.norm(C - Uc @ np.diag(sc) @ Vtc))   # 约 1e-15
print("秩1逼近误差 =", np.linalg.norm(C - sc[0] * np.outer(Uc[:, 0], Vtc[0]), 2),
      " sigma2 =", sc[1])                     # 都等于 1


# ============================================================
# 第二部分：贝叶斯逻辑回归（缺陷风险预测）
# ============================================================
标题("贝叶斯逻辑回归：Laplace 近似与后验预测")

# 训练数据：第 1 列为偏置项，其余三列为特征
X = np.array([[1, .35, .85, .30], [1, .90, .20, .75],
              [1, .55, .60, .45], [1, .75, .35, .65],
              [1, .20, .90, .25], [1, 1.10, .10, .85],
              [1, .45, .70, .40], [1, .65, .45, .55]], dtype=float)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=float)

d, alpha, w = X.shape[1], 1.0, np.zeros(4)     # d 为特征维数，alpha 为先验精度

# 牛顿-拉弗森迭代求后验众数（即 MAP 估计）
print("牛顿迭代过程：")
for it in range(12):
    mu = 1 / (1 + np.exp(-X @ w))              # 逻辑斯蒂预测概率
    W = mu * (1 - mu)                          # 加权矩阵的对角元
    g = X.T @ (y - mu) - alpha * w             # 后验对数梯度
    H = -(X.T @ (X * W[:, None])) - alpha * np.eye(d)    # 后验对数 Hessian
    step = np.linalg.solve(H, -g)              # 牛顿步长
    w = w + step
    print(f"  第 {it} 次: 梯度范数={np.linalg.norm(g):.3e}  最大步长={np.max(np.abs(step)):.3e}")
    if np.max(np.abs(step)) < 1e-12:           # 收敛判据
        break

print("MAP 估计 w =", np.round(w, 4))           # [-0.1272 0.5963 -0.7910 0.4513]

# Laplace 近似的后验协方差矩阵
S_N = np.linalg.inv(X.T @ (X * W[:, None]) + alpha * np.eye(d))
print("后验协方差 S_N =\n", np.round(S_N, 4))
print("后验标准差 =", np.round(np.sqrt(np.diag(S_N)), 4))   # [0.7667 0.8756 0.8737 0.9213]

# 与 MLE（不做正则化）对比：数据完全可分时 MLE 会发散
for 正则化强度 in [0.0, 1.0]:
    w_mle = np.zeros(d)
    for it in range(300):
        mu_m = 1 / (1 + np.exp(-X @ w_mle))
        g_m = X.T @ (y - mu_m) - 正则化强度 * w_mle
        H_m = -(X.T @ (X * (mu_m * (1 - mu_m))[:, None])) - 正则化强度 * np.eye(d)
        try:
            st = np.linalg.solve(H_m, -g_m)
        except np.linalg.LinAlgError:          # Hessian 奇异时退化为最小二乘解
            st = np.linalg.lstsq(H_m, -g_m, rcond=None)[0]
        w_mle = w_mle + st
        if np.max(np.abs(st)) < 1e-12:
            break
    print(f"MLE（正则化强度={正则化强度}）迭代 {it + 1} 次: ||w|| = {np.linalg.norm(w_mle):.3f}")

# 对新变更做后验预测
x_star = np.array([1, .80, .25, .70])
z = x_star @ w                                 # 线性得分
var_z = x_star @ S_N @ x_star                  # 得分上的参数不确定性
kappa = 1 / np.sqrt(1 + np.pi * var_z / 8)     # probit 近似的收缩因子
print(f"线性得分 z* = {z:.4f}  方差 = {var_z:.4f}  收缩因子 kappa = {kappa:.4f}")
print("点估计预测 =", round(float(1 / (1 + np.exp(-z))), 4))          # 0.6149
print("贝叶斯预测（probit）= ", round(float(1 / (1 + np.exp(-kappa * z))), 4))  # 0.6048

# 蒙特卡洛验证：对后验协方差做 Cholesky 分解后抽样
L = np.linalg.cholesky(S_N)
ws = w + rng.standard_normal((400000, d)) @ L.T
ps = 1 / (1 + np.exp(-(ws @ x_star)))
print("贝叶斯预测（蒙特卡洛）= ", round(float(ps.mean()), 4))         # 0.6032
print("95% 预测区间 =", np.round(np.percentile(ps, [2.5, 97.5]), 4))  # [0.2754 0.8702]

# 系数 w1 的不确定性
print("w1 =", round(float(w[1]), 4),
      " 95% 可信区间 =", np.round(w[1] + 1.96 * np.sqrt(S_N[1, 1]) * np.array([-1, 1]), 4))
print("P(w1>0|D) =", round(float((ws[:, 1] > 0).mean()), 4))          # 约 0.75


# ============================================================
# 第三部分：交叉熵
# ============================================================
标题("二分类交叉熵例子")

x, w2, y2 = np.array([0.5, 2.0]), np.array([-0.5, 0.5]), 1.0
z2 = w2 @ x
p2 = 1 / (1 + np.exp(-z2))                     # sigmoid
L2 = -(y2 * np.log(p2) + (1 - y2) * np.log(1 - p2))
print(f"z = {z2:.6f}   p = {p2:.6f}   交叉熵 = {L2:.6f} nat = {L2 / np.log(2):.6f} bit")
print("dL/dz =", round(float(p2 - y2), 6))     # -0.320821
print("dL/dw =", np.round((p2 - y2) * x, 6))   # [-0.160411 -0.641643]

# 用中心差分检验梯度
eps = 1e-7
损失函数 = lambda ww: -(y2 * np.log(1 / (1 + np.exp(-ww @ x)))
                        + (1 - y2) * np.log(1 - 1 / (1 + np.exp(-ww @ x))))
差分 = [(损失函数(w2 + eps * np.eye(2)[i]) - 损失函数(w2 - eps * np.eye(2)[i])) / (2 * eps)
        for i in range(2)]
print("差分梯度 =", np.round(差分, 6))          # 与解析梯度一致

标题("三分类 softmax 交叉熵例子")

W = np.array([[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]])   # 2 个特征 × 3 个类别
xs, 类别 = np.array([1.0, 0.5]), 2                    # 输入与真实类别（0 起算）
logits = W.T @ xs
# 减去最大值后再求指数，避免数值溢出
p3 = np.exp(logits - logits.max())
p3 = p3 / p3.sum()
print("logits =", np.round(logits, 6), " softmax =", np.round(p3, 6),
      " 概率之和 =", round(float(p3.sum()), 6))
print(f"交叉熵 = {-np.log(p3[类别]):.6f}   均匀预测基准 log3 = {np.log(3):.6f}")

梯度 = p3.copy()
梯度[类别] -= 1                                # 对 logits 的梯度 = 预测 - 真实
print("dL/dlogits =", np.round(梯度, 6), " 分量和 =", round(float(梯度.sum()), 12))
print("dL/dW =\n", np.round(np.outer(梯度, xs).T, 6))     # 外积形式

# 逐元素中心差分检验对权重的梯度
def 交叉熵(WW):
    lg = WW.T @ xs
    e = np.exp(lg - lg.max())
    return -np.log((e / e.sum())[类别])

差分W = np.zeros_like(W)
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        Wp, Wm = W.copy(), W.copy()
        Wp[i, j] += 1e-6
        Wm[i, j] -= 1e-6
        差分W[i, j] = (交叉熵(Wp) - 交叉熵(Wm)) / 2e-6
print("差分 dL/dW =\n", np.round(差分W, 6))

标题("验证 H(p,q) = H(p) + KL(p||q)")

pt = 0.8                                       # 真实分布 p(1) = 0.8
Hp = -(pt * np.log(pt) + (1 - pt) * np.log(1 - pt))
print(f"真实分布的熵 H(p) = {Hp:.6f}")
for q in [0.5, 0.6, 0.7, 0.8, 0.9]:
    CE = -(pt * np.log(q) + (1 - pt) * np.log(1 - q))                 # 交叉熵
    KL = pt * np.log(pt / q) + (1 - pt) * np.log((1 - pt) / (1 - q))  # KL 散度
    print(f"  q={q}: 交叉熵={CE:.6f}  KL={KL:.6f}  H(p)+KL={Hp + KL:.6f}")

print("\n全部验证完成")
