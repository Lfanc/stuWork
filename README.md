# stuWork

课程作业仓库。每次作业一个文件夹，文档用 Markdown + LaTeX 写成，另有少量 Python 脚本用于数值复核。

## 目录结构

```
stuWork/
├── HW1/                              第一次作业：线性代数与概率基础
│   ├── 1_奇异值分解SVD.md              SVD 的定义、完整推导与算例
│   ├── 2_贝叶斯过程及应用案例.md        贝叶斯推断流程与应用案例
│   ├── 3_交叉熵定义原理与应用.md        交叉熵的由来、与 KL 散度的关系
│   └── 验证脚本_verify_hw1.py          用 numpy 复核三份文档中的全部数值
│
├── HW2/                              第二次作业：ResNet 与深度网络的连接结构
│   ├── 1_理论推导_ResNet权重梯度.md     由 x_{l+1}=x_l+F(x_l,W_l) 推出 ∂L/∂W_l
│   └── 2_思考与对比_Highway与DenseNet.md  Highway vs ResNet、DenseNet vs ResNet 对比
│
└── README.md
```

## 各次作业

| 作业 | 主题 | 关键内容 |
| --- | --- | --- |
| HW1 | 线性代数 / 概率 | 奇异值分解的构造性证明、贝叶斯过程、交叉熵与 KL 散度 |
| HW2 | 深度学习 | 残差连接的梯度回传推导、门控短路与拼接短路的对比 |

## 使用说明

- Markdown 文档内含大量 `$...$` / `$$...$$` 数学公式，建议用 Typora、Obsidian 或 VS Code（Markdown Math 预览）打开。
- 运行验证脚本需 Python 3 与 numpy：

```bash
cd HW1
python 验证脚本_verify_hw1.py
```

- `.workbuddy/` 为本地工作数据（记忆、缓存），已写入 `.gitignore`，不纳入版本管理。
