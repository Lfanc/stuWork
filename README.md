# stuWork

课程作业仓库。每次作业一个文件夹，文档用 Markdown + LaTeX 写成。

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
├── HW3/                              第三次作业：小模型训练报告
│   ├── report.md                    实验报告正文：数据、模型、训练、评估、局限
│   ├── img/                         报告插图：语料长度分布、参数量构成、训练曲线
│   ├── nanoGPT-bundle/              云端训练所需的全套脚本与数据
│   │   ├── config/
│   │   │   ├── train_zhihu.py       训练超参，6 层 384 维 / 1500 步（本次采用）
│   │   │   └── train_zhihu_91m.py   备选配置，12 层 768 维 / 91.5M 参数
│   │   ├── data/zhihu_char/         分词结果：train.bin / val.bin / meta.pkl
│   │   ├── prepare_zhihu.py         纯文本转字符级 token，产出 .bin 与词表
│   │   ├── apply_swanlab_patch.py   把 train.py 的日志后端从 WandB 换成 SwanLab
│   │   ├── eval_zhihu.py            评估 loss / PPL / bits-per-char 并生成样本
│   │   ├── make_figures.py          由原始语料与配置生成报告插图
│   │   └── README.md                云端操作手册：时间预算与坑位清单
│   └── zhihu_douban.txt             原始语料，COIG-CQIA 知乎 + 豆瓣，17.8MB / 8799 条
│
|
└── README.md
```

