# nanoGPT 中文训练 · 云端操作手册

目标：在智星云 V100-32G 上，用 nanoGPT 从零训练一个 6 层 384 维（约 13.6M 参数）的中文模型，
SwanLab 记录曲线，最后产出可写进报告的评估数据。

## 文件清单

| 文件 | 作用 |
|---|---|
| `prepare_zhihu.py` | `zhihu_douban.txt` → `train.bin` / `val.bin`（字符级，无需联网） |
| `config/train_zhihu.py` | 训练超参（V100 已调好） |
| `apply_swanlab_patch.py` | 把 train.py 的 WandB 后端换成 SwanLab（自动备份） |
| `eval_zhihu.py` | 算 loss / PPL / bits-per-char，生成样本，导出报告表格 |
| `data/zhihu_char/` | **已在本机生成好的** `train.bin` / `val.bin` / `meta.pkl`，直接上传即可 |

本地实测结果（可直接写进报告）：

| 指标 | 数值 |
|---|---|
| 原始文本 | 17.8 MB / 6,366,690 字符 / 8,799 条问答 |
| 训练集 token | 6,334,937（12.7 MB） |
| 验证集 token | 31,753 |
| 字符级词表 | 8,010 |
| 1 epoch | 约 97 iter（batch32 × block512 × accum4） |
| 1500 iter | 约 15.5 epoch / 9830 万 token |

## 时间预算（单次 2 小时窗口内）

| 阶段 | 预计 | 累计 |
|---|---|---|
| 环境自检 + 装 swanlab | 5–10 min | 10 |
| 上传代码与数据（13 MB） | 3–10 min | 20 |
| 冒烟测试（50 iter） | 2–4 min | 24 |
| **正式训练（1500 iter，约 15.5 epoch）** | **20–40 min** | 64 |
| 评估 + 采样 | 3–5 min | 69 |
| 截图 / 下载 checkpoint | 5 min | 74 |

留 40 分钟以上缓冲。**不需要为"跑不完"而分段，checkpoint 只是保险。**

---

## Step 0 · 本地准备（现在就能做）

```bash
# 1. 克隆官方 nanoGPT（国内网络建议用镜像，见文末坑位 1）
git clone https://github.com/karpathy/nanoGPT.git

# 2. 把本目录 4 个文件复制进 nanoGPT 根目录
#    prepare_zhihu.py / apply_swanlab_patch.py / eval_zhihu.py -> nanoGPT/
#    config/train_zhihu.py                                     -> nanoGPT/config/
#    zhihu_douban.txt 放到 nanoGPT 根目录（或上一级都行，路径自己对应）

# 3. 删掉 .git 便于打包（可选）
# 4. 打包上传
zip -r nanogpt_zhihu.zip nanoGPT
```

> 本地不用装 torch。你只做打包和最后写报告。

## Step 1 · 云端环境

开机后先自检，**不要直接开始训练**：

```bash
nvidia-smi                                  # 确认看到 V100-32GB
python -c "import torch;print(torch.__version__, torch.cuda.is_available())"
```

镜像 `ubuntu22.04_cuda12.4` 一般自带 torch。如果 `cuda.is_available()` 是 False 或 torch 版本过旧：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

装日志与依赖：

```bash
pip install swanlab numpy
swanlab login          # 输入 API Key；或 export SWANLAB_API_KEY=xxx
```

> 若平台提供「无卡/CPU 模式开机」，用它来装环境、传数据、跑 tokenize，
> 只在真正训练时才开 V100，能省出不少 GPU 时长。

## Step 2 · 数据（已完成，可选验证）

本目录 `data/zhihu_char/` 就是跑完的分词结果，**上传后直接用，云端不用再跑一遍**。
如果想在云端复现一遍（几秒钟，也能截图）：

```bash
cd nanoGPT
python prepare_zhihu.py --input zhihu_douban.txt --out_dir data/zhihu_char
```

期望输出：vocab_size = 8010，`train.bin` 6,334,937 token（12.7 MB），"约 97 iter 一个 epoch"。

如果想压缩词表（8010 个字符里有不少只出现一次的生僻字，白占 softmax 计算）：

```bash
python prepare_zhihu.py --input zhihu_douban.txt --out_dir data/zhihu_char --min_freq 3
```

## Step 3 · 换日志后端

```bash
python apply_swanlab_patch.py
```

成功后 `train.py` 里变成 `import swanlab as wandb`，config 里的 `wandb_*` 变量名不用动。

## Step 4 · 训练

先冒烟 50 步，确认 loss 在下降、SwanLab 有曲线：

```bash
python train.py config/train_zhihu.py --max_iters=50 --eval_interval=25 \
    --always_save_checkpoint=False --out_dir=out-smoke
```

看到 `step 50: train loss 5.x` 且 SwanLab 页面有数据，再跑正式的：

```bash
python train.py config/train_zhihu.py
```

预期：从 loss ≈ 8.99（ln 8010）开始，1500 步降到 4–5.5 区间，耗时 20–40 分钟。
验证 loss 若在中后段开始回升，说明小数据上过拟合了——**这是报告里很好写的一段分析**，别慌。

被 2 小时切断或想提前关机，下次开机后：

```bash
python train.py config/train_zhihu.py --init_from=resume
```

## Step 5 · 评估

```bash
python eval_zhihu.py --out_dir out-zhihu --data_dir data/zhihu_char
```

产出：
- `eval_report.md` — 已排好版的指标表格，直接粘进报告
- `samples.txt` — 12 条不同温度/prompt 的生成样本

**关机前务必把 `out-zhihu/ckpt.pt` 和 `samples.txt` 下载到本地**，报告和后续采样都要用。

## Step 6 · 报告

建议章节与配图：

1. **任务与方案** — 一段话说清"复现 nanoGPT + 中文语料 + 小模型"
2. **数据集构建** — 来源（m-a-p/COIG-CQIA 的 zhihu/douban 子集）、清洗规则、
   统计表（17.8 MB / 636 万字符 / 8799 条 QA / 8010 字词表 / 634 万训练 token）、
   2–3 条原文样例、tokenize 终端截图
3. **模型结构** — 参数表（n_layer/n_head/n_embd/block_size/参数量），有精力可以画个结构图
4. **训练设置** — 超参表（含 lr / batch / 累积步数 / dtype / 硬件 / 实际机时）
5. **训练过程** — SwanLab 的 train loss + val loss 双曲线截图、学习率调度曲线截图
6. **评估结果** — `eval_report.md` 的表格 + 3–4 条生成样本 + 人工点评（语法对不对、语义有没有、是否重复）
7. **问题与局限** — 过拟合、字符级无法跨词泛化、只是预训练不是指令微调、
   8500 条数据规模偏小、生成内容存在重复退化
8. **复现步骤** — 把 Step 1–5 的命令贴进去
9. **参考** — nanoGPT / COIG-CQIA / SwanLab

第 7 章是拉开分差的地方，写具体现象（比如"温度 1.0 时第 40 字后开始复读同一短语"）比写套话有用。

---

## 坑位清单

1. **GitHub clone 慢/失败** — 用 `https://ghproxy.net/https://github.com/karpathy/nanoGPT.git`
   或在 Gitee 上 import 后 clone。本地先 clone 好再打包上传最稳。
2. **tiktoken 下载失败** — 这是走 BPE（`--tokenizer gpt2`）才会遇到的问题，
   字符级方案完全绕开。这也是选字符级的原因之一。
3. **SwanLab 连不上** — `export SWANLAB_MODE=local` 先跑，之后 `swanlab watch ./swanlog`
   本地起面板截图。曲线图照样能拿到。
4. **OOM** — 把 `config/train_zhihu.py` 里 `batch_size` 从 32 降到 16，
   同时 `gradient_accumulation_steps` 从 4 提到 8（有效批量不变）。
5. **2 小时被硬切** — `always_save_checkpoint = True` 已打开，每个评估点都在存 `ckpt.pt`，
   续训用 `--init_from=resume`。确认 `out-zhihu/` 在关机后仍保留的目录里
   （多数平台"关机"保留磁盘、"释放/销毁"才清空，开机前看清说明）。
6. **`out of memory` 但显存看着够** — V100 是 32G 版，通常不会；若真遇到先降 `block_size` 到 256。
7. **想要更好的中文效果** — 这是作业，不要求效果。真要提升优先动数据（更多语料）
   而不是加大模型，13M 参数在 650 万 token 上已经接近容量上限。
