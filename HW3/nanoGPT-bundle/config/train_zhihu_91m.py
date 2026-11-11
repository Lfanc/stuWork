# 知乎/豆瓣中文语料 · 0.09B 规模（GPT-2 small 架构）· V100-32GB
# 用法（在 nanoGPT 根目录）：python train.py config/train_zhihu_91m.py
#
# 相对 6L384 版本：参数从 13.9M 提到 91.5M，符合"0.1B 级别"要求。
# 实测基准：6L384 在 V100 上 210ms/iter（31.2 万 tok/s）。
# 91M 模型吞吐会掉到约 5-8 万 tok/s，即 0.8-1.3s/iter，
# 2000 iter 合计约 30-45 分钟。

out_dir = 'out-zhihu-91m'
dataset = 'zhihu_char'

# ---- 模型：GPT-2 small 同构，12 层 / 12 头 / 768 维 ≈ 91.5M 参数 ----
n_layer = 12
n_head = 12
n_embd = 768
block_size = 512
bias = False
dropout = 0.1          # 634 万 token 配 91M 参数，过拟合压力大，保留 0.1

# ---- 批次：32 x 512 x 4 = 65,536 token / iter，约 97 iter 一个 epoch ----
# 若 OOM：batch_size 降到 16，同时 gradient_accumulation_steps 提到 8
batch_size = 32
gradient_accumulation_steps = 4

# ---- 优化器：GPT-2 small 量级不能再吃 1e-3 ----
learning_rate = 6e-4
max_iters = 2000       # 约 20.6 epoch / 1.31 亿 token
lr_decay_iters = 2000
min_lr = 6e-5
warmup_iters = 200
weight_decay = 1e-1
beta2 = 0.99

# ---- 日志与存档 ----
eval_interval = 200    # 2000 iter 共 10 个评估点
eval_iters = 30        # 大模型 eval 更贵，从 50 降到 30 省时间
log_interval = 20
eval_only = False
always_save_checkpoint = True   # 每点存档 ~1.1GB（含优化器状态），可续训
init_from = 'scratch'

# ---- 硬件 ----
device = 'cuda'
dtype = 'float16'      # V100 不支持 bf16
compile = False

# ---- 日志后端：复用 wandb_* 变量名，实现已替换为 SwanLab ----
wandb_log = True
wandb_project = 'nanoGPT-zhihu-douban'
wandb_run_name = 'char-12L768-bs32-6e-4'
