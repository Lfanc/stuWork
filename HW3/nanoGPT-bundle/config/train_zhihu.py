# 知乎/豆瓣中文语料 · V100-32GB 训练配置
# 用法（在 nanoGPT 根目录）：python train.py config/train_zhihu.py
# 续训：把 init_from 改成 'resume' 再执行同一条命令

out_dir = 'out-zhihu'
dataset = 'zhihu_char'

# ---- 模型：6 层 / 6 头 / 384 维，约 13.6M 参数 ----
n_layer = 6
n_head = 6
n_embd = 384
block_size = 512
bias = False
dropout = 0.1          # 650 万 token 配 13M 参数，加一点 dropout 抑制过拟合

# ---- 批次：32 x 512 x 4 = 65,536 token / iter，约 100 iter 一个 epoch ----
batch_size = 32
gradient_accumulation_steps = 4

# ---- 优化器：小模型可以吃更大 lr ----
learning_rate = 1e-3
max_iters = 1500       # 约 15 个 epoch
lr_decay_iters = 1500
min_lr = 1e-4
warmup_iters = 100
weight_decay = 1e-1
beta2 = 0.99

# ---- 日志与存档 ----
eval_interval = 150    # 1500 iter 共 10 个评估点，曲线够看
eval_iters = 50
log_interval = 10
eval_only = False
always_save_checkpoint = True   # 每个评估点都存，2 小时被切断也不丢进度
init_from = 'scratch'           # 续训时改成 'resume'

# ---- 硬件 ----
device = 'cuda'
dtype = 'float16'      # V100 不支持 bf16，必须用 fp16（train.py 会自动开 GradScaler）
compile = False        # 2 小时窗口下 torch.compile 的编译开销不值得

# ---- 日志后端：这里复用 nanoGPT 的 wandb_* 变量名，实现已被替换成 SwanLab ----
wandb_log = True
wandb_project = 'nanoGPT-zhihu-douban'
wandb_run_name = 'char-6L384-bs32-1e-3'
