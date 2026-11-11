"""
训练完成后的评估：验证集 loss / 困惑度 / bits-per-char / 生成样本。

一次跑完，直接产出可以直接贴进报告的 eval_report.md + samples.txt。

在 nanoGPT 根目录运行：
    python eval_zhihu.py --out_dir out-zhihu --data_dir data/zhihu_char
"""

import os
import math
import pickle
import argparse

import numpy as np
import torch

from model import GPT, GPTConfig


def load_model(out_dir, device):
    path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}，先把训练跑完。")
    ckpt = torch.load(path, map_location=device)
    sd = ckpt["model"]
    if any(k.startswith("_orig_mod.") for k in sd):     # 兼容 compile=True 的存档
        sd = {k.replace("_orig_mod.", "", 1): v for k, v in sd.items()}
    model = GPT(GPTConfig(**ckpt["model_args"]))
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, ckpt


def load_tokenizer(data_dir):
    """返回 (encode_fn, decode_fn, is_char)。"""
    meta_path = os.path.join(data_dir, "meta.pkl")
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        stoi, itos = meta["stoi"], meta["itos"]
        unk_id = stoi.get("□", 0)

        def encode(s):
            return np.array([stoi.get(c, unk_id) for c in s], dtype=np.int64)

        def decode(ids):
            return "".join(itos.get(int(i), "") for i in ids)

        return encode, decode, True, meta["vocab_size"]

    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    return (lambda s: np.array(enc.encode_ordinary(s), dtype=np.int64),
            lambda ids: enc.decode([int(i) for i in ids]),
            False, 50304)


@torch.no_grad()
def estimate_loss(model, data, block_size, batch_size, device, iters):
    losses = torch.zeros(iters)
    for k in range(iters):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            _, loss = model(x, y)
        losses[k] = loss.item()
    return losses.mean().item()


@torch.no_grad()
def sample(model, prompt_ids, n, max_new_tokens, temperature, top_k, device):
    idx = prompt_ids.unsqueeze(0).repeat(n, 1).to(device)
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return out.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="out-zhihu")
    ap.add_argument("--data_dir", default="data/zhihu_char")
    ap.add_argument("--eval_iters", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_samples", type=int, default=3)
    ap.add_argument("--max_new_tokens", type=int, default=180)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, ckpt = load_model(args.out_dir, device)
    block_size = ckpt["model_args"]["block_size"]
    n_params = sum(p.numel() for p in model.parameters())
    encode, decode, is_char, vocab_size = load_tokenizer(args.data_dir)

    val = np.memmap(os.path.join(args.data_dir, "val.bin"), dtype=np.uint16, mode="r")
    train = np.memmap(os.path.join(args.data_dir, "train.bin"), dtype=np.uint16, mode="r")
    print(f"val 集 {len(val):,} token / train 集 {len(train):,} token")

    print("评估 loss ...")
    val_loss = estimate_loss(model, val, block_size, args.batch_size, device, args.eval_iters)
    train_loss = estimate_loss(model, train, block_size, args.batch_size, device, args.eval_iters)

    ppl = math.exp(val_loss)
    bits_per_token = val_loss / math.log(2)
    baseline = math.log(vocab_size)          # 均匀分布随机猜的 loss

    print(f"训练集 loss {train_loss:.4f} / 验证集 loss {val_loss:.4f} / PPL {ppl:.1f}")

    # ---- 生成样本 ----
    prompts = ["问题：如何评价", "问题：在豆瓣上", "问题：为什么很多人"]
    temps = [0.8, 1.0]
    lines = []
    for t in temps:
        for p in prompts:
            out = sample(model, torch.from_numpy(encode(p)), args.num_samples,
                         args.max_new_tokens, t, 50, device)
            for j in range(out.shape[0]):
                txt = decode(out[j].tolist())
                lines.append(f"### temperature={t}  prompt={p!r}  sample#{j + 1}\n\n```text\n{txt}\n```\n")

    with open("samples.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("生成样本 -> samples.txt")

    # ---- 报告表格 ----
    unit = "字" if is_char else "token"
    md = f"""## 评估结果

| 指标 | 数值 |
|---|---|
| 模型结构 | {ckpt['model_args']['n_layer']} 层 / {ckpt['model_args']['n_head']} 头 / {ckpt['model_args']['n_embd']} 维 |
| 参数量 | {n_params / 1e6:.2f} M |
| 词表大小 | {vocab_size:,} |
| 训练集 loss | {train_loss:.4f} |
| **验证集 loss** | **{val_loss:.4f}** |
| 困惑度 PPL（每{unit}） | {ppl:.2f} |
| bits per {unit} | {bits_per_token:.3f} |
| 随机基线 loss（ln V） | {baseline:.4f} |
| 相对基线下降 | {(1 - val_loss / baseline) * 100:.1f}% |

> 训练集与验证集 loss 的差距（{train_loss:.3f} vs {val_loss:.3f}）反映过拟合程度；
> 随机基线 loss = ln(词表大小) = {baseline:.3f}，是"完全没学到东西"时的理论值。
"""
    with open("eval_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("报告表格 -> eval_report.md")
    print("\n" + md)


if __name__ == "__main__":
    main()
