"""
把 zhihu_douban.txt 切成 nanoGPT 标准的 train.bin / val.bin。

放在 nanoGPT 仓库根目录运行。

推荐（字符级，无需联网）：
    python prepare_zhihu.py --input zhihu_douban.txt --out_dir data/zhihu_char

备选（GPT-2 BPE，需要能下载 tiktoken 的 gpt2 词表）：
    python prepare_zhihu.py --input zhihu_douban.txt --out_dir data/zhihu_bpe --tokenizer gpt2
"""

import os
import argparse
import pickle
from collections import Counter

import numpy as np


def split_text(text, val_ratio):
    """按段落边界切分，避免把一条问答从句中劈成两半。"""
    cut = len(text) - int(len(text) * val_ratio)
    nl = text.find("\n", cut)
    if nl == -1:
        nl = cut
    return text[:nl], text[nl:]


def build_char_vocab(text, min_freq, unk="□"):
    counts = Counter(text)
    kept = sorted(c for c, n in counts.items() if n >= min_freq)
    stoi = {ch: i for i, ch in enumerate(kept)}

    unk_id = None
    if len(kept) < len(counts):          # 有低频字被裁掉了，需要一个 unk 槽位
        if unk in stoi:
            unk_id = stoi[unk]
        else:
            unk_id = len(stoi)
            stoi[unk] = unk_id
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos, unk_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="zhihu_douban.txt")
    ap.add_argument("--out_dir", default="data/zhihu_char")
    ap.add_argument("--tokenizer", choices=["char", "gpt2"], default="char")
    ap.add_argument("--val_ratio", type=float, default=0.005)
    ap.add_argument("--min_freq", type=int, default=1,
                    help="字符级：出现次数低于该值的字并入 unk。1 表示全部保留。")
    args = ap.parse_args()

    print(f"[1/4] 读取 {args.input} ...")
    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()
    print(f"      总字符数: {len(text):,}")

    train_text, val_text = split_text(text, args.val_ratio)
    print(f"[2/4] train {len(train_text):,} 字符 / val {len(val_text):,} 字符")

    os.makedirs(args.out_dir, exist_ok=True)

    if args.tokenizer == "char":
        stoi, itos, unk_id = build_char_vocab(train_text + val_text, args.min_freq)
        vocab_size = len(stoi)
        print(f"[3/4] 字符级词表 vocab_size = {vocab_size}"
              f"{' (含 unk 槽位)' if unk_id is not None else ''}")
        with open(os.path.join(args.out_dir, "meta.pkl"), "wb") as f:
            pickle.dump({"vocab_size": vocab_size, "stoi": stoi, "itos": itos}, f)

        def encode(s):
            if unk_id is None:
                ids = [stoi[c] for c in s]
            else:
                ids = [stoi.get(c, unk_id) for c in s]
            return np.fromiter(ids, dtype=np.uint16, count=len(ids))

        train_ids, val_ids = encode(train_text), encode(val_text)
    else:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        print("[3/4] 使用 GPT-2 BPE 编码（vocab_size = 50257）")
        train_ids = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
        val_ids = np.array(enc.encode_ordinary(val_text), dtype=np.uint16)
        # 不写 meta.pkl，train.py 会自动用默认词表

    train_ids.tofile(os.path.join(args.out_dir, "train.bin"))
    val_ids.tofile(os.path.join(args.out_dir, "val.bin"))

    print(f"[4/4] 完成 -> {args.out_dir}/")
    print(f"      train.bin {len(train_ids):,} tokens ({len(train_ids) * 2 / 1e6:.1f} MB)")
    print(f"      val.bin   {len(val_ids):,} tokens")
    print(f"      每 epoch 约 {len(train_ids):,} token，"
          f"按 batch32 x block512 x accum4 计算约 {len(train_ids) / 65536:.0f} iter")


if __name__ == "__main__":
    main()
