"""生成报告插图（全部基于真实数据，不含任何虚构数值）。"""

import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"E:\Projects\python_project\AI"
TXT = os.path.join(ROOT, "nanoGPTResources", "zhihu_douban.txt")
OUT = os.path.join(ROOT, "img")
os.makedirs(OUT, exist_ok=True)

INK = "#1f2937"
MUTED = "#6b7280"
BAR = "#2563eb"
BAR2 = "#93c5fd"
GRID = "#e5e7eb"


# ---------------- 图 1：语料记录长度分布 ----------------
text = open(TXT, encoding="utf-8").read()
parts = [p for p in text.split("问题：") if p.strip()]
lengths = np.array([len(p) for p in parts])

median, mean = np.median(lengths), lengths.mean()
p90, p99 = np.percentile(lengths, 90), np.percentile(lengths, 99)

fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=200)
bins = np.logspace(np.log10(max(lengths.min(), 10)), np.log10(lengths.max()), 45)
ax.hist(lengths, bins=bins, color=BAR, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.set_xscale("log")
ax.axvline(median, color="#dc2626", ls="--", lw=1.4, label=f"中位数 {median:.0f} 字")
ax.axvline(mean, color="#f59e0b", ls=":", lw=1.6, label=f"平均值 {mean:.0f} 字")
ax.set_xlabel("单条问答字符数（对数刻度）", color=INK)
ax.set_ylabel("记录数", color=INK)
ax.set_title(f"语料记录长度分布（共 {len(lengths):,} 条）", color=INK, fontsize=12, pad=10)
ax.legend(frameon=False, fontsize=9)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)
fig.text(0.13, -0.03, f"P90 = {p90:.0f} 字    P99 = {p99:.0f} 字    最长 {lengths.max():,} 字",
         color=MUTED, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "语料长度分布.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)


# ---------------- 图 2：模型参数量构成 ----------------
# 与实测输出一致：total = 13,894,272；decayed 13,889,280 + LN 4,992
wte = 8010 * 384
wpe = 512 * 384
ln_per = 384                      # bias=False，每个 LayerNorm 只有 weight
n_ln = 6 * 2 + 1                  # 每块 ln_1/ln_2，加 ln_f
c_attn = 384 * (3 * 384)
attn_proj = 384 * 384
mlp_fc = 384 * (4 * 384)
mlp_proj = (4 * 384) * 384
per_block = c_attn + attn_proj + mlp_fc + mlp_proj
blocks = per_block * 6
total = wte + wpe + blocks + ln_per * n_ln

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.9), dpi=200,
                               gridspec_kw={"width_ratios": [1.25, 1]})

# 左：总体构成
labels = ["Transformer 块 ×6", "词嵌入 wte", "位置嵌入 wpe", "LayerNorm ×7"]
vals = [blocks, wte, wpe, ln_per * n_ln]
colors = ["#2563eb", "#60a5fa", "#a5b4fc", "#cbd5e1"]
y = np.arange(len(labels))[::-1]
ax1.barh(y, vals, color=colors, height=0.6)
for yi, v in zip(y, vals):
    pct = v / total * 100
    txt = f"{v/1e6:.2f}M ({pct:.1f}%)" if v >= 1e5 else f"{v/1e3:.1f}K ({pct:.2f}%)"
    ax1.text(v + total * 0.012, yi, txt, va="center", fontsize=9, color=INK)
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=9.5, color=INK)
ax1.set_xlim(0, total * 0.95)
ax1.set_xlabel("参数量", color=INK, fontsize=9.5)
ax1.set_title(f"总参数量 13.89M（0.0139B）", color=INK, fontsize=11, pad=8)
ax1.grid(axis="x", color=GRID, lw=0.8)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax1.spines[s].set_color(GRID)
ax1.tick_params(colors=MUTED, labelsize=8.5)
ax1.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{x/1e6:.0f}M" if x else "0"))

# 右：单个 Transformer 块的内部构成
labels2 = ["MLP 升维 4d→1536", "MLP 降维 1536→4d", "注意力 QKV", "注意力输出投影"]
vals2 = [mlp_fc, mlp_proj, c_attn, attn_proj]
y2 = np.arange(len(labels2))[::-1]
ax2.barh(y2, vals2, color=[BAR2, BAR2, "#3b82f6", "#3b82f6"], height=0.6)
for yi, v in zip(y2, vals2):
    ax2.text(v + per_block * 0.015, yi, f"{v/1e3:.0f}K",
             va="center", fontsize=8.5, color=INK)
ax2.set_yticks(y2)
ax2.set_yticklabels(labels2, fontsize=9, color=INK)
ax2.set_xlim(0, per_block * 1.02)
ax2.set_xlabel("参数量", color=INK, fontsize=9.5)
ax2.set_title(f"单个 Transformer 块内部构成（{per_block/1e3:.0f}K）",
              color=INK, fontsize=11, pad=8)
ax2.grid(axis="x", color=GRID, lw=0.8)
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax2.spines[s].set_color(GRID)
ax2.tick_params(colors=MUTED, labelsize=8.5)
ax2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
    lambda x, _: f"{x/1e3:.0f}K" if x else "0"))

fig.tight_layout()
fig.savefig(os.path.join(OUT, "参数量构成.png"), bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"记录数 {len(lengths):,}  中位数 {median:.0f}  平均 {mean:.0f}  P90 {p90:.0f}  P99 {p99:.0f}  最长 {lengths.max():,}")
print(f"参数量核对: total {total:,} (期望 13,894,272)")
print("已生成 img/语料长度分布.png  img/参数量构成.png")
