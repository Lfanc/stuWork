"""
把 nanoGPT 的 train.py 日志后端从 WandB 换成 SwanLab（不改动其它逻辑）。

放在 nanoGPT 根目录运行：
    python apply_swanlab_patch.py

做了什么：
  1. 备份 train.py -> train.py.wandb.bak（已存在则不覆盖）
  2. 所有 `import wandb` 替换为 `import swanlab as wandb`
  3. `wandb.init(project=..., name=...)` 的 name= 改成 SwanLab 的 experiment_name=
  4. 把 wandb.finish() 之类的收尾换成 swanlab.finish()（若存在）

因为变量名保持不变，config 里的 wandb_log / wandb_project / wandb_run_name 不用改。
"""

import re
import shutil
from pathlib import Path

TRAIN = Path("train.py")
BACKUP = Path("train.py.wandb.bak")


def main():
    if not TRAIN.exists():
        raise SystemExit("找不到 train.py，请在 nanoGPT 仓库根目录运行本脚本。")

    src = TRAIN.read_text(encoding="utf-8")

    if "import swanlab" in src:
        raise SystemExit("train.py 里已经有 swanlab 了，可能已经打过补丁，未做改动。")

    if not BACKUP.exists():
        shutil.copy2(TRAIN, BACKUP)
        print(f"[备份] {TRAIN} -> {BACKUP}")

    out = src

    # 1) import wandb -> import swanlab as wandb（保留原缩进）
    out, n_import = re.subn(r"^([ \t]*)import wandb[ \t]*$",
                            r"\1import swanlab as wandb", out, flags=re.M)

    # 2) init 的参数名：name= -> experiment_name=
    out, n_init = re.subn(r"(wandb\.init\([^)]*?)\bname\s*=",
                          r"\1experiment_name=", out)
    # 如果 init 跨行，上面的非贪婪匹配会失效，补一次针对 name=wandb_run_name 的替换
    if n_init == 0:
        out, n_init = re.subn(r"\bname\s*=\s*wandb_run_name", "experiment_name=wandb_run_name", out)

    # 3) 收尾
    out = out.replace("wandb.finish()", "swanlab.finish()")

    if n_import == 0:
        raise SystemExit(
            "没找到 `import wandb`，你的 train.py 可能不是官方原版。\n"
            "请手动把 import wandb 改成 import swanlab as wandb（共 2 处），"
            "并把 wandb.init 的 name= 改成 experiment_name=。"
        )

    TRAIN.write_text(out, encoding="utf-8")

    print(f"[完成] 替换 import: {n_import} 处 / init 参数: {n_init} 处 / finish: 已处理")
    print("[提示] 删除 WandB 依赖不必要，swanlab 与 wandb 可共存；"
          "train.py 不再 import wandb。")
    print("[提示] 回滚：copy train.py.wandb.bak train.py")


if __name__ == "__main__":
    main()
