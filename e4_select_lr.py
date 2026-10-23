# -*- coding: utf-8 -*-
"""
e4_select_lr.py — E4 阶段一汇总: 按 val_AP 为每个外部基线选学习率
================================================================================
读 E4_公平外部基线_调参.bat 阶段一产生的 9 个试点 CSV
(命名: TGM-Lite_审计结果_supervised_amlsim_tune_lr<LR>_seeds_raw.csv,
 其中 <LR> 为 5em4 / 5em3 / 5em2), 按 (model, lr) 聚合 val_ap 均值,
对每个模型给出 argmax 学习率, 并打印可直接复制的阶段二命令。

选择只用验证窗 AP (val 37-48), 测试窗全程不参与 —— 与主文
"checkpoint selection uses validation AP only" 的口径一致。

用法: python e4_select_lr.py
"""
import glob
import os
import re
from collections import defaultdict

import pandas as pd

PAT = re.compile(r"TGM-Lite_审计结果_supervised_amlsim_tune_lr(\w+)_seeds_raw\.csv$")


def decode(tag):
    # 5em4 -> 5e-4
    m = re.match(r"(\d+)em(\d+)", tag)
    return float(f"{m.group(1)}e-{m.group(2)}") if m else None


def main():
    rows = defaultdict(list)   # (model, lr) -> [val_ap, ...]
    files = sorted(glob.glob("TGM-Lite_审计结果_supervised_amlsim_tune_lr*_seeds_raw.csv"))
    if not files:
        print("未找到试点 CSV —— 先运行 E4_公平外部基线_调参.bat 的阶段一。")
        return
    for f in files:
        lr = decode(PAT.search(os.path.basename(f)).group(1))
        df = pd.read_csv(f)
        if "val_ap" not in df.columns:
            raise SystemExit(f"{f} 缺少 val_ap 列 —— 请使用仓库里已打补丁的 "
                             f"audit_external.py (2026-10-17 之后版本)。")
        for _, r in df.iterrows():
            rows[(r["model"], lr)].append(float(r["val_ap"]))

    print(f"{'model':<16} {'lr':<8} {'n':<3} mean val_AP")
    best = {}
    for (model, lr) in sorted(rows, key=lambda k: (k[0], k[1])):
        v = rows[(model, lr)]
        mval = sum(v) / len(v)
        print(f"{model:<16} {lr:<8} {len(v):<3} {mval:.4f}")
        if model not in best or mval > best[model][1]:
            best[model] = (lr, mval)

    print("\n== 阶段二推荐命令 (每模型以其 val-AP 最优 lr 跑 10 seeds) ==")
    for model, (lr, mval) in sorted(best.items()):
        lr_s = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e+")
        print(f"python audit_external.py --data amlsim --models {model} "
              f"--lr {lr_s} --epochs 100 --pretrain_epochs 5 "
              f"--seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10")
    print("\n把阶段二 CSV 发给助手以更新 Table 9 (调参后行)。")


if __name__ == "__main__":
    main()
