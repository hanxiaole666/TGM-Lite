# -*- coding: utf-8 -*-
"""
fig_trajectory.py — E6 配套作图: 崩溃 vs 健康 seed 的调制轨迹双面板图
================================================================================
读 audit_trajectory.py 生成的 CSV, 输出 fig_trajectory.pdf/.png:

  面板 A (左): ||U·diag(s_t)·V||_F / ||W0||_F 随窗口 t 的轨迹
               —— 权重空间实际修改量占静态主干的比例
  面板 B (右): max|s_t| 随窗口 t 的轨迹 —— 控制器输出的最大幅值

健康 seed 蓝色实线、崩溃 seed 红色实线; 三条竖线分别标注
训练/验证边界、切除点 t_exc (红虚线)、验证/测试边界。
图例自动带每种子的 w3 F1, 便于一眼对应"轨迹形态 -> 测试结局"。

用法:
  python fig_trajectory.py --csv TGM-Lite_轨迹审计_amlsim_traj.csv ^
      --train_end 36 --val_end 48 --floor_from 40 --out fig_trajectory
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--train_end", type=int, default=36)
    ap.add_argument("--val_end", type=int, default=48)
    ap.add_argument("--floor_from", type=int, default=40)
    ap.add_argument("--out", default="fig_trajectory")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    seeds = sorted({int(r["seed"]) for r in rows})
    per = {}
    for sd in seeds:
        rs = sorted((r for r in rows if int(r["seed"]) == sd),
                    key=lambda r: int(r["t"]))
        dw = [float(r["dW_rel"]) for r in rs]
        # 逐窗变化量 (regime-shift 检测): s_t=tanh 饱和, max|s_t| 恒≈1 无信息,
        # 机制信号在符号模式的重构 —— 用 |Δ‖ΔW‖| 刻画
        ddw = [0.0] + [abs(dw[i] - dw[i - 1]) for i in range(1, len(dw))]
        per[sd] = dict(
            t=[int(r["t"]) for r in rs],
            dW_rel=dw,
            s_max=ddw,
            w3=float(rs[0]["w3"]), collapsed=int(rs[0]["collapsed"]))

    C_OK, C_BAD = "#1f77b4", "#d62728"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for ax, key, ylab in (
            (axes[0], "dW_rel", r"$\|U\,\mathrm{diag}(s_t)\,V\|_F\,/\,\|W_0\|_F$"),
            (axes[1], "s_max", r"$|\,\Delta\|U\,\mathrm{diag}(s_t)\,V\|_F/\|W_0\|_F\,|$ (window-over-window)")):
        for sd in seeds:
            d = per[sd]
            col = C_BAD if d["collapsed"] else C_OK
            lab = (f"seed {sd} (collapsed, w3={d['w3']:.3f})" if d["collapsed"]
                   else f"seed {sd} (healthy, w3={d['w3']:.3f})")
            ax.plot(d["t"], d[key], color=col, lw=2.0 if d["collapsed"] else 1.6,
                    alpha=0.95 if d["collapsed"] else 0.8, label=lab)
        ax.axvline(args.train_end + 0.5, color="0.55", ls=":", lw=1)
        ax.axvline(args.floor_from - 0.5, color=C_BAD, ls="--", lw=1.2)
        ax.axvline(args.val_end + 0.5, color="0.55", ls=":", lw=1)
        ax.text(args.floor_from - 0.5, ax.get_ylim()[1], r" $t_{\mathrm{exc}}$",
                color=C_BAD, fontsize=10, va="top")
        ax.set_xlabel("window $t$", fontsize=12)
        ax.set_ylabel(ylab, fontsize=12)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9, framealpha=0.9)
    axes[0].set_title("(a) weight-space modification magnitude", fontsize=12)
    axes[1].set_title("(b) per-window sign-pattern change", fontsize=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{args.out}.{ext}", dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
    print(f"[已保存] {args.out}.pdf / .png")


if __name__ == "__main__":
    main()
