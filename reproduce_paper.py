#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reproduce_paper.py — 从 results/ 的 CSV 一键再生成论文全部表格与图
=====================================================================
输入: experiments.py 跑出的聚合 CSV (<name>.csv) 与逐种子 CSV (<name>_raw.csv)
输出 (默认写到 paper_assets/):
  tables/tab_main.tex      Table I   AMLSim 时间外推
  tables/tab_bank.tex      Table II  跨银行
  tables/tab_drift.tex     Table III 漂移块 F1 + 种子稳定性
  tables/tab_scarcity.tex  Table V   5% 标签稀缺 (ΔAP 对照主表全标签)
  tables/tab_eff.tex       Table VI  效率 (epoch/峰值显存/时延取种子中位数)
  tables/tab_elliptic.tex  Table VII Elliptic 边界结果
  tables/tab_pu.tex        Appendix A PU 学习
  figures/fig_drift.{pdf,png}    Fig. 2 漂移块 F1 折线 (误差棒=种子std)
  figures/fig_stability.{pdf,png} Fig. 3 AP/F1 逐种子箱线图
  (Fig. 1 架构图为论文源码中的 TikZ 矢量图, 不依赖数据, 无需生成。)

用法 (仓库自带 5-seed CSV, 直接可跑):
  python reproduce_paper.py
指向自定义结果 (例如 10-seed 原生文件名):
  python reproduce_paper.py --results_dir . --out_dir paper_assets_10s ^
      --main "TGM-Lite_实验结果_supervised_amlsim_10s.csv" ^
      --bank "TGM-Lite_实验结果_supervised_amlsim_bank_10s.csv" ...
raw 文件自动探测: 依次尝试 <base>_raw.csv 与 <base>_seeds_raw.csv。
仅依赖 numpy/pandas/scipy/matplotlib。
"""

import argparse
import os
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from scipy import stats as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- 显示约定
NAME = {"rf": "RF", "mlp": "MLP", "gcn": "GCN", "evolvegcn": "EvolveGCN",
        "tgmlite_nograph": "TGM-Lite (no-graph)",
        "tgmlite_nossm": "TGM-Lite (no-controller)",
        "tgmlite": "TGM-Lite"}
COLOR = {"rf": "#7f7f7f", "mlp": "#a9a9a9", "gcn": "#1f77b4",
         "evolvegcn": "#ff7f0e", "tgmlite_nossm": "#9467bd",
         "tgmlite_nograph": "#bc8f6f", "tgmlite": "#d62728"}

MAIN_ORDER = ["rf", "mlp", "gcn", "evolvegcn",
              "tgmlite_nograph", "tgmlite_nossm", "tgmlite"]
EFF_ORDER = ["rf", "mlp", "gcn", "evolvegcn",
             "tgmlite_nossm", "tgmlite_nograph", "tgmlite"]
FIVE_ORDER = ["rf", "mlp", "gcn", "evolvegcn", "tgmlite"]
STAB_ORDER = ["rf", "mlp", "gcn", "evolvegcn",
              "tgmlite_nossm", "tgmlite_nograph", "tgmlite"]
DRIFT_FIG_MODELS = ["rf", "mlp", "evolvegcn", "gcn", "tgmlite_nossm", "tgmlite"]


# ----------------------------------------------------------------- 工具
def load(results_dir, fname):
    path = fname if os.path.isabs(fname) else os.path.join(results_dir, fname)
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df


def raw_of(results_dir, agg_fname):
    """由聚合文件名推 raw 文件名, 兼容仓库命名与脚本原生命名."""
    base = agg_fname[:-4] if agg_fname.lower().endswith(".csv") else agg_fname
    for cand in (base + "_raw.csv", base + "_seeds_raw.csv"):
        p = cand if os.path.isabs(cand) else os.path.join(results_dir, cand)
        if os.path.exists(p):
            return load(results_dir, cand)
    raise FileNotFoundError(f"找不到 {agg_fname} 对应的 raw CSV")


def row(agg, model):
    r = agg[agg["model"] == model]
    if r.empty:
        return None
    return r.iloc[0]


def s4(std):
    """std 保留前导 0（IEEE 风格，避免 PDF 抽取歧义）."""
    return f"{std:.4f}"


def pm(mean, std, bold=False, dagger=False):
    core = f"{mean:.4f}"
    if bold:
        core = "\\textbf{%s}" % core
    cell = "%s$\\pm$%s" % (core, s4(std))
    if dagger:
        cell += "$^{\\dagger}$"
    return cell


def welch(a, b):
    t, p = st.ttest_ind(a, b, equal_var=False)
    return t, p


def ms(raw, model, metric):
    """从逐种子 raw 重算 mean / std(ddof=0), 与论文数值约定一致:
    聚合 CSV 的存储精度可能引起末位取舍差异 (如 0.087048 -> .0870/.0871),
    论文表格一律以 raw 重算为准。"""
    v = pd.to_numeric(raw.loc[raw.model == model, metric], errors="coerce").dropna()
    if v.empty:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=0))


def r2(x):
    """十进制四舍五入到 2 位 (与论文手工舍入一致, 避免二进制 1.845->1.84)."""
    return str(Decimal(repr(float(x))).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP))


def write(path, lines):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("  [表]", path)


# ----------------------------------------------------------------- 各表
def tab_main(agg, raw, out):
    cols = [("auc", "AUC"), ("ap", "AP"), ("f1", "F1"),
            ("p", "P"), ("r", "R"), ("p100", "P@100")]
    bold_cols = {"auc", "ap", "f1", "p100"}          # 与论文一致: P/R 不加粗
    stats = {k: {m: ms(raw, k, m) for m, _ in cols}
             for k in MAIN_ORDER if row(agg, k) is not None}
    best = {m: max(stats[k][m][0] for k in stats) for m, _ in cols}
    params = {k: (int(raw[raw.model == k]["params"].iloc[0])
                  if not raw[raw.model == k].empty else 0)
              for k in MAIN_ORDER}
    # †: tgmlite vs evolvegcn (Welch, raw)
    dag = {}
    for m in ("ap", "f1"):
        _, p = welch(raw[raw.model == "tgmlite"][m],
                     raw[raw.model == "evolvegcn"][m])
        dag[m] = p < 0.05
    lines = ["% Table I -- AMLSim time extrapolation (auto-generated)",
             "\\begin{tabular}{l r c c c c c c}", "\\toprule",
             "Model & \\#Params & AUC & AP & F1 & P & R & P@100 \\\\",
             "\\midrule"]
    for k in MAIN_ORDER:
        r = row(agg, k)
        if r is None:
            continue
        label = "\\textbf{TGM-Lite (default)}" if k == "tgmlite" else NAME[k]
        prm = "---" if k == "rf" or params.get(k, 0) == 0 \
            else f"{params[k]:,}"
        cells = []
        for m, _ in cols:
            mu_, sd_ = stats[k][m]
            cells.append(pm(mu_, sd_,
                            bold=(m in bold_cols and mu_ == best[m]),
                            dagger=(k == "tgmlite" and dag.get(m, False))))
        if k == "tgmlite":
            lines.append("\\midrule")
        lines.append(" & ".join([label, prm] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tables", "tab_main.tex"), lines)


def tab_bank(agg, raw, out):
    cols = [("auc", "AUC"), ("ap", "AP"), ("f1", "F1"),
            ("p", "P"), ("r", "R"), ("p100", "P@100")]
    bold_cols = {"auc", "ap", "f1", "r", "p100"}
    stats = {k: {m: ms(raw, k, m) for m, _ in cols}
             for k in MAIN_ORDER if row(agg, k) is not None}
    best = {m: max(stats[k][m][0] for k in stats) for m, _ in cols}
    lines = ["% Table II -- cross-bank (auto-generated)",
             "\\begin{tabular}{l c c c c c c}", "\\toprule",
             "Model & AUC & AP & F1 & P & R & P@100 \\\\", "\\midrule"]
    for k in MAIN_ORDER:
        r = row(agg, k)
        if r is None:
            continue
        label = "\\textbf{TGM-Lite (default)}" if k == "tgmlite" else NAME[k]
        cells = [pm(stats[k][m][0], stats[k][m][1],
                    bold=(m in bold_cols and stats[k][m][0] == best[m]))
                 for m, _ in cols]
        if k == "tgmlite":
            lines.append("\\midrule")
        lines.append(" & ".join([label] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tables", "tab_bank.tex"), lines)


def tab_drift(agg, raw, out):
    stats = {k: {"w1": ms(raw, k, "drift_w1"), "w2": ms(raw, k, "drift_w2"),
                 "w3": ms(raw, k, "drift_w3"), "f1": ms(raw, k, "f1"),
                 "ap": ms(raw, k, "ap")}
             for k in MAIN_ORDER if row(agg, k) is not None}
    best_w = {w: max(stats[k][w][0] for k in stats) for w in ("w1", "w2", "w3")}
    # F1 std: 论文规则 = 神经模型内最优 (RF 方差小但精度不具竞争力, 见正文)
    best_f1std = min(stats[k]["f1"][1] for k in stats if k != "rf")
    lines = ["% Table III -- drift blocks & stability (auto-generated)",
             "\\begin{tabular}{l c c c c c}", "\\toprule",
             "Model & w1 & w2 & w3 (tail) & F1 std & AP std \\\\", "\\midrule"]
    for k in MAIN_ORDER:
        r = row(agg, k)
        if r is None:
            continue
        label = "\\textbf{TGM-Lite (default)}" if k == "tgmlite" else NAME[k]
        cells = []
        for w in ("w1", "w2", "w3"):
            v = f"{stats[k][w][0]:.4f}"
            cells.append("\\textbf{%s}" % v if stats[k][w][0] == best_w[w]
                         else v)
        v = f"{stats[k]['f1'][1]:.4f}"
        cells.append("\\textbf{%s}" % v if (k != "rf" and abs(
            stats[k]["f1"][1] - best_f1std) < 1e-12) else v)
        cells.append(f"{stats[k]['ap'][1]:.4f}")
        if k == "tgmlite":
            lines.append("\\midrule")
        lines.append(" & ".join([label] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tables", "tab_drift.tex"), lines)


def tab_scarcity(agg5, agg5_egcn, agg_full, out):
    """5% 标签表; ΔAP = 5% AP - 全标签 AP (主表), 百分点."""
    pool = pd.concat([agg5, agg5_egcn], ignore_index=True).drop_duplicates(
        subset="model", keep="last")
    cols = [("ap", "AP (5\\%)"), ("f1", "F1 (5\\%)"), ("p100", "P@100 (5\\%)")]
    best = {m: max(row(pool, k)[f"{m}_mean"] for k in FIVE_ORDER
                   if row(pool, k) is not None) for m, _ in cols}
    lines = ["% Table IV -- label scarcity (auto-generated)",
             "\\begin{tabular}{l c c c c}", "\\toprule",
             "Model & AP (5\\%) & $\\Delta$AP (pp) & F1 (5\\%) & P@100 (5\\%) "
             "\\\\", "\\midrule"]
    for k in FIVE_ORDER:
        r = row(pool, k)
        if r is None:
            continue
        full = row(agg_full, k)
        label = "\\textbf{TGM-Lite (default)}" if k == "tgmlite" else NAME[k]
        dap = (r["ap_mean"] - full["ap_mean"]) * 100 if full is not None \
            else float("nan")
        cells = [pm(r["ap_mean"], r["ap_std"],
                    bold=r["ap_mean"] == best["ap"]),
                 f"${dap:.1f}$",
                 pm(r["f1_mean"], r["f1_std"],
                    bold=r["f1_mean"] == best["f1"]),
                 pm(r["p100_mean"], r["p100_std"],
                    bold=r["p100_mean"] == best["p100"])]
        if k == "tgmlite":
            lines.append("\\midrule")
        lines.append(" & ".join([label] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tables", "tab_scarcity.tex"), lines)


def tab_eff(raw, out):
    lines = ["% Table V -- efficiency (auto-generated; medians over seeds)",
             "\\begin{tabular}{l r c c c}", "\\toprule",
             "Model & \\#Params & s/epoch & peak GPU & inference \\\\",
             "\\midrule"]
    for k in EFF_ORDER:
        sub = raw[raw.model == k]
        if sub.empty:
            continue
        label = {"rf": "RF (CPU)", "tgmlite": "\\textbf{TGM-Lite (default)}"}.get(
            k, NAME[k])
        prm = "---" if k == "rf" else f"{int(sub['params'].iloc[0]):,}"
        ep_v = sub["epoch_s"].median()           # epoch 用中位数 (抗热节流离群)
        ep = f"{ep_v:.1f}" if ep_v >= 10 else f"{ep_v:.3g}"
        if k == "rf":
            peak, lat = "---", f"{sub['lat_ms'].mean():.0f}~ms"
        else:                                    # latency 用均值 (与论文一致)
            peak = f"{r2(sub['peak_mb'].median()/1000)}~GB"
            lat = f"{r2(sub['lat_ms'].mean()/1000)}~s"
        if k == "tgmlite":
            lines.append("\\midrule")
        lines.append(" & ".join([label, prm, ep, peak, lat]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tables", "tab_eff.tex"), lines)


def tab_elliptic(agg, out):
    best = {m: max(row(agg, k)[f"{m}_mean"] for k in FIVE_ORDER
                   if row(agg, k) is not None) for m in ("ap", "f1")}
    lines = ["% Table VI -- Elliptic boundary result (auto-generated)",
             "\\begin{tabular}{l c c c}", "\\toprule",
             "Model & AP & F1 & w3 F1 \\\\", "\\midrule"]
    for k in FIVE_ORDER:
        r = row(agg, k)
        if r is None:
            continue
        label = ("\\textbf{%s}" % NAME[k]) if k == "rf" else ("TGM-Lite (default)" if k == "tgmlite" else NAME[k])
        cells = [pm(r["ap_mean"], r["ap_std"], bold=r["ap_mean"] == best["ap"]),
                 pm(r["f1_mean"], r["f1_std"], bold=r["f1_mean"] == best["f1"]),
                 pm(r["w3_mean"], r["w3_std"])]
        lines.append(" & ".join([label] + cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tables", "tab_elliptic.tex"), lines)


def tab_pu(agg, raw, out):
    lines = ["% Table VII (Appendix A) -- PU learning (auto-generated)",
             "\\begin{tabular}{l c c c}", "\\toprule",
             "Model & naive AP & nnPU AP & Welch $t$ \\\\", "\\midrule"]
    for fam, disp in (("mlp", "MLP"), ("tgmlite", "TGM-Lite")):
        rn, rp = row(agg, fam + "_naive"), row(agg, fam + "_pu")
        if rn is None or rp is None:
            continue
        t, p = welch(raw[raw.model == fam + "_pu"]["ap"],
                     raw[raw.model == fam + "_naive"]["ap"])
        tcell = f"${t:+.2f}$ " + ("*" if p < 0.05 else "n.s.")
        lines.append(" & ".join([disp, pm(rn["ap_mean"], rn["ap_std"]),
                                 pm(rp["ap_mean"], rp["ap_std"]), tcell])
                     + " \\\\")
    rn = row(agg, "rf_naive")
    if rn is not None:
        lines.append("\\midrule")
        lines.append(" & ".join(["RF (naive)", pm(rn["ap_mean"], rn["ap_std"]),
                                 "---", "---"]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tables", "tab_pu.tex"), lines)


# ----------------------------------------------------------------- 图
def tab_floor(raw_adapt, raw_floor, out):
    """Table VIII -- 调制切除(静态地板)认证: 自适应 vs 地板逐种子配对。
    raw_adapt: 主实验 tgmlite 逐种子(如 --main 的 _raw);
    raw_floor: --freeze_mod_from <测试起点> 跑的 tgmlite 逐种子。"""
    ra = raw_adapt[raw_adapt.model == "tgmlite"].reset_index(drop=True)
    rf = raw_floor[raw_floor.model == "tgmlite"].reset_index(drop=True)
    n = min(len(ra), len(rf))
    rows = []
    for i in range(n):
        a, f = ra.loc[i], rf.loc[i]
        rows.append(dict(seed=42 + i, ap_a=float(a.ap), ap_f=float(f.ap),
                         w3_a=float(a.drift_w3), w3_f=float(f.drift_w3)))
    # 崩溃种子判定: 切除后 w3 提升 > 0.1 (披露于表注)
    collapsed = [r["seed"] for r in rows if r["w3_f"] - r["w3_a"] > 0.1]
    healthy = [r["seed"] for r in rows if r["seed"] not in collapsed]

    def stat(key):
        v = np.array([r[key] for r in rows])
        return v.mean(), v.std(), v.min()

    # --- 汇总表(正文) ---
    lines = ["% Table VIII -- excision (static-floor) certification (auto-generated)",
             f"% collapsed seeds (rescue Δw3>0.1): {collapsed}",
             "\\begin{tabular}{l c c c c}", "\\toprule",
             "Variant & AP (mean$\\pm$std) & min AP & w3 (mean$\\pm$std) & min w3 \\\\",
             "\\midrule"]
    for name, pre in (("Adaptive", "_a"), ("Static floor ($s_t{=}0$)", "_f")):
        m, s, mn = stat("ap" + pre); m3, s3, mn3 = stat("w3" + pre)
        lines.append(f"{name} & {m:.4f}$\\pm${s:.4f} & {mn:.4f} & "
                     f"{m3:.4f}$\\pm${s3:.4f} & {mn3:.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    write(os.path.join(out, "tables", "tab_floor.tex"), lines)

    # --- 逐种子配对表(附录) ---
    lines = ["% Table A-floor -- per-seed adaptive vs floor (auto-generated)",
             "\\begin{tabular}{c c c c c c c}", "\\toprule",
             "Seed & AP$_{\\rm adapt}$ & AP$_{\\rm floor}$ & $\\Delta$AP & "
             "w3$_{\\rm adapt}$ & w3$_{\\rm floor}$ & $\\Delta$w3 \\\\", "\\midrule"]
    for r in rows:
        mark = "$^{\\dagger}$" if r["seed"] in collapsed else ""
        lines.append(f"{r['seed']}{mark} & {r['ap_a']:.4f} & {r['ap_f']:.4f} & "
                     f"{r['ap_f']-r['ap_a']:+.4f} & {r['w3_a']:.4f} & {r['w3_f']:.4f} & "
                     f"{r['w3_f']-r['w3_a']:+.4f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"% collapsed: {collapsed}; healthy tax "
              f"Δw3={np.mean([r['w3_f']-r['w3_a'] for r in rows if r['seed'] in healthy]):+.4f}, "
              f"rescue Δw3={np.mean([r['w3_f']-r['w3_a'] for r in rows if r['seed'] in collapsed]):+.4f}"
              if collapsed else "% no collapsed seeds"]
    write(os.path.join(out, "tables", "tab_floor_seeds.tex"), lines)


def fig_floor(raw_adapt, raw_floor, out):
    """Fig.4 -- 切除救援哑铃图: 每种子的 w3 F1, 自适应 vs 静态地板."""
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    ra = raw_adapt[raw_adapt.model == "tgmlite"].reset_index(drop=True)
    rf = raw_floor[raw_floor.model == "tgmlite"].reset_index(drop=True)
    n = min(len(ra), len(rf))
    seeds = list(range(42, 42 + n))
    wa = [float(ra.loc[i, "drift_w3"]) for i in range(n)]
    wf = [float(rf.loc[i, "drift_w3"]) for i in range(n)]
    collapsed = [seeds[i] for i in range(n) if wf[i] - wa[i] > 0.1]
    fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=200)
    for i, s in enumerate(seeds):
        c = "#d62728" if s in collapsed else "#7f7f7f"
        ax.plot([wa[i], wf[i]], [i, i], color=c, lw=1.4, zorder=1)
    ax.scatter(wa, range(n), color="#1f77b4", zorder=2, s=42, label="Adaptive")
    ax.scatter(wf, range(n), color="#2ca02c", zorder=2, s=42,
               marker="s", label="Static floor ($s_t{=}0$)")
    ax.set_yticks(range(n)); ax.set_yticklabels([f"seed {s}" for s in seeds])
    ax.set_xlabel("Drift-tail F1 (w3, validation threshold)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Excision rescue: collapsed seeds recover to a uniform floor",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "fig_floor.pdf"))
    fig.savefig(os.path.join(out, "figures", "fig_floor.png"))
    plt.close(fig)
    print("  [图] fig_floor.pdf/.png")


def fig_drift(agg, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [0, 1, 2]
    for k in DRIFT_FIG_MODELS:
        r = row(agg, k)
        if r is None:
            continue
        y = [r["w1_mean"], r["w2_mean"], r["w3_mean"]]
        e = [r["w1_std"], r["w2_std"], r["w3_std"]]
        ours = k == "tgmlite"
        lab = "TGM-Lite (ours)" if ours else NAME[k]
        ax.errorbar(xs, y, yerr=e, marker="o", ms=10 if ours else 7,
                    lw=4 if ours else 2, color=COLOR[k],
                    capsize=5 if ours else 3, label=lab, zorder=5 if ours else 3)
    ax.set_xticks(xs)
    ax.set_xticklabels(["w1 (near)", "w2 (mid)", "w3 (far)"], fontsize=13)
    ax.set_xlabel("Test-period drift block", fontsize=15)
    ax.set_ylabel("F1 (validation threshold)", fontsize=15)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    # legend 两列: 左列 RF/MLP/EvolveGCN, 右列 GCN/nossm/ours (行主序填充)
    h, l = ax.get_legend_handles_labels()
    want = ["RF", "GCN", "MLP", NAME["tgmlite_nossm"], "EvolveGCN",
            "TGM-Lite (ours)"]
    order = [l.index(w) for w in want if w in l]
    ax.legend([h[i] for i in order], [l[i] for i in order],
              ncol=2, fontsize=12, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"fig_drift.{ext}"), dpi=200)
    plt.close(fig)
    print("  [图]", os.path.join(out, "figures", "fig_drift.pdf/.png"))


def fig_stability(raw, out):
    n = int(raw.groupby("model").size().max())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, met, tag in ((axes[0], "ap", "AP"), (axes[1], "f1", "F1")):
        data, colors, labels = [], [], []
        for k in STAB_ORDER:
            sub = raw[raw.model == k][met].dropna().values
            if len(sub) == 0:
                continue
            data.append(sub)
            colors.append(COLOR[k])
            labels.append("TGM-Lite (ours)" if k == "tgmlite" else
                          NAME[k].replace("TGM-Lite", "TGM-Lite"))
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        medianprops=dict(color="black", lw=1.8),
                        whiskerprops=dict(color="black"),
                        capprops=dict(color="black"),
                        flierprops=dict(marker=""))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.55 if c not in ("#d62728",) else 0.85)
            patch.set_edgecolor("black")
        rng = np.random.default_rng(0)
        for i, d in enumerate(data):
            ax.scatter(1 + i + rng.uniform(-0.12, 0.12, len(d)), d,
                       color="black", alpha=0.65, s=28, zorder=5)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=28, ha="right", fontsize=11)
        ax.set_ylabel(tag, fontsize=15)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(f"({'ab'[met == 'f1']}) {n}-seed {tag} distribution "
                     f"(AMLSim, time extrapolation)", fontsize=13)
    fig.tight_layout()
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, "figures", f"fig_stability.{ext}"),
                    dpi=200)
    plt.close(fig)
    print("  [图]", os.path.join(out, "figures", "fig_stability.pdf/.png"))


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="paper_assets")
    ap.add_argument("--main", default="amlsim_time_extrapolation_5seeds.csv")
    ap.add_argument("--bank", default="amlsim_cross_bank_5seeds.csv")
    ap.add_argument("--scarcity",
                    default="amlsim_label_scarcity_5pct_5seeds.csv")
    ap.add_argument("--scarcity_egcn",
                    default="amlsim_label_scarcity_5pct_evolvegcn_5seeds.csv")
    ap.add_argument("--elliptic",
                    default="elliptic_time_extrapolation_5seeds.csv")
    ap.add_argument("--pu", default="elliptic_pu_5seeds.csv")
    ap.add_argument("--floor", default="",
                    help="调制切除(静态地板)认证的逐种子 CSV(--freeze_mod_from "
                         "跑出); 提供后生成 Table VIII 及附录逐种子配对表")
    args = ap.parse_args()
    rd, out = args.results_dir, args.out_dir
    os.makedirs(out, exist_ok=True)

    print("== 生成论文表格 ==")
    agg_main, raw_main = load(rd, args.main), raw_of(rd, args.main)
    tab_main(agg_main, raw_main, out)
    tab_bank(load(rd, args.bank), raw_of(rd, args.bank), out)
    tab_drift(agg_main, raw_main, out)
    tab_scarcity(load(rd, args.scarcity), load(rd, args.scarcity_egcn),
                 agg_main, out)
    tab_eff(raw_main, out)
    tab_elliptic(load(rd, args.elliptic), out)
    agg_pu = load(rd, args.pu)
    tab_pu(agg_pu, raw_of(rd, args.pu), out)
    if args.floor:
        tab_floor(raw_main, raw_of(rd, args.floor), out)
        fig_floor(raw_main, raw_of(rd, args.floor), out)

    print("== 生成论文图 ==")
    fig_drift(agg_main, out)
    fig_stability(raw_main, out)
    print("完成。Fig.1 架构图为论文 LaTeX 源码内 TikZ, 无需数据。")


if __name__ == "__main__":
    main()
