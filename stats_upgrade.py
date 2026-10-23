#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stats_upgrade.py — TGM-Lite 统计升级分析脚本
=====================================================================
针对审稿意见的统计强化, 直接读取 experiments.py 生成的 *_raw.csv
(每行 = 一个 model x 一个 seed 的一次运行), 输出:

  1) 描述统计: n, mean, sd (ddof=1), 95% CI of mean
  2) Welch t 检验 (vs 参考模型, 默认 tgmlite): t, df, p_raw
  3) 多重比较校正: Holm (FWER) 与 Benjamini-Hochberg (FDR), 在每个
     实验 x 指标 的比较族内分别进行
  4) 均值差 95% CI (Welch)
  5) 效应量: Hedges' g (小样本校正 Cohen's d) 与 Cliff's delta
  6) 方差检验: Levene (center=mean) 与 Brown-Forsythe (center=median),
     及方差比 F 检验  —— 回应 "稳定性差异需正式检验" 的意见
  7) 非劣效 / 等效检验: 对指定 margin (默认 0.02) 做单侧非劣效检验与
     TOST 等效检验 —— 把 "与 GCN 打平" 转化为 "可证明非劣/等效"

输出 (写入 --out_dir, 默认 <results_dir>/stats):
  <group>_stats_long.csv   全部数值结果 (长表)
  <group>_table.tex        LaTeX (booktabs) 效应量表, 可直接进论文
  all_stats_long.csv       全部实验合并
  控制台打印每个实验的 headline 结论

用法 (Windows / PyCharm 均可, 仅依赖 numpy/pandas/scipy):
  python stats_upgrade.py --results_dir results --ref tgmlite
  # 5% 标签档: tgmlite 与 evolvegcn 在两个文件里, 用 --pool 合并为一个比较族
  python stats_upgrade.py --results_dir results --ref tgmlite ^
      --pool "amlsim_label_scarcity_5pct_5seeds_raw.csv+amlsim_label_scarcity_5pct_evolvegcn_5seeds_raw.csv:amlsim_label5_pooled"
  # 自定义非劣效界值与指标
  python stats_upgrade.py --results_dir results --ref tgmlite --margin 0.02 --metrics ap,f1,p100
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as st

METRICS_DEFAULT = ["ap", "f1", "p100", "auc",
                   "drift_w1", "drift_w2", "drift_w3",
                   "epoch_s", "lat_ms"]


# ---------------------------------------------------------------- 核心统计
def welch_test(a, b):
    """Welch t 检验 + 均值差 95% CI. 返回 dict."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    diff = ma - mb
    if se == 0:
        return dict(diff=diff, t=np.nan, df=np.nan, p=np.nan,
                    ci_lo=np.nan, ci_hi=np.nan)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1)
                                     + (vb / nb) ** 2 / (nb - 1))
    t = diff / se
    p = 2 * st.t.sf(abs(t), df)
    half = st.t.ppf(0.975, df) * se
    return dict(diff=diff, t=t, df=df, p=p,
                ci_lo=diff - half, ci_hi=diff + half)


def hedges_g(a, b):
    """Hedges' g: 小样本校正的 Cohen's d (正 = a 优于 b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    if sp == 0:
        return np.nan
    d = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (na + nb) - 9)          # 小样本校正因子
    return J * d


def cliffs_delta(a, b):
    """Cliff's delta: 非参数效应量, P(a>b) - P(a<b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    gt = sum(np.sum(x > b) for x in a)
    lt = sum(np.sum(x < b) for x in a)
    return (gt - lt) / (len(a) * len(b))


def variance_tests(a, b):
    """Levene (mean), Brown-Forsythe (median), 方差比 F 检验."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    va0, vb0 = a.var(ddof=1), b.var(ddof=1)
    if va0 == 0 and vb0 == 0:                    # 两组均无变异 (如 p100 恒为 1)
        return np.nan, np.nan, np.nan, np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        lev_p = st.levene(a, b, center="mean").pvalue
        bf_p = st.levene(a, b, center="median").pvalue
    if min(va0, vb0) == 0:
        return lev_p, bf_p, np.nan, np.nan
    va, vb = va0, vb0
    big, small = max(va, vb), min(va, vb)
    dfn = len(a) - 1 if va >= vb else len(b) - 1
    dfd = len(b) - 1 if va >= vb else len(a) - 1
    F = big / small
    f_p = 2 * st.f.sf(F, dfn, dfd)           # 双侧
    return lev_p, bf_p, F, min(f_p, 1.0)


def noninferiority_tost(a, b, margin):
    """a=参考(tgmlite), b=对照. margin 为非劣效/等效界值 (原始刻度).
    ni_p : H0: mu_a - mu_b <= -margin (非劣效, 越小越非劣)
    tost_p: 等效 (TOST, 两个单侧取 max)"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return np.nan, np.nan
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1)
                                     + (vb / nb) ** 2 / (nb - 1))
    diff = a.mean() - b.mean()
    t_ni = (diff + margin) / se              # vs 下界 -margin
    p_ni = st.t.sf(t_ni, df)
    t_up = (margin - diff) / se              # vs 上界 +margin
    p_up = st.t.sf(t_up, df)
    return p_ni, max(p_ni, p_up)


# ---------------------------------------------------------------- 校正与标注
def holm_adjust(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min((m - rank) * p[idx], 1.0))
        adj[idx] = running
    return adj


def bh_adjust(pvals):
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p, kind="stable")[::-1]
    adj = np.empty(m)
    running = 1.0
    for rank_rev, idx in enumerate(order):
        rank = m - rank_rev
        running = min(running, p[idx] * m / rank)
        adj[idx] = min(running, 1.0)
    return adj


def stars(p):
    if pd.isna(p):
        return ""
    return "***" if p < 1e-3 else "**" if p < 1e-2 else \
           "*" if p < 5e-2 else "." if p < 1e-1 else ""


def mag_g(g):
    g = abs(g)
    return "large" if g >= 0.8 else "medium" if g >= 0.5 else \
           "small" if g >= 0.2 else "negligible"


def mag_c(d):
    d = abs(d)
    return "large" if d >= 0.474 else "medium" if d >= 0.33 else \
           "small" if d >= 0.147 else "negligible"


# ---------------------------------------------------------------- 分组分析
def load_raw(results_dir, fname):
    path = os.path.join(results_dir, fname)
    df = pd.read_csv(path, encoding="utf-8-sig")   # 处理 BOM
    df.columns = [c.strip() for c in df.columns]
    return df


def resolve_ref(models, ref):
    """精确匹配 -> 唯一前缀匹配 -> 报错列出可用模型."""
    if ref in models:
        return ref
    pref = [m for m in models if m.startswith(ref)]
    if len(pref) == 1:
        return pref[0]
    return None


def analyze_group(df, name, ref, metrics, margin, out_dir):
    metrics = [m for m in metrics if m in df.columns]
    models = list(dict.fromkeys(df["model"]))
    ref_m = resolve_ref(models, ref)
    desc_rows, pair_rows = [], []

    for met in metrics:
        # ---- 描述统计
        for mdl in models:
            x = df.loc[df["model"] == mdl, met].dropna().values
            if len(x) == 0:
                continue
            sd = x.std(ddof=1) if len(x) > 1 else 0.0
            half = (st.t.ppf(0.975, len(x) - 1) * sd / np.sqrt(len(x))
                    if len(x) > 1 else 0.0)
            desc_rows.append(dict(experiment=name, metric=met, model=mdl,
                                  n=len(x), mean=x.mean(), sd=sd,
                                  ci_lo=x.mean() - half, ci_hi=x.mean() + half))
        if ref_m is None or len(models) < 2:
            continue
        # ---- 与参考模型两两比较
        fam = []                                   # 一个 指标 内的比较族
        xr = df.loc[df["model"] == ref_m, met].dropna().values
        for mdl in models:
            if mdl == ref_m:
                continue
            xo = df.loc[df["model"] == mdl, met].dropna().values
            if len(xr) < 2 or len(xo) < 2:
                continue
            w = welch_test(xr, xo)
            g = hedges_g(xr, xo)
            cd = cliffs_delta(xr, xo)
            lev_p, bf_p, vratio, f_p = variance_tests(xr, xo)
            ni_p, tost_p = noninferiority_tost(xr, xo, margin)
            fam.append(dict(experiment=name, metric=met, ref=ref_m,
                            other=mdl, n_ref=len(xr), n_other=len(xo),
                            mean_ref=xr.mean(), mean_other=xo.mean(),
                            sd_ref=xr.std(ddof=1), sd_other=xo.std(ddof=1),
                            diff=w["diff"], ci_lo=w["ci_lo"], ci_hi=w["ci_hi"],
                            t=w["t"], df=w["df"], p_raw=w["p"],
                            hedges_g=g, g_mag=mag_g(g) if pd.notna(g) else "",
                            cliffs_delta=cd, c_mag=mag_c(cd),
                            levene_p=lev_p, brownforsythe_p=bf_p,
                            var_ratio=vratio, f_test_p=f_p,
                            ni_p=ni_p, tost_p=tost_p))
        if fam:
            adj_h = holm_adjust([r["p_raw"] for r in fam])
            adj_b = bh_adjust([r["p_raw"] for r in fam])
            for r, ph, pb in zip(fam, adj_h, adj_b):
                r["p_holm"] = ph
                r["p_bh"] = pb
                r["sig"] = stars(ph)
            pair_rows.extend(fam)

    desc_df = pd.DataFrame(desc_rows)
    pair_df = pd.DataFrame(pair_rows)
    long_df = pair_df
    out_csv = os.path.join(out_dir, f"{name}_stats_long.csv")
    long_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    desc_df.to_csv(os.path.join(out_dir, f"{name}_descriptives.csv"),
                   index=False, encoding="utf-8-sig")
    if not pair_df.empty:
        write_tex(pair_df, name, os.path.join(out_dir, f"{name}_table.tex"))
    return desc_df, pair_df


def esc(s):
    return str(s).replace("_", "\\_")


def write_tex(df, name, path):
    """booktabs 效应量表: 参考模型 vs 各对照, 按指标分块."""
    lines = [
        f"% Auto-generated by stats_upgrade.py -- experiment: {esc(name)}",
        "\\begin{table}[!t]", "\\centering",
        "\\caption{Pairwise statistics vs.\\ the reference model "
        "(Welch $t$, Holm-adjusted $p$, Hedges' $g$, Cliff's $\\delta$, "
        "Brown--Forsythe variance test).}",
        "\\label{tab:stats_%s}" % name.replace("-", "_"),
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{llrrrrr}", "\\toprule",
        "Metric & Other & $\\Delta$ [95\\% CI] & $p_{\\mathrm{Holm}}$ "
        "& $g$ & $\\delta_C$ & $p_{\\mathrm{BF}}$ \\\\",
        "\\midrule"]
    for met, sub in df.groupby("metric", sort=False):
        first = True
        for _, r in sub.iterrows():
            ci = f"{r['diff']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]"
            ph = f"{r['p_holm']:.3f}{r['sig']}" if pd.notna(r["p_holm"]) else "--"
            g = f"{r['hedges_g']:+.2f}" if pd.notna(r["hedges_g"]) else "--"
            cd = f"{r['cliffs_delta']:+.2f}"
            bf = f"{r['brownforsythe_p']:.3f}"
            lines.append("%s & %s & %s & %s & %s & %s & %s \\\\" % (
                esc(met) if first else "", esc(r["other"]), ci, ph, g, cd, bf))
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}", ""]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--ref", default="tgmlite")
    ap.add_argument("--margin", type=float, default=0.02,
                    help="非劣效/等效界值 (原始指标刻度, 默认 0.02)")
    ap.add_argument("--metrics", default=",".join(METRICS_DEFAULT))
    ap.add_argument("--pool", action="append", default=[],
                    help='合并多个 raw 文件为一个比较族: "f1.csv+f2.csv:组名"')
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    rd = args.results_dir
    out_dir = args.out_dir or os.path.join(rd, "stats")
    os.makedirs(out_dir, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    groups = []
    for f in sorted(glob.glob(os.path.join(rd, "*_raw.csv"))):
        base = os.path.basename(f)
        if any(base in p for p in args.pool):
            continue                                # 被 pool 收编的不单独跑
        groups.append((base[:-len("_raw.csv")], [base]))
    for spec in args.pool:
        files, _, gname = spec.partition(":")
        fl = [s.strip() for s in files.split("+") if s.strip()]
        groups.append((gname or "pooled_" + "_".join(fl)[:40], fl))

    all_pairs = []
    for name, files in groups:
        df = pd.concat([load_raw(rd, f) for f in files], ignore_index=True)
        df = df.drop_duplicates()
        models = sorted(df["model"].unique())
        n_max = df.groupby("model").size().max()
        print(f"\n{'='*72}\n[{name}]  models={models}  (max n={n_max})")
        if n_max < 10:
            print(f"  !! 警告: n={n_max} < 10, 检验功效不足, "
                  f"请补跑 10-seed (seeds 42-51) 后重跑本脚本")
        ref_m = resolve_ref(models, args.ref)
        if ref_m is None:
            print(f"  !! 参考模型 '{args.ref}' 不在本组, 仅输出描述统计")
        desc, pair = analyze_group(df, name, ref_m or args.ref,
                                   metrics, args.margin, out_dir)
        if not pair.empty:
            all_pairs.append(pair)
            for met, sub in pair.groupby("metric", sort=False):
                if met not in ("ap", "f1", "p100"):
                    continue
                for _, r in sub.iterrows():
                    ni = ("NI✓" if pd.notna(r["ni_p"]) and r["ni_p"] < .05
                          else "NI✗")
                    print(f"  {met:>4}: {ref_m} vs {r['other']:<10} "
                          f"Δ={r['diff']:+.4f} "
                          f"[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] "
                          f"p_Holm={r['p_holm']:.3f}{r['sig']:<3} "
                          f"g={r['hedges_g']:+.2f}({r['g_mag']}) "
                          f"δc={r['cliffs_delta']:+.2f} "
                          f"BF_p={r['brownforsythe_p']:.3f} {ni}")

    if all_pairs:
        big = pd.concat(all_pairs, ignore_index=True)
        big.to_csv(os.path.join(out_dir, "all_stats_long.csv"),
                   index=False, encoding="utf-8-sig")
    print(f"\n输出目录: {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
