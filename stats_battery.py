# -*- coding: utf-8 -*-
"""stats_battery.py — 重新生成论文附录 Table (tab_stats.tex) 的完整显著性检验电池.

从原始逐种子 CSV 重新计算论文引用的全部 20 个显著性检验:
Welch t(独立) / 配对 t, Hedges g / d_z 效应量, 20,000 次 percentile bootstrap
均值差 95% CI, RQ 族内 Holm-Bonferroni 校正, 全电池 Benjamini-Hochberg q 值.

用法:
  python stats_battery.py --results_dir <CSV目录> --out_dir <输出目录>

所需 CSV (文件名与仓库 results/ 一致):
  TGM-Lite_实验结果_supervised_amlsim_10s_seeds_raw.csv            (RQ1 主表)
  TGM-Lite_实验结果_supervised_amlsim_bank_10s_seeds_raw.csv       (RQ3 跨银行)
  TGM-Lite_实验结果_supervised_amlsim_lr0.05_10s_seeds_raw.csv     (RQ4 5% 标签)
  TGM-Lite_实验结果_supervised_elliptic_10s_seeds_raw.csv          (RQ6 Elliptic)
  TGM-Lite_实验结果_supervised_elliptic_midstream_seeds_raw.csv    (RQ6 中流对照)
  TGM-Lite_实验结果_supervised_elliptic_fz32_midstreamfz32_seeds_raw.csv
  TGM-Lite_实验结果_supervised_amlsim_10s_r4_seeds_raw.csv         (RQ7)
  TGM-Lite_实验结果_supervised_amlsim_fz40_floor10_seeds_raw.csv   (RQ8 地板)
  TGM-Lite_审计结果_supervised_amlsim_ext10_seeds_raw.csv          (RQ9 外部机制)
  TGM-Lite_审计结果_supervised_amlsim_static10_seeds_raw.csv       (RQ9 静态对照)
  TGM-Lite_对抗审计_amlsim_adv10_seeds_raw.csv                     (RQ10 对抗)
  TGM-Lite_实验结果_pu_elliptic_10s_seeds_raw.csv                  (附录A PU)
"""
import argparse, os
import numpy as np
import pandas as pd
from scipy import stats as st

def welch(a, b):
    return st.ttest_ind(a, b, equal_var=False)

def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a)-np.mean(b))/sp * (1 - 3/(4*(na+nb)-9))

def dz(d):
    return np.mean(d)/np.std(d, ddof=1)

def boot_ci(a, b=None, paired=False, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, float)
    if paired:
        d0 = a - np.asarray(b, float)
        idx = rng.integers(0, len(d0), (n, len(d0)))
        return np.percentile(d0[idx].mean(1), [2.5, 97.5])
    b = np.asarray(b, float)
    ia = rng.integers(0, len(a), (n, len(a)))
    ib = rng.integers(0, len(b), (n, len(b)))
    return np.percentile(a[ia].mean(1) - b[ib].mean(1), [2.5, 97.5])

def holm(ps):
    ps = np.asarray(ps); m = len(ps); o = np.argsort(ps)
    adj = np.empty(m); run = 0.0
    for i, idx in enumerate(o):
        run = max(run, (m-i)*ps[idx]); adj[idx] = min(1, run)
    return adj

def bh(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 1.0
    for i, idx in enumerate(o[::-1]):
        run = min(run, ps[idx]*m/(m-i)); adj[idx] = run
    return adj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='.')
    ap.add_argument('--out_dir', default='paper_assets_10s')
    a_ = ap.parse_args()
    rd = lambda name: pd.read_csv(os.path.join(a_.results_dir, name))

    main_ = rd('TGM-Lite_实验结果_supervised_amlsim_10s_seeds_raw.csv')
    bank  = rd('TGM-Lite_实验结果_supervised_amlsim_bank_10s_seeds_raw.csv')
    sca   = rd('TGM-Lite_实验结果_supervised_amlsim_lr0.05_10s_seeds_raw.csv')
    ell   = rd('TGM-Lite_实验结果_supervised_elliptic_10s_seeds_raw.csv')
    mid   = rd('TGM-Lite_实验结果_supervised_elliptic_midstream_seeds_raw.csv')
    midfz = rd('TGM-Lite_实验结果_supervised_elliptic_fz32_midstreamfz32_seeds_raw.csv')
    r4    = rd('TGM-Lite_实验结果_supervised_amlsim_10s_r4_seeds_raw.csv')
    fz    = rd('TGM-Lite_实验结果_supervised_amlsim_fz40_floor10_seeds_raw.csv')
    ext   = rd('TGM-Lite_审计结果_supervised_amlsim_ext10_seeds_raw.csv')
    extst = rd('TGM-Lite_审计结果_supervised_amlsim_static10_seeds_raw.csv')
    adv   = rd('TGM-Lite_对抗审计_amlsim_adv10_seeds_raw.csv')
    pu    = rd('TGM-Lite_实验结果_pu_elliptic_10s_seeds_raw.csv')

    g = lambda d, m: d[d.model == m].reset_index(drop=True)
    M = {m: g(main_, m) for m in main_.model.unique()}
    B = {m: g(bank, m) for m in bank.model.unique()}
    S = {m: g(sca, m) for m in sca.model.unique()}
    E = {m: g(ell, m) for m in ell.model.unique()}
    Md = {m: g(mid, m) for m in mid.model.unique()}
    P = {m: g(pu, m) for m in pu.model.unique()}
    XE = {m: ext[ext.model == m].sort_values('seed') for m in ext.model.unique()}
    XS = {m: extst[extst.model == m].sort_values('seed') for m in extst.model.unique()}

    def acell(model, attack, eps, variant):
        m = ((adv.model == model) & (adv.attack == attack)
             & (adv.eps.round(6) == round(eps, 6)) & (adv.variant == variant))
        return adv[m].sort_values('seed').w3.values

    healthy = [i for i in range(10) if i not in (5, 8)]  # seeds 47/50 崩溃, 其余健康
    T = []
    add = T.append
    add(('RQ1', 'TGM-Lite vs GCN, AP', 'welch', M['tgmlite'].ap, M['gcn'].ap))
    add(('RQ1', 'TGM-Lite vs EvolveGCN, AP', 'welch', M['tgmlite'].ap, M['evolvegcn'].ap))
    add(('RQ1', 'GCN vs EvolveGCN, AP', 'welch', M['gcn'].ap, M['evolvegcn'].ap))
    add(('RQ3', 'TGM-Lite vs GCN, AP (cross-bank)', 'welch', B['tgmlite'].ap, B['gcn'].ap))
    add(('RQ4', 'TGM-Lite vs GCN, AP (5\\% labels)', 'welch', S['tgmlite'].ap, S['gcn'].ap))
    add(('RQ6', 'RF vs GCN, AP (Elliptic)$^{\\ddagger}$', 'welch', E['rf'].ap, E['gcn'].ap))
    add(('RQ6', 'RF vs TGM-Lite, AP (Elliptic)$^{\\ddagger}$', 'welch', E['rf'].ap, E['tgmlite'].ap))
    add(('RQ6', 'TGM-Lite vs GCN, AP (mid-stream)', 'welch', Md['tgmlite'].ap, Md['gcn'].ap))
    add(('RQ6', 'floor vs adaptive, w3 (mid-stream)$^{\\mathsection}$', 'paired',
         g(midfz, 'tgmlite').drift_w3.values, Md['tgmlite'].drift_w3.values))
    add(('RQ7', 'TGM-Lite $r{=}4$ vs GCN, AP', 'welch', r4.ap, M['gcn'].ap))
    add(('RQ8', 'floor vs adaptive, w3 (healthy $n{=}8$)$^{\\mathsection}$', 'paired',
         fz.drift_w3.values[healthy], M['tgmlite'].drift_w3.values[healthy]))
    add(('RQ8', 'floor vs adaptive, AP (healthy $n{=}8$)$^{\\mathsection}$', 'paired',
         fz.ap.values[healthy], M['tgmlite'].ap.values[healthy]))
    add(('RQ9', 'node-state SSM vs its static trunk, w3', 'welch',
         XE['graphmamba'].drift_w3, XS['graphmamba_static'].drift_w3))
    add(('RQ9', 'node-state SSM vs its static trunk, AP', 'welch',
         XE['graphmamba'].ap, XS['graphmamba_static'].ap))
    add(('RQ9', 'DySAT vs its static trunk, AP', 'welch', XE['dysat'].ap, XS['dysat_static'].ap))
    add(('RQ10', 'floor (cert.) vs floor (inf.), clean w3$^{\\mathsection}$', 'paired',
         acell('tgmlite', 'clean', 0, 'floor_fz'), acell('tgmlite', 'clean', 0, 'floor')))
    add(('RQ10', 'floor (cert.) vs adaptive, evasion $\\varepsilon{=}0.1$ w3$^{\\mathsection}$', 'paired',
         acell('tgmlite', 'evasion', 0.1, 'floor_fz'), acell('tgmlite', 'evasion', 0.1, 'adaptive')))
    add(('RQ10', 'adaptive vs GCN, evasion $\\varepsilon{=}0.05$ w3$^{\\mathsection}$', 'paired',
         acell('tgmlite', 'evasion', 0.05, 'adaptive'), acell('gcn', 'evasion', 0.05, 'adaptive')))
    add(('RQ10', 'adaptive, drift $\\varepsilon{=}0.1$ vs clean, w3$^{\\mathsection}$', 'paired',
         acell('tgmlite', 'drift', 0.1, 'adaptive'), acell('tgmlite', 'clean', 0, 'adaptive')))
    add(('App.A', 'TGM-Lite nnPU vs naive, AP (Welch)', 'welch', P['tgmlite_pu'].ap, P['tgmlite_naive'].ap))

    rows = []
    for fam, lab, kind, x, y in T:
        x = np.asarray(x, float); y = np.asarray(y, float)
        if kind == 'welch':
            t, p = welch(x, y); eff = hedges_g(x, y); ci = boot_ci(x, y); d = x.mean()-y.mean()
        else:
            t, p = st.ttest_rel(x, y); eff = dz(x-y); ci = boot_ci(x, y, paired=True); d = (x-y).mean()
        rows.append(dict(family=fam, label=lab, kind=kind, delta=d, eff=eff, p=p,
                         ci_lo=ci[0], ci_hi=ci[1]))
    df = pd.DataFrame(rows)
    df['p_holm'] = np.nan
    for fam, gg in df.groupby('family'):
        df.loc[gg.index, 'p_holm'] = holm(gg.p.values)
    df['p_bh'] = bh(df.p.values)
    order = ['RQ1', 'RQ3', 'RQ4', 'RQ6', 'RQ7', 'RQ8', 'RQ9', 'RQ10', 'App.A']
    df['fo'] = df.family.map({f: i for i, f in enumerate(order)})
    df = df.sort_values(['fo', 'p']).drop(columns='fo').reset_index(drop=True)

    def fp(v): return '$<10^{-3}$' if v < 0.001 else f'{v:.4f}'
    lines = [r"% Multiple-comparison control battery (auto-generated by stats_battery.py)",
             r"\begin{tabular}{l l c c c c c}", r"\toprule",
             r"RQ & comparison & $\Delta$ [95\% CI] & ES & $p$ & $p_{\mathrm{Holm}}$ & $q_{\mathrm{BH}}$ \\",
             r"\midrule"]
    for _, r0 in df.iterrows():
        dci = f"${r0.delta:+.4f}$\\ $[{r0.ci_lo:.4f}, {r0.ci_hi:.4f}]$"
        es = (f"$g={r0.eff:+.2f}$" if r0.kind == 'welch' else f"$d_z={r0.eff:+.2f}$")
        star = '$\\checkmark$' if r0.p_bh < 0.05 else '---'
        lines.append(f"{r0.family} & {r0.label} & {dci} & {es} & {fp(r0.p)} & "
                     f"{fp(r0.p_holm)} & {r0.p_bh:.4f}\\ {star} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    os.makedirs(os.path.join(a_.out_dir, 'tables'), exist_ok=True)
    out = os.path.join(a_.out_dir, 'tables', 'tab_stats.tex')
    open(out, 'w', encoding='utf-8').write("\n".join(lines) + "\n")
    print('[表]', out, f'({len(df)} tests)')
    sig = df[(df.p < 0.05) & (df.p_holm >= 0.05)]
    print('Holm 后转为边际的检验:', sig.label.tolist() if len(sig) else '无')

if __name__ == '__main__':
    main()
