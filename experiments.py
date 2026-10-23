# ==============================================================================
# TGM-Lite: A Drift-Gated Selective State-Space Temporal Graph Network
#           for Stable Money Laundering Detection under Concept Drift
#
# Official experiment script accompanying the IEEE TDSC submission.
# Author: Xiaole Han (Business School, Qilu Institute of Technology, Jinan, China)
# Contact: hanxiaoleop@163.com | ORCID: 0009-0009-9611-2368
# License: MIT
#
# Single-file pipeline (pure PyTorch + NumPy, no PyG/DGL; runs on a 6 GB GPU or
# CPU): dataset loading -> windowing -> graph construction -> training ->
# evaluation, with all baselines and ablations from the paper built in.
#
# Quick start:
#   python experiments.py                          # synthetic smoke test (~1 min)
#   python experiments.py --data elliptic          # Elliptic benchmark
#   python experiments.py --data amlsim            # IBM AMLSim HI-Small
#
# To reproduce every paper table see README.md ("Reproducing the paper tables").
# The inline documentation below is in Chinese; the README provides the English
# reference (arguments, protocols, metrics).
# ==============================================================================

# -*- coding: utf-8 -*-
"""
================================================================================
TGM-Lite: 面向链上金融风险检测的"漂移门控选择性状态空间"时序图小模型
--------------------------------------------------------------------------------
单文件可跑通实验脚本 | 纯 PyTorch + NumPy 实现（无需 PyG/DGL，Windows 友好）
适配硬件: RTX 2060 6GB 显存 或 纯 CPU（12 核 Intel）

模型定位（论文故事）:
  - 极小参数（<100K）时序图网络，用于链上（比特币/以太坊）非法交易检测
  - 核心创新: 用"选择性状态空间模型(Selective SSM, Mamba 思想)"在低秩空间中
    演化 GCN 权重，替代 EvolveGCN 的 GRU 演化，线性复杂度、可外推到未来时间片
  - 漂移门控: 以每个时间片的图统计上下文（均值/方差/密度/时间编码）驱动
    SSM 的门控，使模型对"诈骗模式漂移"自适应
  - 自监督预训练: 掩码特征重建，利用 Elliptic 中 ~78% 无标签节点

内置对比/消融模型（--model all 一键全跑，直接生成论文对比表）:
  mlp            纯特征 MLP（无图、无时间）
  gcn            静态权重 GCN（无时间演化）
  evolvegcn      低秩 GRU 权重演化（EvolveGCN 思想的轻量实现基线）
  tgmlite_nograph  TGM-Lite 去掉图传播（消融: 图的贡献）
  tgmlite_nossm    TGM-Lite 去掉 SSM 时序记忆（消融: 状态记忆的贡献）
  tgmlite        完整模型（本文方法）

数据:
  --data synthetic   内置合成时序交易图（含渐变式模式漂移），1 分钟验证管线
  --data elliptic    Elliptic 比特币数据集（kagglehub 免登录自动下载，
                     或用 --elliptic_dir 指向本地解压目录）
  --data amlsim      IBM AML-Data HI-Small（5M 笔真实风格银行交易, 逐笔洗钱标签,
                     交易级节点 + 同账户资金链条; kagglehub 免登录自动下载,
                     或用 --amlsim_dir 指向本地目录/文件）

用法示例:
  python TGM-Lite_实验脚本.py                          # 默认: 合成数据全模型对比
  python TGM-Lite_实验脚本.py --data elliptic          # Elliptic 全模型对比
  python TGM-Lite_实验脚本.py --data amlsim            # AMLSim HI-Small 全模型对比
  python TGM-Lite_实验脚本.py --data amlsim --task pu  # AMLSim PU 学习
  python TGM-Lite_实验脚本.py --data amlsim --split_mode bank  # 跨银行泛化(更难的新协议)
  python TGM-Lite_实验脚本.py --data amlsim --edge_fan 0       # 消融: 关闭同窗口团边
================================================================================
"""

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # 激活检查点: 训练回路显存封顶, 数值完全等价


# ==============================================================================
# 0. 工具函数: 随机种子 / 指标（不依赖 sklearn）
# ==============================================================================

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _avg_ranks(s: np.ndarray) -> np.ndarray:
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=np.float64)
    sv = s[order]
    i, n = 0, len(s)
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _avg_ranks(np.asarray(s, dtype=np.float64))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def average_precision(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y).astype(int)
    if y.sum() == 0:
        return float("nan")
    order = np.argsort(-np.asarray(s, dtype=np.float64))
    ys = y[order]
    tp = np.cumsum(ys)
    prec = tp / np.arange(1, len(ys) + 1)
    return float((prec * ys).sum() / ys.sum())


def best_f1_threshold(y: np.ndarray, s: np.ndarray, n_grid: int = 200):
    if len(s) == 0:
        return 0.0, float("nan")
    qs = np.unique(np.quantile(s, np.linspace(0, 1, n_grid)))
    best_t, best_f1 = 0.5, -1.0
    for t in qs:
        pred = (s >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def prf_at(y: np.ndarray, s: np.ndarray, thr: float):
    pred = (s >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    return p, r, f1


def precision_at_k(y: np.ndarray, s: np.ndarray, k: int = 100) -> float:
    if len(s) == 0:
        return float("nan")
    k = min(k, len(s))
    top = np.argsort(-s)[:k]
    return float(y[top].mean())


# ==============================================================================
# 1. 数据集: 合成时序交易图（内置漂移） & Elliptic 加载器
# ==============================================================================

def make_synthetic(T=49, n_per_t=120, feat_dim=166, drift_at=35, seed=0):
    """合成时序交易图: 非法行为模式随时间渐变漂移（模式A"爆发式分散转移"
    权重递减、模式B"多跳混币"权重递增），叠加一个全程稳定的弱信号。
    用于验证: (1) 管线可学习; (2) 时序自适应模型在漂移后窗口的鲁棒性。"""
    rng = np.random.default_rng(seed)
    N = T * n_per_t
    node_time = np.repeat(np.arange(1, T + 1), n_per_t)
    X = rng.normal(0.0, 0.6, size=(N, feat_dim)).astype(np.float32)
    y = np.zeros(N, dtype=np.int64)

    for t in range(1, T + 1):
        idx = np.where(node_time == t)[0]
        n_ill = max(3, int(0.05 * len(idx)))
        ill = rng.choice(idx, size=n_ill, replace=False)
        y[ill] = 1
        alpha = (t - 1) / (T - 1)                # 漂移进度: 0 -> 1
        wA, wB = 1.0 - alpha, alpha
        for i in ill:
            hard = rng.random() < 0.25          # 25% 难例（弱信号）
            scale = 0.6 if hard else 1.0
            # 全程稳定弱信号（所有模型都可学习, 保证 sanity）
            X[i, 60] += 1.4 * scale
            X[i, 90] -= 1.1 * scale
            # 模式A（早期主导）: 爆发式 fan-out（快进快出）
            X[i, 0] += 2.4 * scale * wA           # 出度/输入数异常
            X[i, 3] += 2.0 * scale * wA           # 唯一对手方数量
            X[i, 7] -= 1.8 * scale * wA           # 平均金额小
            X[i, 21] += 1.4 * scale * wA
            # 模式B（后期主导）: 多跳混币长链
            X[i, 10] += 2.4 * scale * wB          # 跳数/链长特征
            X[i, 14] += 2.0 * scale * wB
            X[i, 30] += 1.7 * scale * wB
            X[i, 45] -= 1.3 * scale * wB

    # 边: 快照内随机边 + 非法节点的跨时间洗钱链
    src_l, dst_l = [], []
    for t in range(1, T + 1):
        idx = np.where(node_time == t)[0]
        m = int(0.5 * len(idx))
        s = rng.choice(idx, size=m)
        d = rng.choice(idx, size=m)
        keep = s != d
        src_l.append(s[keep]); dst_l.append(d[keep])
        if t < T:
            nxt = np.where(node_time == t + 1)[0]
            ill_next = nxt[y[nxt] == 1]
            ill_now = idx[y[idx] == 1]
            for u in ill_now:
                k = rng.integers(3, 7)
                # 洗钱链倾向流向下一时间片的非法节点(同配时序结构), 少量随机
                if len(ill_next) > 0 and rng.random() < 0.8:
                    dst_l.append(rng.choice(ill_next, size=k))
                else:
                    dst_l.append(rng.choice(nxt, size=k))
                src_l.append(np.full(k, u))
    edge_src = np.concatenate(src_l)
    edge_dst = np.concatenate(dst_l)

    # 模拟 Elliptic 的标签稀缺: 55% 合法 / 30% 非法 设为无标签
    labeled = np.ones(N, dtype=bool)
    lic_idx = np.where(y == 0)[0]
    ill_idx = np.where(y == 1)[0]
    labeled[rng.choice(lic_idx, int(0.55 * len(lic_idx)), replace=False)] = False
    labeled[rng.choice(ill_idx, int(0.30 * len(ill_idx)), replace=False)] = False

    return dict(X=X, y=y, labeled=labeled, node_time=node_time,
                edge_src=edge_src, edge_dst=edge_dst, T=T, name="synthetic")


ELLIPTIC_FILES = ("elliptic_txs_features.csv",
                  "elliptic_txs_classes.csv",
                  "elliptic_txs_edgelist.csv")

# 本地数据默认目录(存在则优先使用, 不存在则自动下载到该目录)
ELLIPTIC_DEFAULT_DIR = r"D:\TGM-Lite\Data\elliptic"
AMLSIM_DEFAULT_DIR = r"D:\TGM-Lite\Data\ealtman2019"


def _find_elliptic_files(root):
    """在 root 下递归定位 Elliptic 三个 csv, 找到返回 {文件名: 完整路径}。"""
    hits = {}
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f in ELLIPTIC_FILES and f not in hits:
                hits[f] = os.path.join(dirpath, f)
    return hits if len(hits) == 3 else None


def ensure_elliptic(folder):
    """解析 Elliptic 数据文件位置。优先本地目录; 否则用 kagglehub 免登录下载。
    若免登录失败(Kaggle 策略变动等), 打印手动下载指引并退出。"""
    cands = []
    if folder and folder != "auto":
        cands.append(folder)
    cands.append(ELLIPTIC_DEFAULT_DIR)            # 默认本地目录优先
    for cand in cands:
        if cand and os.path.isdir(cand):
            found = _find_elliptic_files(cand)
            if found:
                print(f"[数据] 使用本地目录: {cand}")
                return found
            print(f"[数据] {cand} 未找到完整三文件, 继续尝试其他途径 ...")

    try:
        os.makedirs(ELLIPTIC_DEFAULT_DIR, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = ELLIPTIC_DEFAULT_DIR  # 下载缓存到数据目录
    except OSError:
        pass
    try:
        import kagglehub
    except ImportError:
        print("[数据] 安装 kagglehub ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub

    print("[数据] 通过 kagglehub 下载 Elliptic(公开数据集, 免账号) ...")
    try:
        path = kagglehub.dataset_download("ellipticco/elliptic-data-set")
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "\n[下载失败] kagglehub 免登录下载未成功:\n  " + str(e) + "\n"
            "备用方案(任选其一):\n"
            "  1) 注册免费 Kaggle 账号后, 打开 "
            "https://www.kaggle.com/datasets/ellipticco/elliptic-data-set 手动下载解压;\n"
            "  2) Elliptic 公司官网申请下载;\n"
            "然后用 --elliptic_dir 指向解压目录(内含三个 elliptic_txs_*.csv)即可。")
    print(f"[数据] 下载完成: {path}")
    found = _find_elliptic_files(path)
    if not found:
        raise SystemExit(f"[错误] 下载目录 {path} 中未找到 elliptic_txs_*.csv 三文件,"
                         f" 请检查后用 --elliptic_dir 手动指定。")
    return found


def _find_file(root, fname):
    """在 root 下递归定位指定文件名, 找到返回完整路径。"""
    if not root or not os.path.isdir(root):
        return None
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f == fname:
                return os.path.join(dirpath, f)
    return None


def _extract_single_from_archive(archive_path, member_name, out_dir):
    """从 zip 压缩包中只解压指定成员(避免 7.6GB 整包解压撑爆磁盘)。"""
    import zipfile
    out_path = os.path.join(out_dir, member_name)
    with zipfile.ZipFile(archive_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(member_name)]
        if not names:
            return None
        with zf.open(names[0]) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, 1024 * 1024)
    return out_path


def _find_archive(root):
    """在 root 下递归寻找 kagglehub 下载残留/缓存的 .archive 压缩包。"""
    if not root or not os.path.isdir(root):
        return None
    for dirpath, _, files in os.walk(root):
        for f in files:
            if ".archive" in f:
                return os.path.join(dirpath, f)
    return None


def ensure_amlsim(folder):
    """解析 AMLSim 交易 CSV 位置(默认 HI-Small_Trans.csv)。--amlsim_dir 指向尚不存在的
    具体 csv 路径(如 ...\\HI-Medium_Trans.csv)时, 自动按该文件名获取。优先本地 csv;
    其次已下载的压缩包(单文件解压, 避免 7.6GB 整包解压撑爆磁盘); 最后 kagglehub 免登录
    下载(先尝试单文件下载, 失败再整包下载并自动降级单文件提取)。"""
    handle = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
    if folder and folder != "auto" and os.path.isfile(folder):
        print(f"[数据] 使用本地文件: {folder}")
        return folder
    target = "HI-Small_Trans.csv"
    if (folder and folder != "auto" and not os.path.isdir(folder)
            and folder.lower().endswith(".csv")):
        target = os.path.basename(folder)
        print(f"[数据] 目标文件 {target} 尚不存在, 将自动获取 ...")
    cands = []
    if folder and folder != "auto" and os.path.isdir(folder):
        cands.append(folder)
    cands.append(AMLSIM_DEFAULT_DIR)
    for cand in cands:
        hit = _find_file(cand, target)
        if hit:
            print(f"[数据] 使用本地文件: {hit}")
            return hit

    try:
        os.makedirs(AMLSIM_DEFAULT_DIR, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = AMLSIM_DEFAULT_DIR  # 下载缓存到数据目录
    except OSError:
        pass

    # 1) 已有下载好的压缩包(上次下载完成但解压中断的残留) -> 直接单文件解压, 不重复下载
    for cand in cands:
        arch = _find_archive(cand)
        if arch:
            free_gb = shutil.disk_usage(AMLSIM_DEFAULT_DIR).free / 2**30
            print(f"[数据] 发现已下载的压缩包: {arch} (磁盘剩余 {free_gb:.1f}GB)")
            if free_gb < 2.0:
                raise SystemExit(
                    f"[磁盘不足] 仅剩 {free_gb:.1f}GB。请先清理: 删除 "
                    f"{os.path.join(AMLSIM_DEFAULT_DIR, 'datasets')} 下 versions 文件夹里"
                    f" 已部分解压的文件(保留 8.archive 压缩包), 释放至少 2GB 后重跑。")
            try:
                hit = _extract_single_from_archive(arch, target,
                                                   AMLSIM_DEFAULT_DIR)
                if hit:
                    print(f"[数据] 已从压缩包单独提取: {hit}")
                    print(f"[提示] 现在可以手动删除 {arch} 释放 7.6GB 空间")
                    return hit
            except Exception as e:  # noqa: BLE001
                print(f"[警告] 压缩包单文件解压失败({e}), 转用下载流程")

    # 2) kagglehub 下载(整包解压若爆盘, 自动降级为从压缩包单文件提取)
    try:
        import kagglehub
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub
    print(f"[数据] 通过 kagglehub 下载 IBM AML-Data(公开数据集, 免账号),"
          f" 目标文件 {target}, 保存到 {AMLSIM_DEFAULT_DIR} ...")
    try:
        hit = kagglehub.dataset_download(handle, path=target)  # 单文件下载(远小于整包 7.6GB)
        if hit and os.path.isfile(hit):
            print(f"[数据] 单文件下载完成: {hit}")
            return hit
    except Exception as e:  # noqa: BLE001
        print(f"[提示] 单文件下载未成功({e}), 转为整包下载(7.6GB) ...")
    root = None
    try:
        root = kagglehub.dataset_download(handle)
    except OSError as e:
        print(f"[警告] 整包下载/解压中断({e}), 尝试从压缩包单独提取 ...")
    if root:
        hit = _find_file(root, target)
        if hit:
            return hit
    arch = _find_archive(AMLSIM_DEFAULT_DIR)
    if arch:
        hit = _extract_single_from_archive(arch, target,
                                           AMLSIM_DEFAULT_DIR)
        if hit:
            print(f"[数据] 已从压缩包单独提取: {hit}")
            print(f"[提示] 现在可以手动删除 {arch} 释放 7.6GB 空间")
            return hit
    raise SystemExit(f"[错误] 未能获取 {target}。请检查磁盘空间后重试,"
                     " 或手动下载后用 --amlsim_dir 指向该文件。")


def load_amlsim(folder, window_hours=4, max_nodes=300000, seed=0, edge_fan=True,
                window_mode="quantile", n_windows=60):
    """IBM AML-Data (HI-Small): 交易级节点 + 同账户时序链边 + 同窗口团边。
    节点=交易(带时间戳与 Is Laundering 标签)。
    边: (a) 同一账户的相邻交易链(资金流向链);
        (b) edge_fan=True 时, 同账户同时间片的交易两两连边(fan-in/fan-out 汇集结构)。
    原始特征无预聚合邻居信息 —— 图结构是唯一的邻居信息来源。
    全量 5M 笔交易; 为适配 6GB 显存, 默认分层子采样(保留全部洗钱交易)至 max_nodes。"""
    import pandas as pd
    path = ensure_amlsim(folder)
    print(f"[数据] 读取 {path} ...")
    df = pd.read_csv(path)
    print(f"[数据] 原始交易 {len(df):,} 笔, 列: {list(df.columns)}")

    # ---- 时间与标签 ----
    ts = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")
    lab = df["Is Laundering"].astype(str).str.strip().str.lower()
    y = lab.isin(["yes", "1", "true"]).values

    # ---- 分层子采样: 保留全部洗钱交易 + 等比抽样合法交易(控制 N, 防显存溢出) ----
    if max_nodes and len(df) > max_nodes:
        rng = np.random.default_rng(seed)
        pos = np.where(y)[0]
        n_neg = max_nodes - len(pos)
        neg = rng.choice(np.where(~y)[0], size=min(n_neg, int((~y).sum())),
                         replace=False)
        keep = np.sort(np.concatenate([pos, neg]))
        df = df.iloc[keep].reset_index(drop=True)
        ts = ts.iloc[keep].reset_index(drop=True)
        y = y[keep]
        print(f"[数据] 子采样: {len(keep):,} 笔 (洗钱 {y.sum():,} 全部保留, "
              f"占比 {y.mean()*100:.2f}%)")

    t0 = ts.min()
    if window_mode == "quantile":
        # 等频分箱: 每个窗口交易量相近, 抗离群时间戳导致的尾部空窗
        n_win = max(2, n_windows)
        secs = (ts - t0).dt.total_seconds().values
        qs = np.quantile(secs, np.linspace(0, 1, n_win + 1)[1:-1])
        win = np.searchsorted(qs, secs, side="right").astype(np.int64) + 1
    else:
        win = ((ts - t0).dt.total_seconds() // (window_hours * 3600)).values.astype(np.int64) + 1
    T = int(win.max())
    labeled = np.ones(len(df), dtype=bool)          # 该数据集逐笔有标签

    # ---- 节点特征(仅原始属性, ~20 维) ----
    amt = np.log1p(df["Amount Received"].astype(float).values)
    hour = ts.dt.hour.values + ts.dt.minute.values / 60.0
    same_bank = (df["From Bank"].values == df["To Bank"].values).astype(float)
    same_curr = (df["Receiving Currency"].astype(str).values
                 == df["Payment Currency"].astype(str).values).astype(float)
    pay_dum = pd.get_dummies(df["Payment Format"].astype(str))
    top_cur = df["Receiving Currency"].astype(str).value_counts().head(8).index
    cur_dum = pd.get_dummies(
        df["Receiving Currency"].astype(str).where(
            df["Receiving Currency"].astype(str).isin(top_cur), "OTHER"))
    X = np.column_stack([
        amt, np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        same_bank, same_curr,
        pay_dum.values.astype(np.float32),
        cur_dum.values.astype(np.float32)]).astype(np.float32)

    # ---- 边: 同一账户的相邻交易链(付款方与收款方两侧) ----
    # pandas 对重复列名自动加后缀("Account" 出现两次 -> "Account" 与 "Account.1")
    acc_cols = [c for c in df.columns if str(c).startswith("Account")]
    a_from = df[acc_cols[0]].values
    a_to = df[acc_cols[1]].values if len(acc_cols) > 1 else a_from

    src_l, dst_l = [], []
    ts_v = ts.values
    for acc_arr in (a_from, a_to):
        sub = pd.DataFrame({"acc": acc_arr, "t": ts_v, "idx": np.arange(len(df))})
        sub = sub.sort_values(["acc", "t"], kind="mergesort")
        acc = sub["acc"].values
        idx = sub["idx"].values
        same = acc[1:] == acc[:-1]
        s, d = idx[:-1][same], idx[1:][same]
        keep_e = win[s] <= win[d]                  # 时间向前的资金链
        src_l.append(s[keep_e]); dst_l.append(d[keep_e])

    if edge_fan:
        # 同账户、同时间片的交易两两连边: 捕获 fan-in(汇集)/fan-out(分散)结构
        for acc_arr in (a_from, a_to):
            grp = pd.DataFrame({"acc": acc_arr, "w": win, "idx": np.arange(len(df))})
            for _, gdf in grp.groupby(["acc", "w"], sort=False):
                ii = gdf["idx"].values
                k = len(ii)
                if k < 2:
                    continue
                if k <= 20:                        # 小团: 全连接
                    a, b = np.triu_indices(k, k=1)
                    src_l.append(ii[a]); dst_l.append(ii[b])
                else:                              # 大团: 星型(以最早交易为中心), 防边爆炸
                    src_l.append(np.full(k - 1, ii[0])); dst_l.append(ii[1:])

    edge_src = np.concatenate(src_l).astype(np.int64)
    edge_dst = np.concatenate(dst_l).astype(np.int64)

    # ---- 银行分组(跨银行泛化协议用) ----
    bank = df["From Bank"].values.astype(np.int64)

    bc = np.bincount(win)[1:]                   # win 从 1 开始
    print(f"[数据] AMLSim {os.path.basename(path)}: N={len(df):,}, E={len(edge_src):,}, T={T}, "
          f"特征 {X.shape[1]} 维, 洗钱 {y.mean()*100:.2f}% "
          f"(分箱={window_mode}, 每窗节点 {int(bc.min()):,}~{int(bc.max()):,}, "
          f"团边={'开' if edge_fan else '关'})")
    return dict(X=X, y=y.astype(np.int64), labeled=labeled,
                node_time=win.astype(np.int64), bank=bank,
                edge_src=edge_src, edge_dst=edge_dst, T=T, name="amlsim")


def load_elliptic(files):
    """加载 Elliptic 数据集（标准三文件格式）。"""
    f_feat = files["elliptic_txs_features.csv"]
    f_cls = files["elliptic_txs_classes.csv"]
    f_edge = files["elliptic_txs_edgelist.csv"]

    print("[数据] 读取特征文件 ...")
    raw = np.loadtxt(f_feat, delimiter=",", dtype=np.float64)
    tx_ids = raw[:, 0].astype(np.int64)
    node_time = raw[:, 1].astype(np.int64)
    X = raw[:, 2:].astype(np.float32)              # 165 维特征
    X = np.concatenate([X, np.zeros((len(X), 1), np.float32)], 1)  # 对齐 166 维
    id2idx = {tid: i for i, tid in enumerate(tx_ids)}
    N = len(tx_ids)
    T = int(node_time.max())

    print("[数据] 读取标签文件 ...")
    y = np.full(N, -1, dtype=np.int64)
    with open(f_cls) as f:
        rdr = csv.reader(f)
        next(rdr, None)                       # 跳过表头(若有, 数据行会被 try 保留)
        for row in rdr:
            if len(row) < 2:
                continue
            try:
                i = id2idx.get(int(row[0]))   # 字符串 id -> 整数索引
            except ValueError:
                continue                      # 兼容带表头的文件
            if i is None:
                continue
            if row[1] == "1":
                y[i] = 1
            elif row[1] == "2":
                y[i] = 0
    labeled = y >= 0
    y = np.clip(y, 0, 1)
    if labeled.sum() == 0:
        raise SystemExit("[错误] 未解析到任何标签, 请检查 elliptic_txs_classes.csv 格式。")

    print("[数据] 读取边文件 ...")
    src_l, dst_l = [], []
    with open(f_edge) as f:
        rdr = csv.reader(f)
        next(rdr, None)
        for row in rdr:
            try:
                a, b = id2idx.get(int(row[0])), id2idx.get(int(row[1]))
            except (ValueError, IndexError):
                continue                      # 兼容带表头的文件
            if a is not None and b is not None and a != b:
                src_l.append(a); dst_l.append(b)
    edge_src = np.array(src_l, dtype=np.int64)
    edge_dst = np.array(dst_l, dtype=np.int64)

    print(f"[数据] Elliptic: N={N}, E={len(edge_src)}, T={T}, "
          f"已标注={labeled.sum()} (非法 {y[labeled].mean()*100:.1f}%)")
    return dict(X=X, y=y, labeled=labeled, node_time=node_time,
                edge_src=edge_src, edge_dst=edge_dst, T=T, name="elliptic")


def load_custom(path):
    """加载 data_adapter/paysim_adapter 生成的通用时序图 npz (E3 真实数据通道)。
    必须与 load_elliptic 返回同一字典结构:
    X(N,F float32) / y(N int64, 0/1) / labeled(bool) / node_time(int64, 1..T) /
    edge_src / edge_dst / T / name。"""
    z = np.load(path, allow_pickle=False)
    need = ["X", "y", "labeled", "node_time", "edge_src", "edge_dst", "T"]
    miss = [k for k in need if k not in z]
    if miss:
        raise SystemExit(f"[错误] {path} 缺少字段 {miss} —— 请用 paysim_adapter.py 生成。")
    data = dict(X=z["X"].astype(np.float32), y=z["y"].astype(np.int64),
                labeled=z["labeled"].astype(bool),
                node_time=z["node_time"].astype(np.int64),
                edge_src=z["edge_src"].astype(np.int64),
                edge_dst=z["edge_dst"].astype(np.int64),
                T=int(z["T"]),
                name=str(z["name"]) if "name" in z else "custom")
    print(f"[数据] custom({data['name']}): N={len(data['y'])}, "
          f"E={len(data['edge_src'])}, T={data['T']}, "
          f"已标注={data['labeled'].sum()} "
          f"(非法 {data['y'][data['labeled']].mean()*100:.2f}%)")
    return data


# ==============================================================================
# 2. 图结构预处理: 按时间片切分 + 归一化邻接(含自环, 对称化)
# ==============================================================================

def build_snapshots(data, device):
    N, T = data["X"].shape[0], data["T"]
    t_src = data["node_time"][data["edge_src"]]    # 边归属到源节点时间片
    Ahats, snap_nodes = [], {}
    arangeN = torch.arange(N, device=device)
    for t in range(1, T + 1):
        sel = np.where(t_src == t)[0]
        s = torch.from_numpy(data["edge_src"][sel]).to(device)
        d = torch.from_numpy(data["edge_dst"][sel]).to(device)
        row = torch.cat([s, d, arangeN])           # 对称化 + 全节点自环
        col = torch.cat([d, s, arangeN])
        deg = torch.bincount(torch.cat([s, d]), minlength=N).float()
        dinv = (deg + 1.0).pow(-0.5)
        val = dinv[row] * dinv[col]
        A = torch.sparse_coo_tensor(torch.stack([row, col]), val, (N, N)).coalesce()
        Ahats.append(A)
        snap_nodes[t] = np.where(data["node_time"] == t)[0]
    return Ahats, snap_nodes


def build_context(Z, data, device):
    """每个时间片的图统计上下文 c_t = [均值, 方差, sin/cos 时间编码, log密度]"""
    T, d = data["T"], Z.shape[1]
    ctx = torch.zeros(T + 1, 2 * d + 3, device=device)
    for t in range(1, T + 1):
        mask = torch.from_numpy(data["node_time"] == t).to(device)
        Zt = Z[mask]
        mu = Zt.mean(0)
        sd = Zt.var(0, unbiased=False).sqrt()
        te = torch.tensor([math.sin(2 * math.pi * t / T),
                           math.cos(2 * math.pi * t / T),
                           math.log1p(float(mask.sum())) / 12.0], device=device)
        ctx[t] = torch.cat([mu, sd, te])
    return ctx.detach()


# ==============================================================================
# 3. 模型: 时序控制器(静态/无记忆/GRU/选择性SSM) + 低秩动态权重 GCN
# ==============================================================================

class SSMController(nn.Module):
    """选择性状态空间控制器(Mamba 核心思想的轻量实现):
    h_t = σ(W_a c_t)⊙h_{t-1} + σ(W_b c_t)⊙tanh(W_in c_t); s_t = tanh(W_out h_t)。
    O(T) 线性复杂度。

    漂移护栏 (drift-guard): 10-seed 压力测试发现约 20% 初始化下, 控制器在
    测试尾段(上下文分布走出训练包络)产生反向适应(mal-adaptation), recall
    崩溃至 ~0.05, 而训练/验证曲线完全健康(确诊为测试期开环失控, 而非优化
    失败或状态发散——LayerNorm 对照实验不能消除该现象)。
    护栏在训练期累计上下文 c_t 的一阶/二阶统计与最大归一化偏差 d_max
    (仅用训练窗口, 无测试信息泄漏); 推理期当某窗口上下文的归一化最大偏差
    d_t 超出 τ·d_max 时, 门 g=σ(κ(τ·d_max−d_t)) 将调制 s_t 平滑退火到 0,
    模型退化为静态 W₀, 把最坏情况钉在"静态 GNN 地板"上。训练期门恒不生效,
    因此训练动力学与无护栏版完全一致。
    --guard off 关闭(消融用); --state_norm ln 为另一独立稳定器选项。"""

    def __init__(self, ctx_dim, m, r, state_norm=False, guard=True,
                 guard_tau=1.5, guard_kappa=4.0):
        super().__init__()
        self.inp = nn.Linear(ctx_dim, m)
        self.gate_a = nn.Linear(ctx_dim, m)
        self.gate_b = nn.Linear(ctx_dim, m)
        self.out = nn.Linear(m, r)
        self.norm = nn.LayerNorm(m) if state_norm else nn.Identity()
        self.m = m
        # ---- 漂移护栏状态(缓冲区, 非参数, 不计入 #params) ----
        self.guard = guard
        self.guard_tau = float(guard_tau)      # 包络缓冲倍数 τ
        self.guard_kappa = float(guard_kappa)  # 过渡陡度 κ
        self.register_buffer("c_mu", torch.zeros(ctx_dim))
        self.register_buffer("c_m2", torch.zeros(ctx_dim))
        self.register_buffer("c_n", torch.tensor(0.0))
        self.register_buffer("d_max", torch.tensor(1.0))
        self.warmup = 12                       # 统计预热步数(首 chunk 不计包络)
        self.last_gate = 1.0                   # 供诊断记录
        self.gate_trace = {}                   # eval 期 {时间片: 门值}

    def init_state(self, device):
        return torch.zeros(self.m, device=device)

    def _guard_update(self, c):
        """训练期: 在线更新上下文均值/二阶矩与包络半径 d_max(无梯度)。"""
        with torch.no_grad():
            self.c_n += 1.0
            delta = c - self.c_mu
            self.c_mu += delta / self.c_n
            self.c_m2 += delta * (c - self.c_mu)
            if self.c_n >= self.warmup:
                var = self.c_m2 / self.c_n
                d = ((c - self.c_mu) / var.clamp_min(1e-8).sqrt()).abs().max()
                self.d_max = torch.maximum(self.d_max, d)

    def _guard_gate(self, c):
        """推理期: 门值 g∈(0,1]; 上下文在训练包络内≈1, 出包络平滑→0。"""
        with torch.no_grad():
            if self.c_n < self.warmup:
                self.last_gate = 1.0
                return 1.0                 # 统计未就绪: 直通不门控(防御默认)
            var = self.c_m2 / self.c_n
            d = ((c - self.c_mu) / var.clamp_min(1e-8).sqrt()).abs().max()
            g = torch.sigmoid(self.guard_kappa * (self.guard_tau * self.d_max - d))
        self.last_gate = float(g)
        return g

    def step(self, c, h):
        a = torch.sigmoid(self.gate_a(c))
        b = torch.sigmoid(self.gate_b(c))
        h = self.norm(a * h + b * torch.tanh(self.inp(c)))
        s = torch.tanh(self.out(h))
        if self.guard:
            if self.training:
                self._guard_update(c)      # 训练期只记账、不门控
            else:
                s = s * self._guard_gate(c)
        return s, h


class GRUController(nn.Module):
    """EvolveGCN 式基线: GRU 在低秩调制向量空间上演化。"""

    def __init__(self, ctx_dim, r):
        super().__init__()
        self.gru = nn.GRUCell(ctx_dim, r)
        self.r = r

    def init_state(self, device):
        return torch.zeros(self.r, device=device)

    def step(self, c, h):
        h = self.gru(c.unsqueeze(0), h.unsqueeze(0)).squeeze(0)
        return torch.tanh(h), h


class NoMemController(nn.Module):
    """消融: 有上下文调制、无时序记忆。"""

    def __init__(self, ctx_dim, r):
        super().__init__()
        self.lin = nn.Linear(ctx_dim, r)

    def init_state(self, device):
        return None

    def step(self, c, h):
        return torch.tanh(self.lin(c)), None


class StaticController(nn.Module):
    """消融: 静态权重。"""

    def __init__(self, r):
        super().__init__()
        self.r = r

    def init_state(self, device):
        return None

    def step(self, c, h):
        return None, None


class RiskNet(nn.Module):
    """统一骨架: proj -> [动态权重 GCN x 2] -> clf
    mode: mlp / gcn / evolvegcn / tgmlite_nograph / tgmlite_nossm / tgmlite"""

    def __init__(self, feat_dim, d=64, r=16, m=128, ctx_dim=None, mode="tgmlite",
                 dropout=0.3, state_norm=False, guard=True,
                 guard_tau=1.5, guard_kappa=4.0, freeze_mod_from=-1):
        super().__init__()
        self.mode = mode
        self.d, self.r = d, r
        # 诊断开关: t>=freeze_mod_from 的窗口强制 s_t=0(静态地板测试, -1=关闭)
        self.freeze_mod_from = freeze_mod_from
        self.proj = nn.Linear(feat_dim, d)
        self.W0 = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W0)
        self.U = nn.Parameter(torch.randn(d, r) * 0.1)
        self.V = nn.Parameter(torch.randn(r, d) * 0.1)
        self.W2 = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W2)
        self.clf = nn.Linear(d, 1)
        self.decoder = nn.Linear(d, feat_dim)      # 自监督预训练用
        # 可学习图门控: H = (1-λ)·ZW + λ·Â(ZW)。λ 初值小(偏向特征驱动),
        # 训练中自动决定图传播的使用程度; λ 的收敛值可作为论文分析素材
        self.g1 = nn.Parameter(torch.tensor(-2.0))
        self.g2 = nn.Parameter(torch.tensor(-2.0))
        self.dropout = dropout

        if mode in ("tgmlite", "tgmlite_nograph"):
            self.controller = SSMController(ctx_dim, m, r,
                                            state_norm=state_norm,
                                            guard=guard,
                                            guard_tau=guard_tau,
                                            guard_kappa=guard_kappa)
        elif mode == "evolvegcn":
            self.controller = GRUController(ctx_dim, r)
        elif mode == "tgmlite_nossm":
            self.controller = NoMemController(ctx_dim, r)
        else:
            self.controller = StaticController(r)

    @property
    def use_graph(self):
        return self.mode not in ("mlp", "tgmlite_nograph")

    def encode(self, Z, s_t, Ahat):
        if s_t is not None:
            Wt = self.W0 + self.U @ torch.diag(s_t) @ self.V
        else:
            Wt = self.W0
        lam1, lam2 = torch.sigmoid(self.g1), torch.sigmoid(self.g2)
        H = Z @ Wt
        if self.use_graph:
            H = (1.0 - lam1) * H + lam1 * torch.sparse.mm(Ahat, H)
        H = F.relu(H)
        H2 = H @ self.W2
        if self.use_graph:
            H2 = (1.0 - lam2) * H2 + lam2 * torch.sparse.mm(Ahat, H2)
        return F.relu(H2)

    def forward_logits(self, Z, ctx, times, Ahats, device):
        """按时间顺序前向, 返回 {t: logits(该时间片全部节点)} 与末态。"""
        state = self.controller.init_state(device)
        if hasattr(self.controller, "gate_trace"):
            self.controller.gate_trace = {}
        out = {}
        for t in times:
            s_t, state = self.controller.step(ctx[t], state)
            if self.freeze_mod_from >= 0 and t >= self.freeze_mod_from and s_t is not None:
                s_t = torch.zeros_like(s_t)   # 静态地板诊断: 该窗口起退化为纯 W0
            if not self.training and hasattr(self.controller, "gate_trace"):
                self.controller.gate_trace[t] = self.controller.last_gate
            H = self.encode(Z, s_t, Ahats[t - 1])
            out[t] = self.clf(F.dropout(H, self.dropout, self.training)).squeeze(-1)
        return out, state


# ==============================================================================
# 4. 训练 / 评估
# ==============================================================================

def chunks(lst, k):
    for i in range(0, len(lst), k):
        yield lst[i:i + k]


def pretrain(model, data, Ahats, X, ctx, train_times, args, device):
    """掩码特征重建自监督预训练(利用全部节点, 含无标签)。"""
    if args.pretrain_epochs <= 0:
        return
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    N, Fd = X.shape
    for ep in range(args.pretrain_epochs):
        model.train()
        opt.zero_grad()
        mask = torch.rand(N, Fd, device=device) < 0.15
        Xm = X.clone()
        Xm[mask] = 0.0
        with torch.no_grad():
            ctx = build_context(model.proj(Xm), data, device)
        tot, nb = 0.0, 0
        for ck in chunks(train_times, args.chunk):
            Z = model.proj(Xm)          # 每个 chunk 重建计算图, 避免重复反传
            state = model.controller.init_state(device)
            loss = 0.0
            for t in ck:
                s_t, state = model.controller.step(ctx[t], state)
                H = torch.utils.checkpoint.checkpoint(
                    model.encode, Z, s_t, Ahats[t - 1], use_reentrant=False)
                nodes = torch.from_numpy(data["node_time"] == t).to(device)
                m_sub = mask[nodes]
                if m_sub.any():
                    rec = model.decoder(H[nodes])
                    loss = loss + F.mse_loss(rec[m_sub], X[nodes][m_sub])
            if isinstance(loss, torch.Tensor):
                (loss / len(ck)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                tot += float(loss.detach()); nb += 1
        opt.step()
        print(f"  [预训练] epoch {ep+1}/{args.pretrain_epochs}  recon_loss={tot/max(nb,1):.4f}")


def train_supervised(model, data, Ahats, X, ctx, split, args, device):
    train_times, val_times, test_times = split
    y = torch.from_numpy(data["y"].astype(np.float32)).to(device)
    labeled_t = torch.from_numpy(data["labeled"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    tr_mask = labeled_t & torch.from_numpy(
        np.isin(data["node_time"], train_times)).to(device)
    n_pos = float(y[tr_mask].sum())
    n_neg = float(tr_mask.sum() - y[tr_mask].sum())
    pos_weight = torch.tensor(min(max(n_neg / max(n_pos, 1.0), 1.0), 20.0),
                              device=device)

    with torch.no_grad():
        ctx = build_context(model.proj(X), data, device)

    best = {"val_ap": -2.0}                   # 低于 nan 兜底值, 保证首轮必记录
    t_epoch = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        opt.zero_grad()
        tot, nb = 0.0, 0
        for ck in chunks(train_times, args.chunk):
            Z = model.proj(X)           # 每个 chunk 重建计算图, 避免重复反传
            state = model.controller.init_state(device)
            loss = 0.0
            for t in ck:
                s_t, state = model.controller.step(ctx[t], state)
                H = torch.utils.checkpoint.checkpoint(
                    model.encode, Z, s_t, Ahats[t - 1], use_reentrant=False)
                m_t = labeled_t & (torch.from_numpy(data["node_time"] == t).to(device))
                if m_t.any():
                    logit = model.clf(
                        F.dropout(H, model.dropout, model.training)).squeeze(-1)
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logit[m_t], y[m_t], pos_weight=pos_weight)
            if isinstance(loss, torch.Tensor):
                (loss / len(ck)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                tot += float(loss.detach()); nb += 1
        opt.step()
        t_epoch.append(time.time() - t0)

        # ---- 验证 / 测试 ----
        with torch.no_grad():
            Z = model.proj(X)
        val = infer_scores(model, Z, ctx, data, Ahats, val_times, device)
        tst = infer_scores(model, Z, ctx, data, Ahats, test_times, device)
        va = average_precision(*val)
        if va != va:                          # nan 兜底(如验证集无正样本)
            va = -1.0
        if va > best["val_ap"]:
            thr, _ = best_f1_threshold(*val)
            p, r, f1 = prf_at(*tst, thr)
            best = dict(val_ap=va, epoch=ep + 1, thr=thr,
                        auc=roc_auc(*tst), ap=average_precision(*tst),
                        p=p, r=r, f1=f1, p100=precision_at_k(*tst, 100),
                        test_scores=tst,
                        gate_trace=dict(getattr(model.controller,
                                                "gate_trace", {})))
        if (ep + 1) % max(1, args.epochs // 5) == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss={tot/max(nb,1):.4f}  "
                  f"val_AP={va:.4f}  (best@{best.get('epoch','-')})")
    best["epoch_time"] = float(np.mean(t_epoch))
    return best


@torch.no_grad()
def infer_scores(model, Z, ctx, data, Ahats, times, device):
    model.eval()
    # 时序控制器从 t=1 开始推演(仅前向, 利用历史上下文预热状态, 无标签泄漏)
    all_times = list(range(1, max(times) + 1))
    logits, _ = model.forward_logits(Z, ctx, all_times, Ahats, device)
    mask = np.isin(data["node_time"], times) & data["labeled"]
    idx = np.where(mask)[0]
    t_of = data["node_time"]
    s = np.zeros(len(idx), dtype=np.float64)
    for j, g in enumerate(idx):
        s[j] = float(logits[t_of[g]][g])   # logits[t] 为全图 N 维输出, 按全局节点号取值
    y = data["y"][mask]
    return y, s


# ==============================================================================
# 5. 无监督异常检测 (--task unsup): 训练完全不用标签, 以重建误差为异常分数
#    基线: if=孤立森林, knn=k近邻距离检测器, ae=深度自编码器
#    本文: tgmlite_nograph / tgmlite = 掩码预训练 + (SSM 时序状态) 的重建误差
# ==============================================================================

class AE(nn.Module):
    """深度自编码器基线: 重建误差作为异常分数。"""

    def __init__(self, f, d=32):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(f, 64), nn.ReLU(), nn.Linear(64, d))
        self.dec = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, f))

    def forward(self, x):
        return self.dec(self.enc(x))


def _unsup_metrics(data, split, s_va, s_te):
    y_va = data["y"][np.isin(data["node_time"], split[1]) & data["labeled"]]
    y_te = data["y"][np.isin(data["node_time"], split[2]) & data["labeled"]]
    thr, _ = best_f1_threshold(y_va, s_va)
    p, r_, f1 = prf_at(y_te, s_te, thr)
    drift = []
    te_times = data["node_time"][np.isin(data["node_time"], split[2]) & data["labeled"]]
    for blk in np.array_split(np.array(split[2]), 3):
        sel = np.isin(te_times, blk)
        if sel.sum() > 0 and y_te[sel].sum() > 0:
            drift.append(prf_at(y_te[sel], s_te[sel], thr)[2])
        else:
            drift.append(float("nan"))
    return dict(auc=roc_auc(y_te, s_te), ap=average_precision(y_te, s_te),
                f1=f1, p=p, r=r_, p100=precision_at_k(y_te, s_te, 100),
                drift=drift)


def _knn_ad_scores(Xtr_t, Xev_t, k=10, chunk=4096):
    """kNN 距离异常分数(GPU 实现): 与训练集 k 个最近邻的平均距离, 越大越异常。"""
    out = []
    for i in range(0, Xev_t.shape[0], chunk):
        d = torch.cdist(Xev_t[i:i + chunk], Xtr_t)
        out.append(d.topk(k, largest=False).values.mean(1))
    return torch.cat(out).cpu().numpy()


def run_unsup(mode, data, Ahats, X, split, args, device):
    tr_win = (data["node_time"] <= args.train_end) & data.get(
        "train_eligible", True)   # 仅用训练期(且训练银行)节点, 全程无标签
    va_mask = np.isin(data["node_time"], split[1]) & data["labeled"]
    te_mask = np.isin(data["node_time"], split[2]) & data["labeled"]
    Xnp = data["X"]
    t0 = time.time()

    if mode in ("if", "knn"):
        fit_idx = np.where(tr_win)[0]
        if mode == "if":
            try:
                from sklearn.ensemble import IsolationForest
            except ImportError:
                print("[跳过] 未检测到 scikit-learn"); return None
            clf = IsolationForest(n_estimators=200, contamination=0.1,
                                  random_state=args.seed, n_jobs=-1)
            clf.fit(Xnp[fit_idx])
            s_va = -clf.score_samples(Xnp[va_mask])
            t1 = time.time()
            s_te = -clf.score_samples(Xnp[te_mask])
        else:
            if len(fit_idx) > 20000:
                rng = np.random.default_rng(args.seed)
                fit_idx = rng.choice(fit_idx, 20000, replace=False)
            Xtr_t = torch.from_numpy(Xnp[fit_idx]).to(device)
            s_va = _knn_ad_scores(Xtr_t, X[torch.from_numpy(va_mask).to(device)])
            t1 = time.time()
            s_te = _knn_ad_scores(Xtr_t, X[torch.from_numpy(te_mask).to(device)])
        lat = (time.time() - t1) * 1000.0
        params, peak = 0, 0.0

    elif mode == "ae":
        set_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        ae = AE(X.shape[1]).to(device)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
        Xtr = X[torch.from_numpy(tr_win).to(device)]
        for ep in range(40):
            opt.zero_grad()
            loss = F.mse_loss(ae(Xtr), Xtr)
            loss.backward(); opt.step()
        with torch.no_grad():
            e = ((ae(X) - X) ** 2).mean(1).cpu().numpy()
        t1 = time.time()
        s_va, s_te = e[va_mask], e[te_mask]
        lat = (time.time() - t1) * 1000.0
        params = sum(p.numel() for p in ae.parameters())
        peak = (torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda" else 0.0)

    else:  # tgmlite / tgmlite_nograph: 掩码重建预训练后的重建误差
        set_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        d = args.hidden
        model = RiskNet(X.shape[1], d=d, r=args.rank, m=args.state,
                        ctx_dim=2 * d + 3, mode=mode, dropout=args.dropout,
                        state_norm=(args.state_norm == "ln"),
                    guard=(args.guard == "on"),
                    guard_tau=args.guard_tau,
                    guard_kappa=args.guard_kappa,
                    freeze_mod_from=args.freeze_mod_from).to(device)
        params = sum(p.numel() for p in model.parameters())
        args.pretrain_epochs = max(args.pretrain_epochs, 15)
        pretrain(model, data, Ahats, X, None, split[0], args, device)
        with torch.no_grad():
            model.eval()
            Z = model.proj(X)
            ctx = build_context(Z, data, device)
            state = model.controller.init_state(device)
            err = torch.zeros(X.shape[0], device=device)
            t_of = torch.from_numpy(data["node_time"]).to(device)
            for t in range(1, max(split[2]) + 1):
                s_t, state = model.controller.step(ctx[t], state)
                H = torch.utils.checkpoint.checkpoint(
                    model.encode, Z, s_t, Ahats[t - 1], use_reentrant=False)
                e = ((model.decoder(H) - X) ** 2).mean(1)
                err[t_of == t] = e[t_of == t]
        t1 = time.time()
        e = err.cpu().numpy()
        s_va, s_te = e[va_mask], e[te_mask]
        lat = (time.time() - t1) * 1000.0
        peak = (torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda" else 0.0)
        if mode == "tgmlite":
            print(f"  [图门控] λ1={torch.sigmoid(model.g1).item():.3f}  "
                  f"λ2={torch.sigmoid(model.g2).item():.3f}")

    epoch_s = (time.time() - t0)
    m = _unsup_metrics(data, split, s_va, s_te)
    return dict(model=mode, params=params, epoch_s=epoch_s, peak_mb=peak,
                lat_ms=lat, best_epoch=0, **m)


# ==============================================================================
# 5b. PU 学习 (--task pu): 阳性-无标签学习
#     场景: 仅有少量"确认非法"(可靠阳性), 无可靠阴性 -> 朴素法把无标签当阴性有偏
#     nnPU (Kiryo et al., NeurIPS 2017): R = π·E_P[ℓ(s,1)] + max(0, E_U[ℓ(s,0)] - π·E_P[ℓ(s,0)])
# ==============================================================================

def nnpu_risk(lp, lu, pi):
    """lp/lu: 阳性/无标签节点的 logits。含负风险校正。"""
    p_risk = F.softplus(-lp).mean()
    n_risk = F.softplus(lu).mean() - pi * F.softplus(lp).mean()
    if n_risk.item() >= 0:
        return p_risk + n_risk
    return p_risk - n_risk          # 负风险: 反向推回非负区域


PU_MODELS = ["rf_naive", "mlp_naive", "tgmlite_naive", "mlp_pu", "tgmlite_pu"]


def run_pu(mode, data, Ahats, X, split, args, device):
    base, kind = mode.rsplit("_", 1)        # {rf,mlp,tgmlite} × {naive,pu}
    is_pu = kind == "pu"
    if base == "rf" and is_pu:
        print("[跳过] RF 无原生 PU 形式, 仅作 naive 基线")
        return None
    set_seed(args.seed)
    elig = data.get("train_eligible", np.ones(len(data["y"]), dtype=bool))
    tr_win = (data["node_time"] <= args.train_end) & elig
    pos_np = tr_win & (data["y"] == 1) & data["labeled"]      # 可靠阳性: 确认非法
    unl_np = tr_win & ~pos_np                                  # 其余一律视为无标签
    va = np.isin(data["node_time"], split[1]) & data["labeled"]
    te = np.isin(data["node_time"], split[2]) & data["labeled"]
    print(f"  [PU] 阳性 {pos_np.sum()} 条 | 无标签 {unl_np.sum()} 条 "
          f"| π={args.pu_pi} | {'nnPU' if is_pu else 'naive(无标签=阴性)'}")

    if base == "rf":
        try:
            from sklearn.ensemble import RandomForestClassifier
        except ImportError:
            print("[跳过] 未检测到 scikit-learn"); return None
        Xtr = np.concatenate([data["X"][pos_np], data["X"][unl_np]])
        ytr = np.concatenate([np.ones(int(pos_np.sum())),
                              np.zeros(int(unl_np.sum()))])
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=args.seed)
        t0 = time.time()
        clf.fit(Xtr, ytr)
        epoch_s = time.time() - t0
        s_va = clf.predict_proba(data["X"][va])[:, 1]
        t0 = time.time()
        s_te = clf.predict_proba(data["X"][te])[:, 1]
        lat = (time.time() - t0) * 1000.0
        m = _unsup_metrics(data, split, s_va, s_te)
        return dict(model=mode, params=0, epoch_s=epoch_s, peak_mb=0.0,
                    lat_ms=lat, best_epoch=0, **m)

    # ---- 神经模型 ----
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    d = args.hidden
    model = RiskNet(X.shape[1], d=d, r=args.rank, m=args.state,
                    ctx_dim=2 * d + 3, mode=base, dropout=args.dropout,
                    state_norm=(args.state_norm == "ln"),
                    guard=(args.guard == "on"),
                    guard_tau=args.guard_tau,
                    guard_kappa=args.guard_kappa,
                    freeze_mod_from=args.freeze_mod_from).to(device)
    params = sum(p.numel() for p in model.parameters())
    pretrain(model, data, Ahats, X, None, split[0], args, device)

    pos_t = torch.from_numpy(pos_np).to(device)
    unl_t = torch.from_numpy(unl_np).to(device)
    t_of = torch.from_numpy(data["node_time"]).to(device)
    train_times = split[0]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    best = {"val_ap": -2.0}
    t_ep = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        opt.zero_grad()
        with torch.no_grad():
            ctx = build_context(model.proj(X), data, device)
        for ck in chunks(train_times, args.chunk):
            Z = model.proj(X)
            state = model.controller.init_state(device)
            loss = 0.0
            for t in ck:
                s_t, state = model.controller.step(ctx[t], state)
                H = torch.utils.checkpoint.checkpoint(
                    model.encode, Z, s_t, Ahats[t - 1], use_reentrant=False)
                logit = model.clf(
                    F.dropout(H, model.dropout, model.training)).squeeze(-1)
                mt = t_of == t
                lp = logit[mt & pos_t]
                lu = logit[mt & unl_t]
                if lp.numel() == 0 or lu.numel() == 0:
                    continue
                if is_pu:
                    loss = loss + nnpu_risk(lp, lu, args.pu_pi)
                else:
                    loss = loss + F.binary_cross_entropy_with_logits(
                        lp, torch.ones_like(lp)) + F.binary_cross_entropy_with_logits(
                        lu, torch.zeros_like(lu))
            if isinstance(loss, torch.Tensor):
                (loss / len(ck)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        t_ep.append(time.time() - t0)

        with torch.no_grad():
            Z = model.proj(X)
        sva = infer_scores(model, Z, ctx, data, Ahats, split[1], device)
        ste = infer_scores(model, Z, ctx, data, Ahats, split[2], device)
        va_ap = average_precision(*sva)
        if va_ap != va_ap:
            va_ap = -1.0
        if va_ap > best["val_ap"]:
            best = dict(val_ap=va_ap, epoch=ep + 1,
                            s_va=sva[1], s_te=ste[1])
        if (ep + 1) % max(1, args.epochs // 5) == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  val_AP={va_ap:.4f}  (best@{best.get('epoch','-')})")

    if model.use_graph:
        print(f"  [图门控] λ1={torch.sigmoid(model.g1).item():.3f}  "
              f"λ2={torch.sigmoid(model.g2).item():.3f}")
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        Z = model.proj(X)
        infer_scores(model, Z, ctx, data, Ahats, split[2], device)
    lat = (time.time() - t0) * 1000.0
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    m = _unsup_metrics(data, split, best["s_va"], best["s_te"])
    return dict(model=mode, params=params, epoch_s=float(np.mean(t_ep)),
                peak_mb=peak, lat_ms=lat, best_epoch=best["epoch"], **m)


# ==============================================================================
# 6. 主流程
# ==============================================================================

MODELS = ["rf", "mlp", "gcn", "evolvegcn", "tgmlite_nograph", "tgmlite_nossm", "tgmlite"]
UNSUP_MODELS = ["if", "knn", "ae", "tgmlite_nograph", "tgmlite"]


def run_rf(data, split, args):
    """随机森林基线(Elliptic 原论文最强经典基线)。需 scikit-learn, 未装则跳过。"""
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError:
        print("[跳过] 未检测到 scikit-learn (pip install scikit-learn), 跳过 rf")
        return None
    tr = data["labeled"] & np.isin(data["node_time"], split[0])
    va = data["labeled"] & np.isin(data["node_time"], split[1])
    te = data["labeled"] & np.isin(data["node_time"], split[2])
    clf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=args.seed)
    t0 = time.time()
    clf.fit(data["X"][tr], data["y"][tr])
    epoch_s = time.time() - t0
    s_va = clf.predict_proba(data["X"][va])[:, 1]
    t0 = time.time()
    s_te = clf.predict_proba(data["X"][te])[:, 1]
    lat = (time.time() - t0) * 1000.0
    y_va, y_te = data["y"][va], data["y"][te]
    thr, _ = best_f1_threshold(y_va, s_va)
    p, r_, f1 = prf_at(y_te, s_te, thr)
    drift = []
    te_times = data["node_time"][te]
    for blk in np.array_split(np.array(split[2]), 3):
        sel = np.isin(te_times, blk)
        if sel.sum() > 0 and y_te[sel].sum() > 0:
            drift.append(prf_at(y_te[sel], s_te[sel], thr)[2])
        else:
            drift.append(float("nan"))
    return dict(model="rf", params=0, auc=roc_auc(y_te, s_te),
                ap=average_precision(y_te, s_te), f1=f1, p=p, r=r_,
                p100=precision_at_k(y_te, s_te, 100), drift=drift,
                epoch_s=epoch_s, peak_mb=0.0, lat_ms=lat, best_epoch=0)


def run_one(mode, data, Ahats, X, split, args, device):
    if mode == "rf":
        return run_rf(data, split, args)
    set_seed(args.seed)
    d = args.hidden
    ctx_dim = 2 * d + 3
    model = RiskNet(X.shape[1], d=d, r=args.rank, m=args.state, ctx_dim=ctx_dim,
                    mode=mode, dropout=args.dropout,
                    state_norm=(args.state_norm == "ln"),
                    guard=(args.guard == "on"),
                    guard_tau=args.guard_tau,
                    guard_kappa=args.guard_kappa,
                    freeze_mod_from=args.freeze_mod_from).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    pretrain(model, data, Ahats, X, None, split[0], args, device)
    best = train_supervised(model, data, Ahats, X, None, split, args, device)
    if model.use_graph:
        print(f"  [图门控] λ1={torch.sigmoid(model.g1).item():.3f}  "
              f"λ2={torch.sigmoid(model.g2).item():.3f}"
              f"  (λ→0=模型少用图; λ→1=重度用图)")

    # 漂移分析: 将测试时间片切成 3 个连续窗口分别算 F1
    y_t, s_t = best["test_scores"]
    thr = best["thr"]
    test_times = np.array(split[2])
    blocks = np.array_split(test_times, 3)
    drift_f1 = []
    for blk in blocks:
        m = np.isin(data["node_time"][data["labeled"]], blk)
        # 与 test_scores 的掩码顺序一致: node_time in test_times & labeled
        full_mask_times = data["node_time"][np.isin(data["node_time"], split[2]) & data["labeled"]]
        sel = np.isin(full_mask_times, blk)
        if sel.sum() > 0 and y_t[sel].sum() > 0:
            _, _, f1b = prf_at(y_t[sel], s_t[sel], thr)
            drift_f1.append(f1b)
        else:
            drift_f1.append(float("nan"))

    # 漂移护栏诊断: 各测试分段的平均门值(1=全速适应; →0=出包络退火到静态 W0)
    gt = best.get("gate_trace", {})
    if gt:
        tw = [float(np.mean([gt[int(t)] for t in blk if int(t) in gt]))
              for blk in blocks]
        print(f"  [漂移护栏] 测试段门值 w1/w2/w3 = "
              + "/".join(f"{x:.2f}" for x in tw)
              + f"  (包络半径 d_max={float(model.controller.d_max):.1f}, "
                f"τ·d_max={float(model.controller.d_max) * model.controller.guard_tau:.1f})")

    # 推理时延
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        Z = model.proj(X)
        ctx = build_context(Z.detach(), data, device)
        infer_scores(model, Z, ctx, data, Ahats, split[2], device)
    lat = (time.time() - t0) * 1000.0

    peak = (torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else 0.0)
    return dict(model=mode, params=n_params, auc=best["auc"], ap=best["ap"],
                f1=best["f1"], p=best["p"], r=best["r"], p100=best["p100"],
                drift=drift_f1, epoch_s=best["epoch_time"],
                peak_mb=peak, lat_ms=lat, best_epoch=best["epoch"])


def print_table(rows):
    hdr = (f"{'model':<16}{'#params':>9}{'AUC':>8}{'AP':>8}{'F1':>8}"
           f"{'P':>8}{'R':>8}{'P@100':>8}{'漂移F1(w1/w2/w3)':>24}"
           f"{'秒/epoch':>10}{'显存MB':>9}{'推理ms':>9}")
    print("\n" + "=" * len(hdr.expandtabs()) )
    print(hdr)
    print("-" * len(hdr.expandtabs()))
    for r in rows:
        drift = "/".join(f"{x:.3f}" if x == x else "nan" for x in r["drift"])
        print(f"{r['model']:<16}{r['params']:>9,}{r['auc']:>8.4f}{r['ap']:>8.4f}"
              f"{r['f1']:>8.4f}{r['p']:>8.4f}{r['r']:>8.4f}{r['p100']:>8.4f}"
              f"{drift:>24}{r['epoch_s']:>10.2f}{r['peak_mb']:>9.0f}{r['lat_ms']:>9.1f}")
    print("=" * len(hdr.expandtabs()))


def save_csv(rows, path):
    keys = ["model", "params", "auc", "ap", "f1", "p", "r", "p100",
            "drift_w1", "drift_w2", "drift_w3", "epoch_s", "peak_mb",
            "lat_ms", "best_epoch"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r["model"], r["params"], f"{r['auc']:.4f}",
                        f"{r['ap']:.4f}", f"{r['f1']:.4f}", f"{r['p']:.4f}",
                        f"{r['r']:.4f}", f"{r['p100']:.4f}",
                        *(f"{x:.4f}" for x in r["drift"]),
                        f"{r['epoch_s']:.2f}", f"{r['peak_mb']:.0f}",
                        f"{r['lat_ms']:.1f}", r["best_epoch"]])
    print(f"[保存] 结果已写入 {path}")


def _append_raw_row(path, row):
    """断点续跑: 多种子模式每完成一个 (seed, model) 组合立即追加落盘,
    格式与 save_csv 完全一致(列序/精度/BOM 均相同)。"""
    new_file = not os.path.isfile(path)
    r = row
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["model", "params", "auc", "ap", "f1", "p", "r", "p100",
                        "drift_w1", "drift_w2", "drift_w3", "epoch_s", "peak_mb",
                        "lat_ms", "best_epoch"])
        w.writerow([r["model"], r["params"], f"{r['auc']:.4f}",
                    f"{r['ap']:.4f}", f"{r['f1']:.4f}", f"{r['p']:.4f}",
                    f"{r['r']:.4f}", f"{r['p100']:.4f}",
                    *(f"{x:.4f}" for x in r["drift"]),
                    f"{r['epoch_s']:.2f}", f"{r['peak_mb']:.0f}",
                    f"{r['lat_ms']:.1f}", r["best_epoch"]])


def _record_progress(progress_path, seed, model):
    """断点续跑: 在 sidecar 文件(raw CSV 同名 + .progress)登记已完成组合。"""
    new_file = not os.path.isfile(progress_path)
    with open(progress_path, "a", newline="", encoding="utf-8") as f:
        if new_file:
            f.write("seed,model\n")
        f.write(f"{seed},{model}\n")


def _load_progress(progress_path):
    """读 sidecar, 返回已完成 (seed, model) 集合。"""
    done = set()
    with open(progress_path, encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln and ln != "seed,model":
                s_, m_ = ln.split(",", 1)
                done.add((int(s_), m_))
    return done


def _read_raw_csv(path):
    """从磁盘读回完整 raw CSV (聚合用; 不依赖本次运行的内存状态, 续跑口径一致)。"""
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for d in csv.DictReader(f):
            rows.append({"model": d["model"], "params": int(d["params"]),
                         "auc": float(d["auc"]), "ap": float(d["ap"]),
                         "f1": float(d["f1"]), "p": float(d["p"]),
                         "r": float(d["r"]), "p100": float(d["p100"]),
                         "drift": [float(d["drift_w1"]), float(d["drift_w2"]),
                                   float(d["drift_w3"])],
                         "epoch_s": float(d["epoch_s"]),
                         "peak_mb": float(d["peak_mb"]),
                         "lat_ms": float(d["lat_ms"]),
                         "best_epoch": int(d["best_epoch"])})
    return rows


def aggregate_seeds(rows):
    """多种子结果聚合: 按模型分组, 主要指标计算 mean±std。"""
    keys = ["auc", "ap", "f1", "p", "r", "p100"]
    agg = {}
    for r in rows:
        a = agg.setdefault(r["model"], {k: [] for k in keys + ["w1", "w2", "w3"]})
        for k in keys:
            a[k].append(r[k])
        for i, w in enumerate(["w1", "w2", "w3"]):
            a[w].append(r["drift"][i])
    out = []
    for m, a in agg.items():
        row = {"model": m, "n_seeds": len(a["ap"])}
        for k, v in a.items():
            arr = np.array([x for x in v if x == x], dtype=np.float64)  # 去 nan
            row[k + "_mean"] = float(arr.mean()) if len(arr) else float("nan")
            row[k + "_std"] = float(arr.std()) if len(arr) > 1 else 0.0
        out.append(row)
    return out


def print_agg_table(agg):
    print(f"\n{'='*100}\n多种子聚合 (mean±std)\n{'='*100}")
    hdr = f"{'model':<16}{'n':>3}{'AUC':>17}{'AP':>17}{'F1':>17}{'w3漂移F1':>17}"
    print(hdr); print("-" * len(hdr))
    for r in agg:
        def ms(k): return f"{r[k+'_mean']:.4f}±{r[k+'_std']:.4f}"
        print(f"{r['model']:<16}{r['n_seeds']:>3}{ms('auc'):>17}{ms('ap'):>17}"
              f"{ms('f1'):>17}{ms('w3'):>17}")
    print("=" * 100)


def save_agg_csv(agg, path):
    keys = ["auc", "ap", "f1", "p", "r", "p100", "w1", "w2", "w3"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["model", "n_seeds"] +
                   [f"{k}_{s}" for k in keys for s in ("mean", "std")])
        for r in agg:
            w.writerow([r["model"], r["n_seeds"]] +
                       [f"{r[k+'_'+s]:.4f}" for k in keys for s in ("mean", "std")])
    print(f"[保存] 聚合结果已写入 {path}")


def main():
    ap_ = argparse.ArgumentParser(description="TGM-Lite 链上金融风险小模型实验脚本")
    ap_.add_argument("--task", default="supervised",
                     choices=["supervised", "unsup", "pu"],
                     help="supervised=有监督; unsup=无监督异常检测; pu=阳性-无标签学习")
    ap_.add_argument("--pu_pi", type=float, default=0.022,
                     help="PU 类先验 π=P(非法), Elliptic 全网约 2.2%")
    ap_.add_argument("--data", default="synthetic",
                     choices=["synthetic", "elliptic", "amlsim", "custom"])
    ap_.add_argument("--custom_npz", default="",
                     help="--data custom 时: paysim_adapter.py 生成的 npz 路径 (E3)")
    ap_.add_argument("--elliptic_dir", default="auto",
                     help="Elliptic 数据目录; 默认 auto = kagglehub 自动下载")
    ap_.add_argument("--amlsim_dir", default="auto",
                     help="AMLSim 数据目录或交易 csv 路径; 默认 auto = 自动获取 HI-Small_Trans.csv;"
                          " 指向尚不存在的具体 csv(如 HI-Medium_Trans.csv)时自动下载该文件")
    ap_.add_argument("--window_hours", type=int, default=4,
                     help="AMLSim 时间窗口长度(小时), 仅 window_mode=time 时有效")
    ap_.add_argument("--window_mode", default="quantile", choices=["quantile", "time"],
                     help="AMLSim 分箱: quantile=等频(每窗交易量相近, 抗尾部空窗, 推荐); time=固定时长")
    ap_.add_argument("--n_windows", type=int, default=60,
                     help="AMLSim 等频分箱的窗口数 T")
    ap_.add_argument("--amlsim_max_nodes", type=int, default=300000,
                     help="AMLSim 分层子采样上限(保留全部洗钱交易); 0=不采样")
    ap_.add_argument("--model", default="all",
                     help="all, 或逗号分隔的模型名: " + ",".join(MODELS))
    ap_.add_argument("--epochs", type=int, default=25)
    ap_.add_argument("--pretrain_epochs", type=int, default=3)
    ap_.add_argument("--hidden", type=int, default=64)
    ap_.add_argument("--rank", type=int, default=16, help="低秩调制秩 r")
    ap_.add_argument("--state", type=int, default=128, help="SSM 状态维 m")
    ap_.add_argument("--dropout", type=float, default=0.3)
    ap_.add_argument("--lr", type=float, default=5e-3)
    ap_.add_argument("--chunk", type=int, default=8, help="截断 BPTT 窗口(时间片)")
    ap_.add_argument("--state_norm", default="none", choices=["ln", "none"],
                     help="控制器状态 LayerNorm。10-seed 实测对尾段崩溃无效且"
                          "压低收入健康盆地, 默认 none(=v1 行为); ln 仅作对照")
    ap_.add_argument("--guard", default="on", choices=["on", "off"],
                     help="漂移护栏: 测试期上下文超出训练包络时调制自动退火到静态 W0;"
                          " off=消融/复现 v1 旧行为")
    ap_.add_argument("--guard_tau", type=float, default=1.5,
                     help="护栏包络缓冲倍数 τ(设计常数, 非测试集调参结果)")
    ap_.add_argument("--guard_kappa", type=float, default=4.0,
                     help="护栏过渡陡度 κ(设计常数)")
    ap_.add_argument("--freeze_mod_from", type=int, default=-1,
                     help="诊断: t>=此时间片的窗口强制 s_t=0(静态地板测试,"
                          "检验崩溃种子在纯 W0 下是否安全); -1=关闭")
    ap_.add_argument("--train_end", type=int, default=34)
    ap_.add_argument("--val_end", type=int, default=39)
    ap_.add_argument("--feature_set", default="all", choices=["all", "local"],
                     help="all=全部166维(含72维预聚合邻居特征); local=仅前94维局部特征"
                          "(EvolveGCN 论文的 LF 设置, 此设置下图结构成为唯一邻居信息来源)")
    ap_.add_argument("--label_rate", type=float, default=1.0,
                     help="训练标注保留比例(标签稀缺实验): 1.0=全量, 0.05=5%")
    ap_.add_argument("--tag", default="",
                     help="输出文件名附加后缀(防覆盖), 如 --tag _evolvegcn")
    ap_.add_argument("--split_mode", default="time", choices=["time", "bank"],
                     help="time=纯时间外推; bank=跨银行泛化(时间外推 + 测试期为未见过的银行,"
                          " 仅 AMLSim 支持)")
    ap_.add_argument("--test_bank_frac", type=float, default=0.3,
                     help="跨银行模式: 留出作为测试的银行比例")
    ap_.add_argument("--edge_fan", type=int, default=1,
                     help="AMLSim 图边: 1=资金链边+同窗口团边(fan-in/out), 0=仅链边")
    ap_.add_argument("--seed", type=int, default=42)
    ap_.add_argument("--seeds", default="",
                     help="多种子模式: 如 '42,43,44,45,46'。数据划分固定(seed=42),"
                          " 仅模型初始化随种子变化, 输出 mean±std")
    args = ap_.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[环境] device={device}  torch={torch.__version__}")

    # 多种子模式: 数据相关随机性固定为 42, 仅模型初始化随种子变化
    data_seed = 42 if args.seeds else args.seed

    if args.data == "synthetic":
        data = make_synthetic(seed=data_seed)
    elif args.data == "amlsim":
        data = load_amlsim(args.amlsim_dir, window_hours=args.window_hours,
                           max_nodes=args.amlsim_max_nodes, seed=data_seed,
                           edge_fan=bool(args.edge_fan),
                           window_mode=args.window_mode, n_windows=args.n_windows)
        # 自适应超参(默认值按 Elliptic 设定, 对 AMLSim 需按数据规模调整)
        if args.train_end == 34 and args.val_end == 39:
            args.train_end = max(2, int(data["T"] * 0.6))
            args.val_end = max(args.train_end + 1, int(data["T"] * 0.8))
            print(f"[自适应] AMLSim 时间划分: 训练 1-{args.train_end} | "
                  f"验证 {args.train_end+1}-{args.val_end} | "
                  f"测试 {args.val_end+1}-{data['T']}")
        if args.chunk == 8:
            args.chunk = 4
            print("[自适应] 节点规模较大, chunk 8 -> 4 (防显存溢出)")
    elif args.data == "elliptic":
        data = load_elliptic(ensure_elliptic(args.elliptic_dir))
    else:  # custom: E3 真实/异源数据通道 (PaySim 派生图 / 工业数据适配器产物)
        if not args.custom_npz:
            raise SystemExit("[错误] --data custom 需要 --custom_npz 指定 npz 路径")
        data = load_custom(args.custom_npz)
        if args.train_end == 34 and args.val_end == 39:
            args.train_end = max(2, int(data["T"] * 0.6))
            args.val_end = max(args.train_end + 1, int(data["T"] * 0.8))
            print(f"[自适应] custom 时间划分: 训练 1-{args.train_end} | "
                  f"验证 {args.train_end+1}-{args.val_end} | "
                  f"测试 {args.val_end+1}-{data['T']}")
        if args.chunk == 8 and len(data["y"]) > 100000:
            args.chunk = 4
            print("[自适应] 节点规模较大, chunk 8 -> 4")

    # 跨银行泛化协议(必须先于标准化与 π 估计, 防测试银行统计泄漏):
    # 留出 test_bank_frac 比例的银行, 训练/验证期只见训练银行, 测试期只评估留出银行
    if args.split_mode == "bank":
        if "bank" not in data:
            raise SystemExit("[错误] --split_mode bank 仅支持含银行信息的数据集(amlsim)")
        rng_b = np.random.default_rng(data_seed)
        banks = np.unique(data["bank"])
        n_tb = max(1, int(len(banks) * args.test_bank_frac))
        test_banks = rng_b.choice(banks, size=n_tb, replace=False)
        in_tb = np.isin(data["bank"], test_banks)
        in_te_win = data["node_time"] > args.val_end
        data["labeled"] = data["labeled"] & np.where(in_te_win, in_tb, ~in_tb)
        data["train_eligible"] = ~in_tb
        n_tr_lab = int((data["labeled"] & ~in_te_win).sum())
        n_te_lab = int((data["labeled"] & in_te_win).sum())
        print(f"[跨银行] 共 {len(banks)} 家银行, 留出 {n_tb} 家作测试 "
              f"({in_tb.mean()*100:.1f}% 节点) | 可训练标注 {n_tr_lab:,} | "
              f"测试银行标注 {n_te_lab:,}")
    else:
        data["train_eligible"] = np.ones(len(data["labeled"]), dtype=bool)

    if args.data in ("amlsim", "custom") and args.pu_pi == 0.022:
        tr0 = (data["node_time"] <= args.train_end) & data["train_eligible"]
        args.pu_pi = float(data["y"][tr0].mean())
        print(f"[自适应] PU 先验 π = 训练区洗钱率 = {args.pu_pi:.4f}")

    # LF 设置: 仅保留局部特征(前 94 维), 去掉数据集预聚合的邻居统计特征(仅 Elliptic 适用)
    if args.feature_set == "local" and data["X"].shape[1] >= 166:
        data["X"] = data["X"][:, :94]
        print(f"[特征] LF 设置: 仅使用前 94 维局部特征 (图结构成为唯一邻居信息来源)")

    # 特征标准化(仅用训练时间片统计量, 防信息泄漏; 跨银行模式下进一步只用训练银行)
    tr = (data["node_time"] <= args.train_end) & data["train_eligible"]
    mu, sd = data["X"][tr].mean(0), data["X"][tr].std(0) + 1e-6
    data["X"] = ((data["X"] - mu) / sd).astype(np.float32)

    Ahats, _ = build_snapshots(data, device)
    X = torch.from_numpy(data["X"]).to(device)

    T = data["T"]
    split = (list(range(1, args.train_end + 1)),
             list(range(args.train_end + 1, args.val_end + 1)),
             list(range(args.val_end + 1, T + 1)))
    print(f"[划分] 训练 t=1-{args.train_end} | 验证 t={args.train_end+1}-{args.val_end} "
          f"| 测试 t={args.val_end+1}-{T}  (严格时间外推)")

    # 标签稀缺实验: 分层保留训练区标注(预训练仍用全部节点, 这是神经模型的优势所在)
    if args.label_rate < 1.0:
        rng = np.random.default_rng(data_seed)
        trm = np.where(data["labeled"] & (data["node_time"] <= args.train_end))[0]
        keep = []
        for cls in (0, 1):
            idx_c = trm[data["y"][trm] == cls]
            k = max(1, int(round(len(idx_c) * args.label_rate)))
            keep.append(rng.choice(idx_c, size=min(k, len(idx_c)), replace=False))
        keep = np.concatenate(keep)
        data["labeled"] = data["labeled"].copy()
        data["labeled"][trm] = False
        data["labeled"][keep] = True
        n_ill = int(data["y"][keep].sum())
        print(f"[标签稀缺] 训练标注 {len(trm)} -> {len(keep)} "
              f"({args.label_rate*100:.0f}%), 其中非法 {n_ill} 条")

    if args.task == "unsup":
        modes = UNSUP_MODELS if args.model == "all" else [
            m.strip() for m in args.model.split(",") if m.strip() in UNSUP_MODELS]
        print(f"[任务] 无监督异常检测: 训练零标签, 评估于带标签的验证/测试节点")
        runner = run_unsup
    elif args.task == "pu":
        modes = PU_MODELS if args.model == "all" else [
            m.strip() for m in args.model.split(",") if m.strip() in PU_MODELS]
        print(f"[任务] PU 学习: 仅阳性+无标签(隐藏合法标签), 评估于带标签的验证/测试节点")
        runner = run_pu
    else:
        modes = MODELS if args.model == "all" else [
            m.strip() for m in args.model.split(",") if m.strip() in MODELS]
        runner = run_one
    # ---- 多种子模式: 循环训练, 聚合 mean±std ----
    seed_list = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    multi = len(seed_list) > 1
    # 文件名标签: 跨银行/标签稀缺实验自动区分, 避免多次运行互相覆盖
    tag = ("_bank" if args.split_mode == "bank" else "") + \
          (f"_lr{args.label_rate:g}" if args.label_rate < 1.0 else "") + \
          ("_nopre" if args.pretrain_epochs <= 0 else "") + \
          ("_ln" if args.state_norm == "ln" else "") + \
          ("_noguard" if getattr(args, "guard", "on") == "off" else "") + \
          (f"_fz{args.freeze_mod_from}" if getattr(args, "freeze_mod_from", -1) >= 0 else "") + args.tag

    raw_path = f"TGM-Lite_实验结果_{args.task}_{args.data}{tag}_seeds_raw.csv"
    progress_path = raw_path + ".progress"

    # ---- 断点续跑: 已有 raw + sidecar 时跳过已完成组合; 旧版无 sidecar 的完整结果文件
    # 不启用续跑(防止重复追加), 保持旧行为(结束时整体覆盖) ----
    resume = multi
    done = set()
    if multi and os.path.isfile(raw_path):
        if os.path.isfile(progress_path):
            done = _load_progress(progress_path)
            if done:
                print(f"[续跑] 检测到 {raw_path} 已有 {len(done)} 个组合, 自动跳过")
        else:
            resume = False
            print(f"[警告] {raw_path} 为旧版一次性结果(无断点记录 .progress), "
                  f"本次不启用续跑, 结束时整体覆盖该文件")

    # ---- 多种子模式: 循环训练, 聚合 mean±std ----
    all_rows = []
    for si, sd in enumerate(seed_list):
        args.seed = sd
        if multi:
            print(f"\n{'#'*70}\n# 种子 {sd} ({si+1}/{len(seed_list)})\n{'#'*70}")
        rows = []
        for m in modes:
            if resume and (sd, m) in done:
                print(f"\n>>> 跳过 {m} (seed={sd}): 已在 {raw_path} 中")
                continue
            print(f"\n>>> 训练模型: {m} (seed={sd})")
            row = runner(m, data, Ahats, X, split, args, device)
            if row is not None:
                row["seed"] = sd
                rows.append(row)
                if resume:
                    _append_raw_row(raw_path, row)          # 逐行落盘, 中断不丢
                    _record_progress(progress_path, sd, m)
        print_table(rows)
        all_rows.extend(rows)

    if not multi:
        save_csv(all_rows, f"TGM-Lite_实验结果_{args.task}_{args.data}{tag}.csv")
    elif resume:
        full_rows = _read_raw_csv(raw_path)   # 含本次运行前已完成的行
        agg = aggregate_seeds(full_rows)
        print_agg_table(agg)
        save_agg_csv(agg, f"TGM-Lite_实验结果_{args.task}_{args.data}{tag}_seeds.csv")
    else:
        save_csv(all_rows, raw_path)
        agg = aggregate_seeds(all_rows)
        print_agg_table(agg)
        save_agg_csv(agg, f"TGM-Lite_实验结果_{args.task}_{args.data}{tag}_seeds.csv")


if __name__ == "__main__":
    main()
