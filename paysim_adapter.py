# -*- coding: utf-8 -*-
"""
paysim_adapter.py — E3: PaySim 交易日志 -> TGM-Lite 通用时序图 npz
================================================================================
评审点 (TIFS 决定性): 崩溃签名目前只在 AMLSim HI-Small 一个生成器配置上复现。
PaySim 是**另一套生成器族**(移动货币模拟器, 参数校准自非洲某国真实移动货币
日志, Lopez-Rojas et al. 2016), 交易级 isFraud 标签, 744 个小时步, 约 636 万笔
交易 (欺诈 8,213 笔, 0.13%)。把 PaySim 转成与 load_elliptic/load_amlsim 完全
同构的时序图后, 整条审计管线 (含 fz40 地板协议) 可以原样跑通 —— 这是对
"不同生成器族"的最强可行检验。

下载: 无需手动 —— --csv 指向的本地文件缺失时, 自动经 kagglehub 免登录下载
       (Kaggle 公开数据集 `ealaxi/paysim1`, 先单文件 ~180MB, 失败再整包);
       也可手动从该数据集页下载 PS_20174392719_1491204439457_log.csv 后用 --csv 指向

图构建 (全部确定性, 固定种子 42):
  窗口   : 每 --hours_per_window (默认 12) 个小时步并为一窗 -> T=62 (≈AMLSim 60)
  节点   : 交易; 分层子采样到 --max_nodes (默认 300k, 保留全部欺诈交易,
           与 --amlsim_max_nodes 同语义)
  边     : (a) 同窗口同付款账户星形边; (b) 同窗口同收款账户星形边;
           (c) 跨窗资金链边: 同一付款账户在 t 窗的最后一笔 -> t+1 窗的第一笔
           (边归属源节点时间片, 与 build_snapshots 约定一致; 对称化由其完成)
  特征   : log1p(amount) + 类型 one-hot x5 + 小时 sin/cos + isFlaggedFraud
           + 按(账户,窗口)在全量数据上预聚合的出入度/出入金额/对手方数 x6
           **不含任何余额列** —— PaySim 中欺诈交易被模拟器作废, 余额字段被
           置零/失真, 含余额会把标签以伪迹形式泄漏给模型 (详见 E3 手稿模板)。
  标签   : isFraud (PaySim 无未知类, 全节点有标注)

用法:
  python paysim_adapter.py --csv PS_20174392719_1491204439457_log.csv ^
      --out paysim_graph.npz --hours_per_window 12 --max_nodes 300000

然后:
  python experiments.py --data custom --custom_npz paysim_graph.npz ...
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd

PAYSIM_HANDLE = "ealaxi/paysim1"
PAYSIM_FILE = "PS_20174392719_1491204439457_log.csv"


def ensure_paysim(csv_path):
    """解析 PaySim 原始 CSV 位置: 已存在则直接用; 缺失则按文件名自动获取
    (kagglehub 免登录, 先单文件下载, 失败再整包)。下载缓存到 CSV 所在目录。"""
    if os.path.isfile(csv_path):
        print(f"[数据] 使用本地文件: {csv_path}")
        return csv_path
    target = os.path.basename(csv_path) if csv_path.lower().endswith(".csv") else PAYSIM_FILE
    folder = os.path.dirname(os.path.abspath(csv_path))
    for cand in dict.fromkeys([folder, os.getcwd()]):   # 同目录/当前目录先找
        hit = os.path.join(cand, target)
        if os.path.isfile(hit):
            print(f"[数据] 使用本地文件: {hit}")
            return hit
    print(f"[数据] {target} 不存在, 通过 kagglehub 免登录下载 "
          f"(Kaggle {PAYSIM_HANDLE}, 公开数据集) ...")
    try:
        import kagglehub
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        import kagglehub
    try:
        os.makedirs(folder, exist_ok=True)
        os.environ["KAGGLEHUB_CACHE"] = folder   # 下载缓存到数据目录
    except OSError:
        pass
    try:
        hit = kagglehub.dataset_download(PAYSIM_HANDLE, path=target)  # 单文件下载
        if hit and os.path.isfile(hit):
            print(f"[数据] 单文件下载完成: {hit}")
            return hit
    except Exception as e:  # noqa: BLE001
        print(f"[提示] 单文件下载未成功({e}), 转为整包下载 ...")
    try:
        root = kagglehub.dataset_download(PAYSIM_HANDLE)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "\n[下载失败] kagglehub 免登录下载未成功:\n  " + str(e) + "\n"
            "备用方案: 打开 https://www.kaggle.com/datasets/ealaxi/paysim1 手动下载 "
            + target + " 后用 --csv 指向该文件。")
    for dirpath, _, files in os.walk(root):
        for f in files:
            if f == target:
                hit = os.path.join(dirpath, f)
                print(f"[数据] 整包下载完成, 使用: {hit}")
                return hit
    raise SystemExit(f"[错误] 下载目录 {root} 中未找到 {target}")


def sniff_extract_csv(path):
    """Kaggle/kagglehub 单文件下载有时以 zip/gzip 容器下发(文件名仍是 .csv,
    直接 pandas 读取会报 UnicodeDecodeError)。嗅探魔数:
    PK -> 解出真正的 csv 到 <path>.real.csv; 1f 8b -> gzip 解压; 否则原样返回。"""
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic[:2] == b"PK":
        out = path + ".real.csv"
        if os.path.isfile(out):
            print(f"[数据] 使用已解压文件: {out}")
            return out
        import shutil as _sh
        import zipfile as _zf
        print("[数据] 检测到 zip 容器(Kaggle 单文件下发格式), 解出真实 CSV ...")
        with _zf.ZipFile(path) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            if not names:
                raise SystemExit(f"[错误] {path} 是 zip 但内部无 .csv 成员")
            with z.open(names[0]) as src, open(out, "wb") as dst:
                _sh.copyfileobj(src, dst, 1024 * 1024)
        print(f"[数据] 解压完成: {out} ({os.path.getsize(out)/1e6:.0f} MB)")
        return out
    if magic[:2] == b"\x1f\x8b":
        out = path + ".real.csv"
        if os.path.isfile(out):
            return out
        import gzip as _gz
        import shutil as _sh
        print("[数据] 检测到 gzip 容器, 解压中 ...")
        with _gz.open(path, "rb") as src, open(out, "wb") as dst:
            _sh.copyfileobj(src, dst, 1024 * 1024)
        return out
    return path


def main():
    ap = argparse.ArgumentParser(description="PaySim -> TGM-Lite 时序图 npz (E3)")
    ap.add_argument("--csv", required=True, help="PaySim 原始 CSV 路径")
    ap.add_argument("--out", default="paysim_graph.npz")
    ap.add_argument("--hours_per_window", type=int, default=12)
    ap.add_argument("--max_nodes", type=int, default=300000,
                    help="0 = 不采样 (6.4M 节点, 显存/内存不可行, 仅供完整性)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.csv = sniff_extract_csv(ensure_paysim(args.csv))

    print("[1/5] 读取 PaySim CSV (仅所需列) ...")
    use = ["step", "type", "amount", "nameOrig", "nameDest",
           "isFraud", "isFlaggedFraud"]
    df = pd.read_csv(args.csv, usecols=use,
                     dtype={"step": np.int32, "amount": np.float64,
                            "isFraud": np.int8, "isFlaggedFraud": np.int8})
    n_all = len(df)
    win = ((df["step"].to_numpy() - 1) // args.hours_per_window + 1).astype(np.int64)
    T = int(win.max())
    print(f"      {n_all:,} 笔交易, T={T} 窗 ({args.hours_per_window}h/窗), "
          f"欺诈 {int(df['isFraud'].sum()):,} ({df['isFraud'].mean()*100:.3f}%)")

    print("[2/5] 账户因子化 + 全量 (账户,窗口) 预聚合特征 ...")
    orig = pd.factorize(df["nameOrig"])[0].astype(np.int64)
    dest = pd.factorize(df["nameDest"])[0].astype(np.int64)
    amt = df["amount"].to_numpy(np.float64)
    n_acc = int(max(orig.max(), dest.max())) + 1

    def agg(acc, cp, w, a):
        """按 (账户,窗口) 全向量化聚合, 返回逐行对齐的 (笔数, 金额和, 对手方数)。
        对手方数 = 组内不同 cp 账户数 (付款方看收款方, 收款方看付款方)。"""
        key = acc * (T + 1) + (w - 1)
        uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        tot = np.bincount(inv, weights=a)
        pair = key * (n_acc + 1) + cp
        upair = np.unique(pair)
        keys_u, opp = np.unique(upair // (n_acc + 1), return_counts=True)
        assert len(keys_u) == len(uk) and (keys_u == uk).all()
        return cnt[inv].astype(np.float64), tot[inv], opp[inv].astype(np.float64)

    f_out_cnt, f_out_tot, f_out_opp = agg(orig, dest, win, amt)
    f_in_cnt, f_in_tot, f_in_opp = agg(dest, orig, win, amt)

    print("[3/5] 组装逐交易特征 (不含余额列, 防作废伪迹泄漏) ...")
    types = pd.Categorical(df["type"],
                           categories=["CASH_IN", "CASH_OUT", "DEBIT",
                                       "PAYMENT", "TRANSFER"])
    onehot = np.eye(5, dtype=np.float32)[types.codes]
    hour = ((df["step"].to_numpy() - 1) % 24).astype(np.float64)
    X_all = np.column_stack([
        np.log1p(amt),
        onehot,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        df["isFlaggedFraud"].to_numpy(np.float64),
        np.log1p(f_out_cnt), np.log1p(f_out_tot), np.log1p(f_out_opp),
        np.log1p(f_in_cnt), np.log1p(f_in_tot), np.log1p(f_in_opp),
    ]).astype(np.float32)
    y_all = df["isFraud"].to_numpy(np.int64)

    print("[4/5] 分层子采样 (保留全部欺诈) ...")
    rng = np.random.default_rng(args.seed)
    fraud_idx = np.where(y_all == 1)[0]
    if args.max_nodes and args.max_nodes < n_all:
        n_legit = args.max_nodes - len(fraud_idx)
        legit_pool = np.where(y_all == 0)[0]
        legit_idx = np.sort(rng.choice(legit_pool, size=n_legit, replace=False))
        keep = np.sort(np.concatenate([fraud_idx, legit_idx]))
    else:
        keep = np.arange(n_all)
    remap = -np.ones(n_all, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    X, y = X_all[keep], y_all[keep]
    node_time = win[keep]
    o_k, d_k = orig[keep], dest[keep]
    print(f"      节点 {len(keep):,} (欺诈 {int(y.sum()):,}, "
          f"{y.mean()*100:.3f}%)")

    print("[5/5] 建边 (同窗星形 + 跨窗资金链) ...")
    src_l, dst_l = [], []

    def star_edges(acc, w):
        """同 (账户,窗口) 内连到该组首笔交易 (星形, 避免团边平方爆炸)。"""
        grp = acc * (T + 1) + (w - 1)
        order = np.argsort(grp, kind="stable")
        g_sorted = grp[order]
        boundaries = np.r_[0, np.where(np.diff(g_sorted) != 0)[0] + 1]
        first_of = np.empty(len(order), dtype=np.int64)
        first_of[boundaries] = order[boundaries]
        first_of = np.maximum.accumulate(first_of)
        tgt = first_of                     # 每行所属组的首行 (排序后坐标)
        rows_sorted = order
        mask = rows_sorted != tgt
        s = tgt[mask]
        d = rows_sorted[mask]
        return s, d

    for acc, tag in ((o_k, "orig"), (d_k, "dest")):
        s, d = star_edges(acc, node_time)
        src_l.append(s); dst_l.append(d)
        print(f"      同窗星形边 ({tag}): {len(s):,}")

    # 跨窗资金链: 同一付款账户 t 窗末笔 -> t+1 窗首笔
    order = np.lexsort((node_time, o_k))
    o_s, w_s = o_k[order], node_time[order]
    same_acc = np.diff(o_s) == 0
    nxt_win = np.diff(w_s) == 1
    ch = np.where(same_acc & nxt_win)[0]
    s = order[ch]; d = order[ch + 1]
    src_l.append(s); dst_l.append(d)
    print(f"      跨窗资金链边 (orig): {len(s):,}")

    edge_src = np.concatenate(src_l).astype(np.int64)
    edge_dst = np.concatenate(dst_l).astype(np.int64)
    dup = edge_src == edge_dst
    edge_src, edge_dst = edge_src[~dup], edge_dst[~dup]

    labeled = np.ones(len(y), dtype=bool)
    np.savez_compressed(
        args.out, X=X, y=y, labeled=labeled, node_time=node_time,
        edge_src=edge_src, edge_dst=edge_dst, T=np.int64(T),
        name=np.str_(f"paysim_hpw{args.hours_per_window}"))
    print(f"[完成] N={len(y):,}, E={len(edge_src):,}, T={T}, "
          f"特征 {X.shape[1]} 维 -> {args.out}")
    print("下一步: python experiments.py --data custom --custom_npz "
          f"{args.out} --model gcn,tgmlite --epochs 100 --pretrain_epochs 5 "
          "--seeds 42,43,44,45,46,47,48,49,50,51 --tag _paysim10")


if __name__ == "__main__":
    main()
