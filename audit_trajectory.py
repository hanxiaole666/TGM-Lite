# -*- coding: utf-8 -*-
"""
audit_trajectory.py — E6: 崩溃 seed vs 健康 seed 的调制轨迹审计
================================================================================
评审点 (Reviewer B, H4): "缺崩溃 seed 与健康 seed 的 ‖U·diag(s_t)·V‖ / s_t
轨迹量化对比——补一张图即可坐实机理论断 (L4)"。

做法: 按 canonical 主表协议 (AMLSim, train 1-36 / val 37-48 / test 49-60,
chunk=4, lr 5e-3, dropout 0.3, 预训练 5 epochs, 共 100 epochs, val-AP 选模)
重训指定 seed 的 tgmlite, 然后对已选检查点做 deployment-fresh 推理期推演
(t = 1..T), 逐窗口记录:

  s_max    = max_i |s_t[i]|            调制向量最大幅值
  s_l2     = ||s_t||_2                 调制向量能量
  dW_fro   = ||U·diag(s_t)·V||_F       权重空间的实际修改量
  dW_rel   = dW_fro / ||W0||_F         相对静态主干的修改比例
  gate     = 控制器护栏门值 (canonical 配置 guard=off, 恒 1.0, 仅备查)

同时按主表规则评估测试期 w1/w2/w3 F1, 给出该 seed 的 collapse 标记
(w3 F1 < 0.6)。崩溃判定同主文: 与地板配对见 audit_external 的 --floor_csv
流程; 本脚本只负责轨迹与 w3。

输出 (与 experiments.py 同目录):
  TGM-Lite_轨迹审计_<data><tag>.csv    逐 (seed, t) 行, 长表

配套作图: fig_trajectory.py 读该 CSV 生成双面板图 (健康蓝 / 崩溃红)。

示例 (AMLSim, 崩溃种子 47,50 + 健康种子 42,43, 约 1 小时 @ RTX 2060 6GB):
  python audit_trajectory.py --data amlsim --seeds 42,43,47,50 ^
      --epochs 100 --pretrain_epochs 5 --tag _traj
断点续跑: 已完成 seed 自动跳过; 删除 CSV 可强制重跑。
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import experiments as E
from audit_adversarial import train_with_snapshot, drift_blocks

FIELDS = ["seed", "t", "phase", "s_max", "s_l2", "dW_fro", "dW_rel",
          "gate", "w1", "w2", "w3", "collapsed"]


def main():
    ap = argparse.ArgumentParser(description="E6: 崩溃/健康种子的调制轨迹审计")
    ap.add_argument("--data", default="amlsim", choices=["amlsim", "elliptic"])
    ap.add_argument("--amlsim_dir", default="auto")
    ap.add_argument("--elliptic_dir", default="auto")
    ap.add_argument("--window_hours", type=int, default=4)
    ap.add_argument("--window_mode", default="quantile",
                    choices=["quantile", "time"])
    ap.add_argument("--n_windows", type=int, default=60)
    ap.add_argument("--amlsim_max_nodes", type=int, default=300000)
    ap.add_argument("--edge_fan", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--pretrain_epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--state", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--train_end", type=int, default=34)
    ap.add_argument("--val_end", type=int, default=39)
    ap.add_argument("--floor_from", type=int, default=40,
                    help="图中切除点标注位置 (AMLSim canonical=40)")
    ap.add_argument("--seeds", default="42,43,47,50",
                    help="默认 = 健康 42,43 + 崩溃 47,50")
    ap.add_argument("--tag", default="_traj")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[环境] device={device}  torch={torch.__version__}")

    # ---- 数据与划分: 与 audit_adversarial / experiments.main 完全同一入口 ----
    if args.data == "amlsim":
        data = E.load_amlsim(args.amlsim_dir, window_hours=args.window_hours,
                             max_nodes=args.amlsim_max_nodes, seed=42,
                             edge_fan=bool(args.edge_fan),
                             window_mode=args.window_mode,
                             n_windows=args.n_windows)
        if args.train_end == 34 and args.val_end == 39:
            args.train_end = max(2, int(data["T"] * 0.6))
            args.val_end = max(args.train_end + 1, int(data["T"] * 0.8))
        if args.chunk == 8:
            args.chunk = 4
            print("[自适应] chunk 8 -> 4 (AMLSim, 与主表一致)")
    else:
        files = E.ensure_elliptic(args.elliptic_dir)
        data = E.load_elliptic(files)
    print(f"[划分] train 1-{args.train_end} | val {args.train_end+1}-{args.val_end} | "
          f"test {args.val_end+1}-{data['T']}")

    data["train_eligible"] = np.ones(len(data["labeled"]), dtype=bool)
    tr_std = (data["node_time"] <= args.train_end) & data["train_eligible"]
    mu, sd = data["X"][tr_std].mean(0), data["X"][tr_std].std(0) + 1e-6
    data["X"] = ((data["X"] - mu) / sd).astype(np.float32)

    Ahats, _ = E.build_snapshots(data, device)
    X = torch.from_numpy(data["X"]).to(device)
    split = (list(range(1, args.train_end + 1)),
             list(range(args.train_end + 1, args.val_end + 1)),
             list(range(args.val_end + 1, data["T"] + 1)))
    T = data["T"]

    out_path = Path(f"TGM-Lite_轨迹审计_{args.data}{args.tag}.csv")
    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {int(r["seed"]) for r in csv.DictReader(f)}
        if done:
            print(f"[续跑] 已完成 seed: {sorted(done)}, 自动跳过")

    for seed in [int(x) for x in args.seeds.split(",")]:
        if seed in done:
            continue
        print(f"\n===== seed={seed}: 按 canonical 主表协议重训 tgmlite =====")
        E.set_seed(seed)
        model = E.RiskNet(X.shape[1], d=args.hidden, r=args.rank, m=args.state,
                          ctx_dim=2 * args.hidden + 3, mode="tgmlite",
                          dropout=args.dropout, state_norm=False, guard=False,
                          freeze_mod_from=-1).to(device)
        E.pretrain(model, data, Ahats, X, None, split[0], args, device)
        best = train_with_snapshot(model, data, Ahats, X, split, args, device)
        model.load_state_dict(best["state"])
        model.eval()

        # ---- 测试期指标 (deployment-fresh ctx), 用于 collapse 标记 ----
        with torch.no_grad():
            Z = model.proj(X)
            ctx = E.build_context(Z, data, device)
        y_t, s_t = E.infer_scores(model, Z, ctx, data, Ahats, split[2], device)
        w = drift_blocks(data, split, y_t, s_t, best["thr"])
        collapsed = int(w[2] == w[2] and w[2] < 0.6)
        print(f"  seed {seed}: w1={w[0]:.4f} w2={w[1]:.4f} w3={w[2]:.4f} "
              f"-> {'COLLAPSED' if collapsed else 'healthy'}")

        # ---- 逐窗口推演控制器, 记录调制轨迹 (无梯度, 不触碰任何参数) ----
        W0_fro = float(model.W0.detach().norm(p="fro"))
        U, V = model.U.detach(), model.V.detach()
        rows = []
        with torch.no_grad():
            state = model.controller.init_state(device)
            for t in range(1, T + 1):
                st, state = model.controller.step(ctx[t], state)
                dW = U @ torch.diag(st) @ V
                gate = model.controller.last_gate
                rows.append(dict(
                    seed=seed, t=t,
                    phase=("train" if t <= args.train_end else
                           "val" if t <= args.val_end else "test"),
                    s_max=f"{float(st.abs().max()):.6f}",
                    s_l2=f"{float(st.norm()):.6f}",
                    dW_fro=f"{float(dW.norm(p='fro')):.6f}",
                    dW_rel=f"{float(dW.norm(p='fro')) / W0_fro:.6f}",
                    gate=f"{float(gate):.6f}" if np.isscalar(gate) or
                         (isinstance(gate, torch.Tensor) and gate.numel() == 1)
                         else "1.000000",
                    w1=f"{w[0]:.4f}", w2=f"{w[1]:.4f}", w3=f"{w[2]:.4f}",
                    collapsed=collapsed))
        new = not out_path.exists()
        with open(out_path, "a", encoding="utf-8", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=FIELDS)
            if new:
                wcsv.writeheader()
            wcsv.writerows(rows)
        print(f"  [已保存] {len(rows)} 行 -> {out_path.name}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n完成。作图: python fig_trajectory.py --csv {out_path.name} "
          f"--train_end {args.train_end} --val_end {args.val_end} "
          f"--floor_from {args.floor_from}")


if __name__ == "__main__":
    main()
