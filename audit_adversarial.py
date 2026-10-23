# -*- coding: utf-8 -*-
"""
audit_adversarial.py — TGM-Lite 对抗维度审计 (TIFS RQ: adversarial triggers)
================================================================================
同一个训练好的检查点, 在测试期施加两类轻量触发 (test-time only, 不污染训练):

  1) evasion   — FGSM 式符号梯度逃逸攻击: 仅扰动测试窗口中"非法"(y=1)节点的特征,
                 方向为最小化其分类 logit (白盒, 对 adaptive 模型求梯度;
                 时序状态在攻击时固定 — "myopic" 攻击);
                 同一扰动同时评估 adaptive 与 floor (迁移性).
  2) drift     — 无目标漂移触发: 测试窗口全部节点特征沿自然漂移方向
                 sign(mu_test - mu_train) 放大 eps 个标准差.

四个变体 (tgmlite; GCN 无控制器, 仅 adaptive 一格):
  adaptive       — 表 I 检查点, 评估时用当前投影重算 ctx (deployment-fresh).
  adaptive_stale — 同一检查点, 但沿用训练期缓存 ctx 评估 (canonical 表 I 协议复刻;
                   锚点: seed42 clean 应≈AP 0.578 / w3 0.694, 十种子复现 2/10 崩溃).
                   与 adaptive 对照可把"崩溃计数对评估协议敏感性"归因到 ctx 新鲜度.
  floor          — 推理时切除: 表 I 检查点, 测试期 t>=40 强制 s_t:=0 (fresh 评估);
                   对应 IV-E "操作员直接停更控制器"的部署方式.
  floor_fz       — 认证地板: 按 RQ8 协议重训 (val 选择全程冻结 t>=40),
                   stale 评估 (canonical 表 VIII 协议复刻;
                   锚点: seed42 clean w3 应≈0.6729, 十种子 0.6714±0.0139 / 0 崩溃).
攻击仍在自适应检查点上构造、迁移到各地板变体 (标准逃逸迁移协议)。

E5 增补 (评审: "针对切除点的对手"): attack=evasion_fz 时, 攻击改在 fz40 认证地板
检查点上、按已切除构型 (t>=floor_from 的 s_t 置零) 构造白盒逃逸, 分别评估于
floor_fz (完全白盒: 对手知道 off-switch 与全部权重) 与 adaptive (迁移)。
若地板在"针对它自己构造"的攻击下仍不崩溃, 地板论证从"对迁移攻击鲁棒"
升级为"对地板感知 (floor-aware) 白盒对手鲁棒"。
用法: --attacks evasion,drift,evasion_fz (默认保持向后兼容 evasion,drift).

回答 TIFS 关心的两个问题:
  (a) 对抗触发是否提高崩溃率 (w3 F1 < 0.6 的种子比例, 以及"诱发崩溃"
      = clean 健康而受攻后 <0.6 的种子数);
  (b) 静态地板在攻击下是否保持 (floor retention: w3 均值 / 最小值 / 崩溃率).

输出 (与 experiments.py 同目录):
  TGM-Lite_对抗审计_<data><tag>_seeds_raw.csv   逐 (seed, model, attack, eps, variant) 行
  TGM-Lite_对抗审计_<data><tag>_summary.csv    按条件聚合 + 诱发崩溃统计

崩溃规则与主文一致: w3 F1 < 0.6 (AMLSim); 地板 = 同一检查点 t>=40 强制 s_t:=0.

示例 (AMLSim, 与主表同一协议, 约 1.5h @ RTX 2060 6GB):
  python audit_adversarial.py --data amlsim --models tgmlite,gcn \
      --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 \
      --eps 0.05,0.1,0.2 --tag _adv10
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import experiments as E


# ----------------------------------------------------------------------------- 训练(含 best 快照)
def train_with_snapshot(model, data, Ahats, X, split, args, device, val_freeze_from=-1):
    """与 experiments.train_supervised 完全相同的训练过程,
    额外在 val-AP 改进时快照 state_dict (供同一检查点的多条件评估)。"""
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
        ctx = E.build_context(model.proj(X), data, device)

    best = {"val_ap": -2.0}
    best_state = None
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        tot, nb = 0.0, 0
        for ck in E.chunks(train_times, args.chunk):
            Z = model.proj(X)
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

        model.freeze_mod_from = val_freeze_from   # fz40 协议: 冻结状态下做验证选择
        with torch.no_grad():
            Z = model.proj(X)
        val = E.infer_scores(model, Z, ctx, data, Ahats, val_times, device)
        va = E.average_precision(*val)
        if va != va:
            va = -1.0
        if va > best["val_ap"]:
            thr, _ = E.best_f1_threshold(*val)
            best = dict(val_ap=va, epoch=ep + 1, thr=thr)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % max(1, args.epochs // 5) == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss={tot/max(nb,1):.4f}  "
                  f"val_AP={va:.4f}  (best@{best.get('epoch','-')})")
    best["state"] = best_state
    best["ctx0"] = ctx          # 训练期缓存上下文 (stale-eval 控制组 / canonical 复现锚点)
    return best


# ----------------------------------------------------------------------------- 评估
@torch.no_grad()
def eval_scores(model, X_in, data, Ahats, times, device, ctx_override=None):
    """同一检查点在给定(可能被扰动的)特征下的测试分数。
    默认用当前投影重算 ctx (deployment-fresh); 传入训练期缓存 ctx 即为
    canonical 主表的 stale-eval 协议 (表 I / 表 VIII 复现锚点)。"""
    model.eval()
    Z = model.proj(X_in)
    ctx = ctx_override if ctx_override is not None else E.build_context(Z, data, device)
    return E.infer_scores(model, Z, ctx, data, Ahats, times, device)


def drift_blocks(data, split, y_t, s_t, thr):
    """与 run_one 相同的 3 段漂移 F1。"""
    test_times = np.array(split[2])
    blocks = np.array_split(test_times, 3)
    out = []
    for blk in blocks:
        full_mask_times = data["node_time"][np.isin(data["node_time"], split[2]) & data["labeled"]]
        sel = np.isin(full_mask_times, blk)
        if sel.sum() > 0 and y_t[sel].sum() > 0:
            _, _, f1b = E.prf_at(y_t[sel], s_t[sel], thr)
            out.append(float(f1b))
        else:
            out.append(float("nan"))
    return out


# ----------------------------------------------------------------------------- 攻击
@torch.no_grad()
def _rollout_states(model, ctx, upto, device):
    """no-grad 推演控制器, 返回 {t: s_t(detached)} (gcn 等静态控制器为 None)。"""
    state = model.controller.init_state(device)
    S = {}
    for t in range(1, upto + 1):
        s_t, state = model.controller.step(ctx[t], state)
        S[t] = s_t.detach() if isinstance(s_t, torch.Tensor) else None
    return S


def evasion_perturb(model, X, data, Ahats, ctx_clean, test_times, scale, eps, device,
                    zero_from=-1):
    """Myopic 白盒逃逸: 对每个测试窗口, 固定时序状态 s_t, 对非法节点的特征
    沿 -sign(grad logit) 方向移动 eps*scale (Linf, 特征标准差单位)。
    zero_from>=0 时, 在构造攻击时把 t>=zero_from 的 s_t 置零 —— 即让梯度
    完全穿过"已切除"的部署构型 (E5: 针对地板检查点/切除点的白盒对手)。"""
    X_adv = X.detach().clone()
    if eps <= 0:
        return X_adv
    with torch.no_grad():
        Z0 = model.proj(X)
    S = _rollout_states(model, ctx_clean, max(test_times), device)
    if zero_from >= 0:
        for t in S:
            if t >= zero_from and S[t] is not None:
                S[t] = torch.zeros_like(S[t])
    t_of = data["node_time"]
    scale_t = torch.from_numpy(scale.astype(np.float32)).to(device)
    for t in test_times:
        rows = np.where((t_of == t) & data["labeled"] & (data["y"] > 0.5))[0]
        if len(rows) == 0:
            continue
        x_sub = X[rows].detach().clone().requires_grad_(True)
        z_sub = model.proj(x_sub)
        Zg = Z0.clone()
        Zg[rows] = z_sub                     # index_put: 梯度可回传到 x_sub
        H = model.encode(Zg, S[t], Ahats[t - 1])
        logit = model.clf(H).squeeze(-1)[rows]
        model.zero_grad(set_to_none=True)
        logit.sum().backward()               # 逃逸: 压低非法节点分数
        X_adv[rows] = X[rows] - eps * scale_t * torch.sign(x_sub.grad)
    return X_adv.detach()


@torch.no_grad()
def drift_perturb(X, data, split, scale, eps, device):
    """无目标漂移触发: 测试窗口全部节点特征沿 sign(mu_test-mu_train) 放大 eps*scale。"""
    X_adv = X.detach().clone()
    if eps <= 0:
        return X_adv
    t_of = data["node_time"]
    te = torch.from_numpy(np.isin(t_of, split[2])).to(device)
    tr = torch.from_numpy(np.isin(t_of, split[0])).to(device)
    direction = torch.sign(X[te].mean(0) - X[tr].mean(0))
    scale_t = torch.from_numpy(scale.astype(np.float32)).to(device)
    X_adv[te] = X[te] + eps * scale_t * direction
    return X_adv.detach()


# ----------------------------------------------------------------------------- CSV
RAW_FIELDS = ["seed", "model", "attack", "eps", "variant",
              "ap", "w1", "w2", "w3", "thr", "best_epoch"]


def append_rows(rows, path):
    path = Path(path)
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(",".join(RAW_FIELDS) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in RAW_FIELDS) + "\n")


def done_keys(path):
    if not Path(path).exists():
        return set()
    import csv as _csv
    with open(path, encoding="utf-8") as f:
        return {(int(r["seed"]), r["model"], r["attack"],
                 round(float(r["eps"]), 6), r["variant"]) for r in _csv.DictReader(f)}


# ----------------------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser(description="TGM-Lite 对抗维度审计 (TIFS)")
    ap.add_argument("--data", default="amlsim", choices=["amlsim", "elliptic"])
    ap.add_argument("--amlsim_dir", default="auto")
    ap.add_argument("--elliptic_dir", default="auto")
    ap.add_argument("--window_hours", type=int, default=4)
    ap.add_argument("--window_mode", default="quantile", choices=["quantile", "time"])
    ap.add_argument("--n_windows", type=int, default=60)
    ap.add_argument("--amlsim_max_nodes", type=int, default=300000)
    ap.add_argument("--edge_fan", type=int, default=1)
    ap.add_argument("--models", default="tgmlite,gcn")
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
                    help="地板评估的冻结起点 (AMLSim canonical=40)")
    ap.add_argument("--eps", default="0.05,0.1,0.2",
                    help="扰动强度列表(特征标准差单位)")
    ap.add_argument("--attacks", default="evasion,drift",
                    help="逗号分隔: evasion(在自适应检查点构造,迁移到地板) / "
                         "drift(无目标漂移) / evasion_fz(E5: 在 fz40 地板检查点上、"
                         "按已切除构型构造的白盒逃逸, 评估于 floor_fz(白盒) 与 "
                         "adaptive(迁移))")
    ap.add_argument("--seeds", default="42,43,44,45,46,47,48,49,50,51")
    ap.add_argument("--tag", default="_adv10")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[环境] device={device}  torch={torch.__version__}")

    # ---- 数据 (与 experiments.main 同一入口; 划分种子固定 42) ----
    if args.data == "amlsim":
        data = E.load_amlsim(args.amlsim_dir, window_hours=args.window_hours,
                             max_nodes=args.amlsim_max_nodes, seed=42,
                             edge_fan=bool(args.edge_fan),
                             window_mode=args.window_mode, n_windows=args.n_windows)
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

    # ---- 与 experiments.main 一致: 训练区统计量标准化 (否则协议不等价!) ----
    data["train_eligible"] = np.ones(len(data["labeled"]), dtype=bool)
    tr_std = (data["node_time"] <= args.train_end) & data["train_eligible"]
    mu, sd = data["X"][tr_std].mean(0), data["X"][tr_std].std(0) + 1e-6
    data["X"] = ((data["X"] - mu) / sd).astype(np.float32)

    Ahats, _ = E.build_snapshots(data, device)
    X = torch.from_numpy(data["X"]).to(device)
    split = (list(range(1, args.train_end + 1)),
             list(range(args.train_end + 1, args.val_end + 1)),
             list(range(args.val_end + 1, data["T"] + 1)))
    eps_list = [float(x) for x in args.eps.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    models = args.models.split(",")

    # 特征尺度: 训练窗口节点的逐维标准差 (eps 的单位)
    tr_mask_np = np.isin(data["node_time"], split[0])
    scale = (X[tr_mask_np].std(0) + 1e-6).detach().cpu().numpy()

    raw_path = Path(f"TGM-Lite_对抗审计_{args.data}{args.tag}_seeds_raw.csv")
    skip = done_keys(raw_path)
    if skip:
        print(f"[续跑] 已完成单元格 (seed,model,attack,eps,variant): {len(skip)} 个, 自动跳过")

    for mode in models:
        for seed in seeds:
            all_variants = (["adaptive"] if mode == "gcn" else
                            ["adaptive", "floor", "adaptive_stale", "floor_fz"])
            atks = [a.strip() for a in args.attacks.split(",") if a.strip()]
            cells = [("clean", 0.0, v) for v in all_variants]
            for eps in eps_list:
                for atk in atks:
                    if atk == "evasion_fz":
                        # E5: 对手精确知道 t_exc 与地板部署, 直接在地板构型上构造;
                        # gcn 无控制器/无地板语义, 跳过
                        if mode == "gcn":
                            continue
                        cells += [(atk, eps, v) for v in ("floor_fz", "adaptive")
                                  if v in all_variants]
                    else:
                        cells += [(atk, eps, v) for v in all_variants]
            missing = [c for c in cells
                       if (seed, mode, c[0], round(float(c[1]), 6), c[2]) not in skip]
            if not missing:
                continue
            print(f"\n===== model={mode}  seed={seed}  (缺 {len(missing)}/{len(cells)} 格) =====")
            # evasion 攻击在自适应检查点上构造 -> 只要缺 evasion 格就需要自适应检查点
            need_adaptive = (any(v != "floor_fz" for _, _, v in missing)
                             or any(a == "evasion" for a, _, _ in missing))
            need_fz = any(v == "floor_fz" for _, _, v in missing)
            d = args.hidden
            model = model_fz = None
            best = best_fz = None

            # ---- 自适应检查点 (表 I 训练协议复刻) ----
            if need_adaptive:
                E.set_seed(seed)
                model = E.RiskNet(X.shape[1], d=d, r=args.rank, m=args.state,
                                  ctx_dim=2 * d + 3, mode=mode, dropout=args.dropout,
                                  state_norm=False, guard=False,
                                  freeze_mod_from=-1).to(device)
                E.pretrain(model, data, Ahats, X, None, split[0], args, device)
                best = train_with_snapshot(model, data, Ahats, X, split, args, device)
                model.load_state_dict(best["state"])
                model.eval()

            # ---- fz40 认证地板检查点 (表 VIII 协议复刻: 冻结状态下做验证选择) ----
            if need_fz:
                E.set_seed(seed)
                model_fz = E.RiskNet(X.shape[1], d=d, r=args.rank, m=args.state,
                                     ctx_dim=2 * d + 3, mode=mode, dropout=args.dropout,
                                     state_norm=False, guard=False,
                                     freeze_mod_from=-1).to(device)
                E.pretrain(model_fz, data, Ahats, X, None, split[0], args, device)
                best_fz = train_with_snapshot(model_fz, data, Ahats, X, split, args, device,
                                              val_freeze_from=args.floor_from)
                model_fz.load_state_dict(best_fz["state"])
                model_fz.eval()
                model_fz.freeze_mod_from = args.floor_from   # 评估全程冻结 (canonical)

            # ---- 攻击特征 (evasion 在自适应检查点上构造, 迁移到各变体;
            #      evasion_fz 在 fz40 地板检查点上按已切除构型构造; drift 与模型无关) ----
            X_ev, X_dr, X_evf = {}, {}, {}
            if any(a == "evasion" for a, _, _ in missing):
                model.freeze_mod_from = -1
                with torch.no_grad():
                    ctx_clean = E.build_context(model.proj(X), data, device)
                for eps in eps_list:
                    X_ev[eps] = evasion_perturb(model, X, data, Ahats, ctx_clean,
                                                split[2], scale, eps, device)
            if any(a == "evasion_fz" for a, _, _ in missing):
                # E5: 针对切除点的对手 —— 梯度穿过冻结后的部署构型
                # (t>=floor_from 的 s_t 置零), 使用地板自己的训练期缓存 ctx
                # (canonical floor_fz 协议), 回答"知道 off-switch 的白盒对手
                # 能否直接打穿地板"。
                for eps in eps_list:
                    X_evf[eps] = evasion_perturb(model_fz, X, data, Ahats,
                                                 best_fz["ctx0"], split[2],
                                                 scale, eps, device,
                                                 zero_from=args.floor_from)
            if any(a == "drift" for a, _, _ in missing):
                for eps in eps_list:
                    X_dr[eps] = drift_perturb(X, data, split, scale, eps, device)

            def _eval(var, X_in):
                """四个变体: adaptive=fresh评估; floor=推理时切除(fresh);
                adaptive_stale=canonical表I协议(缓存ctx); floor_fz=canonical表VIII协议."""
                if var in ("adaptive", "floor"):
                    model.freeze_mod_from = -1 if var == "adaptive" else args.floor_from
                    return eval_scores(model, X_in, data, Ahats, split[2], device)
                if var == "adaptive_stale":
                    model.freeze_mod_from = -1
                    return eval_scores(model, X_in, data, Ahats, split[2], device,
                                       ctx_override=best["ctx0"])
                return eval_scores(model_fz, X_in, data, Ahats, split[2], device,
                                   ctx_override=best_fz["ctx0"])

            # ---- 按缺失单元格评估 ----
            rows = []
            for atk, eps, var in missing:
                X_in = (X if atk == "clean" else
                        X_ev[eps] if atk == "evasion" else
                        X_evf[eps] if atk == "evasion_fz" else X_dr[eps])
                b = best_fz if var == "floor_fz" else best
                y_t, s_t = _eval(var, X_in)
                w = drift_blocks(data, split, y_t, s_t, b["thr"])
                rows.append(dict(seed=seed, model=mode, attack=atk, eps=eps,
                                 variant=var, ap=E.average_precision(y_t, s_t),
                                 w1=w[0], w2=w[1], w3=w[2], thr=b["thr"],
                                 best_epoch=b["epoch"]))
            append_rows(rows, raw_path)
            print(f"  [已保存] {len(rows)} 行 -> {raw_path.name}")

            # sanity: 与 canonical 主表对照 (协议一致性体检, 仅首个种子)
            if seed == seeds[0]:
                if mode == "gcn" and any(r["attack"] == "clean" for r in rows):
                    ap0 = [r for r in rows if r["attack"] == "clean"][0]["ap"]
                    flag = "OK" if abs(ap0 - 0.5922) < 0.05 else "!! 偏离 canonical (0.5922), 检查数据预处理 !!"
                    print(f"  [sanity] GCN clean AP={ap0:.4f} vs canonical 0.5922 -> {flag}")
                if mode == "tgmlite":
                    r_fz = [r for r in rows if r["attack"] == "clean" and r["variant"] == "floor_fz"]
                    if r_fz:
                        flag = "OK" if abs(r_fz[0]["w3"] - 0.6729) < 0.08 else "!! 偏离 canonical fz40 (0.6729) !!"
                        print(f"  [sanity] floor_fz clean w3={r_fz[0]['w3']:.4f} vs canonical 表VIII 0.6729 -> {flag}")
                    r_st = [r for r in rows if r["attack"] == "clean" and r["variant"] == "adaptive_stale"]
                    if r_st:
                        d_ap = abs(r_st[0]["ap"] - 0.5780); d_w3 = abs(r_st[0]["w3"] - 0.6942)
                        flag = "OK" if (d_ap < 0.05 and d_w3 < 0.08) else "!! 偏离 canonical 表I (AP 0.5780 / w3 0.6942) !!"
                        print(f"  [sanity] adaptive_stale clean AP={r_st[0]['ap']:.4f} w3={r_st[0]['w3']:.4f} "
                              f"vs canonical 表I 0.5780/0.6942 -> {flag}")
            del model, model_fz
            if device.type == "cuda":
                torch.cuda.empty_cache()

    summarize(raw_path, Path(f"TGM-Lite_对抗审计_{args.data}{args.tag}_summary.csv"))


def summarize(raw_path, out_path):
    import csv as _csv
    raw_path, out_path = Path(raw_path), Path(out_path)
    rows = list(_csv.DictReader(open(raw_path, encoding="utf-8")))
    if not rows:
        return
    for r in rows:
        for k in ["eps", "ap", "w1", "w2", "w3"]:
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
    # clean 参照: (seed, model, variant) -> clean w3
    clean = {(r["seed"], r["model"], r["variant"]): r["w3"]
             for r in rows if r["attack"] == "clean"}
    groups = {}
    for r in rows:
        groups.setdefault((r["model"], r["attack"], r["eps"], r["variant"]), []).append(r)

    out_fields = ["model", "attack", "eps", "variant", "n",
                  "ap_mean", "ap_std", "w3_mean", "w3_std", "w3_min",
                  "n_collapse_w3lt0.6", "n_induced_vs_clean", "mean_delta_w3_vs_clean"]
    out = []
    print("\n===== 对抗审计汇总 =====")
    for (mode, atk, eps, var) in sorted(groups):
        rs = groups[(mode, atk, eps, var)]
        w3 = np.array([r["w3"] for r in rs])
        ap_ = np.array([r["ap"] for r in rs])
        n_col = int((w3 < 0.6).sum())
        ind, deltas = 0, []
        for r in rs:
            c = clean.get((r["seed"], r["model"], r["variant"]))
            if c is None or atk == "clean":
                continue
            deltas.append(r["w3"] - c)
            if c >= 0.6 and r["w3"] < 0.6:
                ind += 1
        rec = dict(model=mode, attack=atk, eps=eps, variant=var, n=len(rs),
                   ap_mean=f"{ap_.mean():.4f}", ap_std=f"{np.std(ap_, ddof=0):.4f}",
                   w3_mean=f"{w3.mean():.4f}", w3_std=f"{np.std(w3, ddof=0):.4f}",
                   w3_min=f"{w3.min():.4f}",
                   **{"n_collapse_w3lt0.6": n_col, "n_induced_vs_clean": ind,
                      "mean_delta_w3_vs_clean": f"{np.mean(deltas):.4f}" if deltas else ""})
        out.append(rec)
        print(f"{mode:8s} {atk:8s} eps={eps:<5} {var:9s} | w3 {rec['w3_mean']}±{rec['w3_std']} "
              f"min {rec['w3_min']} | 崩溃 {n_col}/{len(rs)} 诱发 {ind} | Δw3 {rec['mean_delta_w3_vs_clean']}")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        wcsv = _csv.DictWriter(f, fieldnames=out_fields)
        wcsv.writeheader()
        wcsv.writerows(out)
    print(f"[已保存] 汇总 -> {out_path.name}")


if __name__ == "__main__":
    main()
