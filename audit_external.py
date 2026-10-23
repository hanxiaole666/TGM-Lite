# -*- coding: utf-8 -*-
"""
audit_external.py — 外部自适应时序图模型的同协议稳定性审计
=========================================================

动机 (审稿意见 M3): 主文稿的 10-seed 稳定性审计只覆盖 TGM-Lite 自身及其内部
消融。本脚本把**同一套审计协议**(相同数据划分、相同训练轮数与优化器、相同
截断 BPTT、相同验证集早选、相同漂移窗口指标、相同 10 个种子 42-51)原样施加到
三类**外部的**自适应时序图机制上, 检验"漂移尾段最坏种子崩溃"是否为这一类
模型的共性, 而非 TGM-Lite 独有:

  1. evolvegcn_full — 忠实的 EvolveGCN-O: 两层 GCN 的**权重矩阵整体**作为 GRU
     的隐状态随时间演化 (weights-as-hidden-state, 零输入), 不使用低秩调制、
     不使用上下文向量。对比点: 权重级演化, 但无数据驱动的适应策略。
  2. dysat — DySAT 式: 每个时间片独立的图自注意力 (GAT, 结构层) + 逐节点的
     时间自注意力 (temporal self-attention)。对比点: 表示级适应, 注意力机制。
  3. graphmamba — SSM 作为骨干 (而非权重控制器): 静态门控 GCN 主干 +
     **逐节点**选择性状态空间模型 h_t = σ(W_a H_t)⊙h_{t-1} + σ(W_b H_t)⊙tanh(W_in H_t)。
     对比点: 与 TGM-Lite 同族的选择性门控, 但作用于节点表示而非全局权重,
     用于检验"崩溃是否由 SSM 本身引起"。

协议一致性说明:
  * 数据加载、标准化(仅用训练窗口统计)、时间划分 (AMLSim 1-36/37-48/49-60,
    Elliptic 1-34/35-39/40-49)、BPTT chunk (AMLSim 自适应为 4)、优化器
    (Adam, lr=5e-3, wd=1e-5)、pos_weight∈[1,20]、梯度裁剪 5.0、
    验证集 AP 早选与阈值选择、漂移窗口 w1/w2/w3 = 验证最优阈值下的 F1,
    均直接复用 experiments.py 的函数或逐行镜像其实现。
  * 多种子协议: 数据相关随机性固定为 seed=42, 仅模型初始化随种子变化。
  * 掩码重建预训练对外部模型同样启用(--pretrain_epochs), 作用于其结构编码器。
  * 切除 (excision) 地板的对应物 (--freeze_mod_from T_exc): 从窗口 T_exc 起
    关闭各模型的时间适应。注意: 仅 evolvegcn_full 的推理期冻结是合法地板
    (分类器在每个窗口都见过当片权重, 冻结保持在训练分布内; 10-seed 实测
    健康种子的地板≈自适应)。dysat/graphmamba 的旁路路径从未被训练,
    推理期切除使分类器面对未见分布 (实测 F1=0), 不构成合法地板 —— 
    这两族改用**从零训练的静态对照** (--models dysat_static,graphmamba_static),
    崩溃规则 = 同种子静态对照 w3 − 自适应 w3 > 0.1。
  * 资源限制 (RTX 2060 6GB): dysat 的时间注意力与 graphmamba 的逐节点 SSM
    按节点分块 (--node_block) 并配合梯度检查点; graphmamba 的逐节点状态维
    固定为 hidden d (64) 以控制显存, 状态维度不是本审计的变量。

用法:
  # 冒烟测试 (CPU 可跑, 约 1 分钟)
  python audit_external.py --data synthetic --models all --epochs 3 --pretrain_epochs 1

  # 正式审计 (AMLSim, 10 种子)
  python audit_external.py --data amlsim --amlsim_dir D:\\TGM-Lite\\Data\\ealtman2019\\HI-Small_Trans.csv \\
      --models evolvegcn_full,dysat,graphmamba --seeds 42,43,44,45,46,47,48,49,50,51 --tag _ext10

  # 地板: evolvegcn_full 用推理期冻结 (同种子重跑, --freeze_mod_from 40);
  #       dysat/graphmamba 用从零训练的静态对照:
  python audit_external.py --data amlsim --models dysat_static,graphmamba_static \
      --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _static10

  # 崩溃规则汇总 + 与地板配对分析 (自动读 --floor_csv 指定的地板原始 CSV)
  python audit_external.py ... --floor_csv TGM-Lite_审计结果_supervised_amlsim_ext10fz40_seeds_raw.csv

输出: 与 experiments.py 兼容的 per-seed 原始 CSV (多一列 seed) 与聚合 CSV,
以及崩溃规则 (Δw3>0.1 vs 同种子地板 / w3 F1<0.6) 汇总表。
长任务防崩: 每个 (模型, 种子) 完成即追加落盘; 重跑同一命令自动跳过已有组合
(删除原始 CSV 可强制全新运行)。
"""

import argparse
import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 复用 experiments.py 的数据管线与指标, 保证协议逐字一致 ----
from experiments import (
    set_seed, make_synthetic, load_amlsim, load_elliptic, ensure_elliptic,
    build_snapshots, build_context, chunks,
    roc_auc, average_precision, best_f1_threshold, prf_at, precision_at_k,
    RiskNet, aggregate_seeds, print_agg_table, save_agg_csv,
)

EXT_MODELS = ["evolvegcn_full", "dysat", "graphmamba"]
# 静态对照 (训练出来的地板, 非推理期切除): dysat/graphmamba 的旁路路径从未被
# 训练过, 推理期切除会让分类器面对未见分布 (10-seed 实测 F1=0), 因此这两族
# 的合法地板是从零训练的静态对应物; evolvegcn_full 的推理期权重冻结仍合法
# (分类器在每个窗口都见过当片权重, 冻结保持在分布内)。
FLOOR_MODELS = ["dysat_static", "graphmamba_static"]


# ==============================================================================
# 1. EvolveGCN-O (full): 权重矩阵整体作为 GRU 隐状态演化
# ==============================================================================

class EvolveFullController(nn.Module):
    """EvolveGCN-O 式控制器: 两层 GCN 权重 W1,W2 分别是两个 GRUCell 的隐状态,
    输入为零 (忠实于 EvolveGCN-O 的"权重即隐状态、无输入"设定)。
    step 返回把两个 d×d 权重矩阵拉直拼接的 2d² 维向量, 作为 encode 的 s_t
    传入 (兼容 experiments 的梯度检查点: s_t 必须是显式张量参数)。"""

    def __init__(self, net, d):
        super().__init__()
        # 引用宿主的初始权重作为演化起点 (Parameter 引用, 非 Module, 无循环引用;
        # torch 的 parameters() 按身份去重, 不会重复计数)
        self.W0_init = net.W0
        self.W2_init = net.W2
        self.d = d
        self.gru1 = nn.GRUCell(d, d)        # 演化第 1 层权重 (按行: d 个 d 维行向量)
        self.gru2 = nn.GRUCell(d, d)        # 演化第 2 层权重

    def init_state(self, device):
        return (self.W0_init, self.W2_init)  # 以可学习初始权重为起点

    def step(self, c, state):
        del c                               # EvolveGCN-O: 无数据驱动输入
        h1, h2 = state
        z = torch.zeros(self.d, self.d, device=h1.device)
        h1 = self.gru1(z, h1)
        h2 = self.gru2(z, h2)
        return torch.cat([h1.reshape(-1), h2.reshape(-1)]), (h1, h2)


class EvolveFullNet(RiskNet):
    """RiskNet 子类: encode 从 s_t 还原演化后的 (W1_t, W2_t); 其余训练/评估
    流程 (pretrain / train_supervised / infer_scores) 与 experiments.py 完全一致。"""

    def __init__(self, feat_dim, d=64, dropout=0.3, freeze_mod_from=-1):
        # mode="gcn": 使用图、不使用 U/V 调制 (StaticController 占位后替换)
        super().__init__(feat_dim, d=d, r=1, m=1, ctx_dim=1, mode="gcn",
                         dropout=dropout, freeze_mod_from=-1)
        self.freeze_mod_from = freeze_mod_from
        self.controller = EvolveFullController(self, d)

    def encode(self, Z, s_t, Ahat):
        d = self.d
        if s_t is not None:
            W1 = s_t[: d * d].view(d, d)
            W2 = s_t[d * d:].view(d, d)
        else:
            W1, W2 = self.W0, self.W2
        lam1, lam2 = torch.sigmoid(self.g1), torch.sigmoid(self.g2)
        H = Z @ W1
        H = (1.0 - lam1) * H + lam1 * torch.sparse.mm(Ahat, H)
        H = F.relu(H)
        H2 = H @ W2
        H2 = (1.0 - lam2) * H2 + lam2 * torch.sparse.mm(Ahat, H2)
        return F.relu(H2)

    def forward_logits(self, Z, ctx, times, Ahats, device):
        """与 RiskNet.forward_logits 相同, 但 freeze 语义 = 权重冻结不再演化
        (而不是把 s_t 清零——那对 EvolveGCN 意味着把 GCN 权重清零)。"""
        state = self.controller.init_state(device)
        out, held = {}, None
        for t in times:
            if self.freeze_mod_from >= 0 and t >= self.freeze_mod_from:
                if held is None:
                    held = state            # 进入冻结: 固定当前权重
                s_t = torch.cat([held[0].reshape(-1), held[1].reshape(-1)])
            else:
                s_t, state = self.controller.step(None, state)
            H = self.encode(Z, s_t, Ahats[t - 1])
            out[t] = self.clf(F.dropout(H, self.dropout, self.training)).squeeze(-1)
        return out, state


# ==============================================================================
# 2. 稀疏 GAT 层 (纯 PyTorch, 无 torch_geometric 依赖) — DySAT 结构层
# ==============================================================================

class SparseGAT(nn.Module):
    """多头图自注意力 (单个时间片): e_ij = LeakyReLU(a_l·Wh_i + a_r·Wh_j),
    对每个目标节点做边 softmax。输入为对称化 + 自环的边列表。"""

    def __init__(self, d, heads=4):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.W = nn.Linear(d, d, bias=False)
        self.a_l = nn.Parameter(torch.randn(heads, self.dh) * 0.1)
        self.a_r = nn.Parameter(torch.randn(heads, self.dh) * 0.1)

    def forward(self, Z, edge_src, edge_dst):
        N = Z.shape[0]
        Wh = self.W(Z).view(N, self.h, self.dh)
        s, d = edge_src, edge_dst
        e = (Wh[s] * self.a_l).sum(-1) + (Wh[d] * self.a_r).sum(-1)  # (E, heads)
        e = F.leaky_relu(e, 0.2)
        # 逐目标节点 softmax (数值稳定: 先减 scatter max)
        m = torch.full((N, self.h), float("-inf"), device=Z.device)
        m.scatter_reduce_(0, d.unsqueeze(-1).expand_as(e), e, reduce="amax")
        p = torch.exp(e - m[d])
        denom = torch.zeros(N, self.h, device=Z.device)
        denom.index_add_(0, d, p)
        alpha = p / denom[d].clamp_min(1e-12)
        out = torch.zeros(N, self.h, self.dh, device=Z.device)
        out.index_add_(0, d, alpha.unsqueeze(-1) * Wh[s])
        return F.elu(out.reshape(N, self.h * self.dh))


def build_edge_lists(data, device):
    """与 build_snapshots 相同的边归属规则 (边归源节点时间片, 对称化+自环),
    返回 {t: (src, dst)} 供 GAT 使用。"""
    T = data["T"]
    arangeN = torch.arange(data["X"].shape[0], device=device)
    t_src = data["node_time"][data["edge_src"]]
    edges = {}
    for t in range(1, T + 1):
        sel = np.where(t_src == t)[0]
        s = torch.from_numpy(data["edge_src"][sel]).to(device)
        d = torch.from_numpy(data["edge_dst"][sel]).to(device)
        edges[t] = (torch.cat([s, d, arangeN]), torch.cat([d, s, arangeN]))
    return edges


# ==============================================================================
# 3. DySAT: 结构自注意力 (逐片 GAT) + 时间自注意力 (逐节点跨片)
# ==============================================================================

class TemporalSelfAttention(nn.Module):
    """逐节点时间自注意力 (手写 MHA)。序列长 = BPTT chunk (4/8), 注意力矩阵
    极小, 手写实现零开销; 同时规避 torch<=2.2 Windows/CUDA 版 SDPA 内核在
    batch×heads 超过 65535 时的 grid 配置限制 (AMLSim 300K 节点实测触发)。
    参数量与 nn.MultiheadAttention 完全相同 (4d²+4d)。"""

    def __init__(self, d, heads=4):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)

    def forward(self, x):                   # x: (B, k, d)
        B, k, d = x.shape
        q = self.q(x).view(B, k, self.h, self.dh).transpose(1, 2)
        kk = self.k(x).view(B, k, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, k, self.h, self.dh).transpose(1, 2)
        p = torch.softmax(q @ kk.transpose(-2, -1) / math.sqrt(self.dh), dim=-1)
        return self.o((p @ v).transpose(1, 2).reshape(B, k, d))


def sinusoidal_pe(times, d, device):
    """按绝对时间片 t 的正弦位置编码 (与 build_context 的时间编码同源)。"""
    pe = torch.zeros(len(times), d, device=device)
    pos = torch.tensor(times, dtype=torch.float32, device=device).unsqueeze(-1)
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32)
                    * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class DySATNet(nn.Module):
    """DySAT 式审计实现: 结构层 = 每片独立 SparseGAT; 时间层 = 逐节点对
    最近 chunk 个时间片的结构嵌入做自注意力 (窗口长度 = BPTT chunk, 与循环
    模型的截断视界一致)。freeze_mod_from>=0 时, 冻结窗口起跳过时间注意力,
    直接使用当片结构嵌入 (切除地板的对应物)。"""

    def __init__(self, feat_dim, d=64, heads=4, dropout=0.3, chunk=8,
                 freeze_mod_from=-1, static=False):
        super().__init__()
        self.d, self.chunk = d, chunk
        self.freeze_mod_from = freeze_mod_from
        self.static = static                # 训练静态对照 (地板): 从不用时间注意力
        self.proj = nn.Linear(feat_dim, d)
        self.structural = SparseGAT(d, heads)
        self.temporal = None if static else TemporalSelfAttention(d, heads)
        self.clf = nn.Linear(d, 1)
        self.decoder = nn.Linear(d, feat_dim)   # 预训练用
        self.dropout = dropout

    def _struct_seq(self, Z, edges, times):
        """逐片结构嵌入, 返回 (N, k, d)。每片包在梯度检查点里以控显存。"""
        Hs = []
        for t in times:
            s, d_ = edges[t]
            Hs.append(torch.utils.checkpoint.checkpoint(
                self.structural, Z, s, d_, use_reentrant=False))
        return torch.stack(Hs, dim=1)           # (N, k, d)

    def _temporal(self, Hseq, times, node_block):
        """逐节点时间自注意力, 按节点分块 + 梯度检查点。返回 (N, k, d)。"""
        pe = sinusoidal_pe(times, self.d, Hseq.device)
        N = Hseq.shape[0]
        outs = []
        for i in range(0, N, node_block):
            hb = Hseq[i:i + node_block] + pe
            eb = torch.utils.checkpoint.checkpoint(
                self.temporal, hb, use_reentrant=False)
            outs.append(eb)
        return torch.cat(outs, dim=0)

    def forward_chunk(self, Z, edges, times, node_block):
        """一个时间 chunk 的前向, 返回 {t: logits (N,)}。"""
        Hseq = self._struct_seq(Z, edges, times)
        E = None if self.static else self._temporal(Hseq, times, node_block)
        out = {}
        for j, t in enumerate(times):
            if self.static or (self.freeze_mod_from >= 0 and t >= self.freeze_mod_from):
                emb = Hseq[:, j, :]             # 静态对照 / 切除: 仅当片结构嵌入
            else:
                emb = E[:, j, :]
            out[t] = self.clf(F.dropout(emb, self.dropout, self.training)).squeeze(-1)
        return out

    @torch.no_grad()
    def forward_range(self, Z, edges, times, node_block):
        """评估: 从 t=1 起按 chunk 连续前向 (历史仅作注意力上下文, 无标签泄漏)。"""
        self.eval()
        out = {}
        for ck in chunks(times, self.chunk):
            out.update(self.forward_chunk(Z, edges, list(ck), node_block))
        return out


# ==============================================================================
# 4. GraphMamba (SSM-as-backbone): 静态主干 + 逐节点选择性 SSM
# ==============================================================================

class GraphMambaNet(nn.Module):
    """SSM 作为骨干的审计实现: 静态门控 GCN 主干 (与 TGM-Lite 相同的 proj→
    2 层门控 GCN, 权重不随时间变化) 产出每片节点表示 H_t; 然后逐节点施加
    与 TGM-Lite 控制器同构的选择性门控 SSM:
        h_t = σ(W_a H_t)⊙h_{t-1} + σ(W_b H_t)⊙tanh(W_in H_t)
    分类器作用于 h_t。与 TGM-Lite 的差别只在 SSM 的作用对象 (节点表示 vs
    全局权重)。逐节点状态维 = d (显存考虑)。freeze_mod_from>=0 时, 冻结
    窗口起绕过 SSM, 直接分类 H_t (切除地板的对应物)。"""

    def __init__(self, feat_dim, d=64, dropout=0.3, chunk=8, freeze_mod_from=-1,
                 static=False):
        super().__init__()
        self.d, self.chunk = d, chunk
        self.freeze_mod_from = freeze_mod_from
        self.static = static                # 训练静态对照 (地板): 从不启用 SSM
        self.proj = nn.Linear(feat_dim, d)
        self.W0 = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W0)
        self.W2 = nn.Parameter(torch.empty(d, d))
        nn.init.xavier_uniform_(self.W2)
        self.g1 = nn.Parameter(torch.tensor(-2.0))
        self.g2 = nn.Parameter(torch.tensor(-2.0))
        if static:
            self.gate_a = self.gate_b = self.inp = None
        else:
            self.gate_a = nn.Linear(d, d)
            self.gate_b = nn.Linear(d, d)
            self.inp = nn.Linear(d, d)
        self.clf = nn.Linear(d, 1)
        self.decoder = nn.Linear(d, feat_dim)   # 预训练用
        self.dropout = dropout

    def trunk(self, Z, Ahat):
        lam1, lam2 = torch.sigmoid(self.g1), torch.sigmoid(self.g2)
        H = Z @ self.W0
        H = (1.0 - lam1) * H + lam1 * torch.sparse.mm(Ahat, H)
        H = F.relu(H)
        H2 = H @ self.W2
        H2 = (1.0 - lam2) * H2 + lam2 * torch.sparse.mm(Ahat, H2)
        return F.relu(H2)

    def ssm_step(self, H_t, h, node_block):
        """逐节点 SSM 一步, 按节点分块 + 梯度检查点。h: (N, d) 或 None。"""
        N = H_t.shape[0]
        if h is None:
            h = torch.zeros(N, self.d, device=H_t.device)
        outs = []
        for i in range(0, N, node_block):
            def _f(x, hp):
                a = torch.sigmoid(self.gate_a(x))
                b = torch.sigmoid(self.gate_b(x))
                return a * hp + b * torch.tanh(self.inp(x))
            outs.append(torch.utils.checkpoint.checkpoint(
                _f, H_t[i:i + node_block], h[i:i + node_block],
                use_reentrant=False))
        return torch.cat(outs, dim=0)

    def forward_chunk(self, Z, Ahats, times, node_block, h0=None):
        """一个时间 chunk 的前向; 状态 h 在 chunk 内逐片携带, 返回 {t: logits} 与末态。"""
        h = h0
        out = {}
        for t in times:
            H_t = torch.utils.checkpoint.checkpoint(
                self.trunk, Z, Ahats[t - 1], use_reentrant=False)
            if not self.static:
                h = self.ssm_step(H_t, h, node_block)
            emb = H_t if (self.static or
                          (self.freeze_mod_from >= 0 and t >= self.freeze_mod_from)) else h
            out[t] = self.clf(F.dropout(emb, self.dropout, self.training)).squeeze(-1)
        return out, h

    @torch.no_grad()
    def forward_range(self, Z, Ahats, times, node_block):
        """评估: 从 t=1 起连续前向, 状态跨全程携带 (与 RiskNet 推理语义一致)。"""
        self.eval()
        out, h = {}, None
        for ck in chunks(times, self.chunk):
            o, h = self.forward_chunk(Z, Ahats, list(ck), node_block, h0=h)
            out.update(o)
        return out


# ==============================================================================
# 5. 序列模型的训练 / 评估循环 —— 逐行镜像 experiments.train_supervised 的协议
# ==============================================================================

def pretrain_seq(model, data, Z_of, forward_embed, times, args, device):
    """掩码特征重建预训练 (与 experiments.pretrain 相同的目标与轮数),
    作用于结构编码器: forward_embed(t) 返回该片全部节点的 d 维嵌入。"""
    if args.pretrain_epochs <= 0:
        return
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    N, Fd = data["X"].shape
    for ep in range(args.pretrain_epochs):
        model.train()
        opt.zero_grad()
        mask = torch.rand(N, Fd, device=device) < 0.15
        tot, nb = 0.0, 0
        for ck in chunks(times, args.chunk):
            Z = Z_of(mask)                  # 每个 chunk 重建计算图
            loss = 0.0
            for t in ck:
                H = forward_embed(Z, t)
                nodes = torch.from_numpy(data["node_time"] == t).to(device)
                m_sub = mask[nodes]
                if m_sub.any():
                    rec = model.decoder(H[nodes])
                    loss = loss + F.mse_loss(rec[m_sub],
                                             torch.from_numpy(data["X"]).to(device)[nodes][m_sub])
            if isinstance(loss, torch.Tensor):
                (loss / len(ck)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                tot += float(loss.detach()); nb += 1
        opt.step()
        print(f"  [预训练] epoch {ep+1}/{args.pretrain_epochs}  recon_loss={tot/max(nb,1):.4f}")


def train_seq(model, data, forward_train, forward_eval, split, args, device):
    """与 experiments.train_supervised 完全相同的协议: Adam(lr, wd=1e-5),
    pos_weight clamp [1,20], chunk 截断 BPTT, 梯度裁剪 5.0, 验证集 AP 早选,
    验证最优 F1 阈值。forward_train(times_ck) -> {t: logits}; 
    forward_eval(times) -> {t: logits} (无梯度, 含历史预热)。"""
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

    def collect(logits, times):
        mask = np.isin(data["node_time"], times) & data["labeled"]
        idx = np.where(mask)[0]
        t_of = data["node_time"]
        s = np.zeros(len(idx), dtype=np.float64)
        for j, g in enumerate(idx):
            s[j] = float(logits[t_of[g]][g])
        return data["y"][mask], s

    best = {"val_ap": -2.0}
    t_epoch = []
    for ep in range(args.epochs):
        model.train()
        t0 = time.time()
        opt.zero_grad()
        tot, nb = 0.0, 0
        for ck in chunks(train_times, args.chunk):
            logits = forward_train(list(ck))
            loss = 0.0
            for t in ck:
                m_t = labeled_t & (torch.from_numpy(data["node_time"] == t).to(device))
                if m_t.any():
                    loss = loss + F.binary_cross_entropy_with_logits(
                        logits[t][m_t], y[m_t], pos_weight=pos_weight)
            if isinstance(loss, torch.Tensor):
                (loss / len(ck)).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                tot += float(loss.detach()); nb += 1
        opt.step()
        t_epoch.append(time.time() - t0)

        with torch.no_grad():
            va_logits = forward_eval(val_times)
            te_logits = forward_eval(test_times)
        val = collect(va_logits, val_times)
        tst = collect(te_logits, test_times)
        va = average_precision(*val)
        if va != va:
            va = -1.0
        if va > best["val_ap"]:
            thr, _ = best_f1_threshold(*val)
            p, r, f1 = prf_at(*tst, thr)
            best = dict(val_ap=va, epoch=ep + 1, thr=thr,
                        auc=roc_auc(*tst), ap=average_precision(*tst),
                        p=p, r=r, f1=f1, p100=precision_at_k(*tst, 100),
                        test_scores=tst)
        if (ep + 1) % max(1, args.epochs // 5) == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  loss={tot/max(nb,1):.4f}  "
                  f"val_AP={va:.4f}  (best@{best.get('epoch','-')})")
    best["epoch_time"] = float(np.mean(t_epoch))
    return best


# ==============================================================================
# 6. 统一运行器: 与 experiments.run_one 相同的返回字段与漂移窗口计算
# ==============================================================================

def drift_blocks_f1(data, split, y_t, s_t, thr):
    """与 run_one 相同: 测试期切 3 个连续窗口, 在验证阈值下分别算 F1。"""
    test_times = np.array(split[2])
    blocks = np.array_split(test_times, 3)
    out = []
    for blk in blocks:
        full_mask_times = data["node_time"][np.isin(data["node_time"], split[2])
                                            & data["labeled"]]
        sel = np.isin(full_mask_times, blk)
        if sel.sum() > 0 and y_t[sel].sum() > 0:
            _, _, f1b = prf_at(y_t[sel], s_t[sel], thr)
            out.append(f1b)
        else:
            out.append(float("nan"))
    return out


def run_one_ext(mode, data, Ahats, X, split, args, device, edges=None):
    set_seed(args.seed)
    d = args.hidden

    if mode == "evolvegcn_full":
        # 直接复用 experiments 的 pretrain/train_supervised/infer_scores, 协议天然一致
        from experiments import pretrain, train_supervised, infer_scores
        model = EvolveFullNet(X.shape[1], d=d, dropout=args.dropout,
                              freeze_mod_from=args.freeze_mod_from).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        pretrain(model, data, Ahats, X, None, split[0], args, device)
        best = train_supervised(model, data, Ahats, X, None, split, args, device)

        from experiments import infer_scores as _inf
        model.eval()
        lat_t0 = time.time()
        with torch.no_grad():
            _inf(model, model.proj(X), None, data, Ahats, split[2], device)
        lat = (time.time() - lat_t0) * 1000.0
        y_t, s_t = best["test_scores"]
        drift = drift_blocks_f1(data, split, y_t, s_t, best["thr"])
        peak = (torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda" else 0.0)
        return dict(model=mode, params=n_params, auc=best["auc"], ap=best["ap"],
                    f1=best["f1"], p=best["p"], r=best["r"], p100=best["p100"],
                    drift=drift, epoch_s=best["epoch_time"], peak_mb=peak,
                    lat_ms=lat, best_epoch=best["epoch"],
                    val_ap=best["val_ap"])

    # ---- dysat / graphmamba (+ 静态对照) ----
    static = mode.endswith("_static")
    cls = {"dysat": DySATNet, "graphmamba": GraphMambaNet,
           "dysat_static": DySATNet, "graphmamba_static": GraphMambaNet}[mode]
    model = cls(X.shape[1], d=d, dropout=args.dropout, chunk=args.chunk,
                freeze_mod_from=args.freeze_mod_from,
                static=static).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    nb = args.node_block

    X_t = torch.from_numpy(data["X"]).to(device)

    if mode.startswith("dysat"):
        def forward_train(ck):
            Z = model.proj(X_t)
            return model.forward_chunk(Z, edges, ck, nb)

        def forward_eval(times):
            all_times = list(range(1, max(times) + 1))
            Z = model.proj(X_t)
            logits = model.forward_range(Z, edges, all_times, nb)
            return {t: logits[t] for t in all_times}

        def embed_fn(Z, t):
            s, dd = edges[t]
            return model.structural(Z, s, dd)
    else:
        def forward_train(ck):
            Z = model.proj(X_t)
            out, _ = model.forward_chunk(Z, Ahats, ck, nb)
            return out

        def forward_eval(times):
            all_times = list(range(1, max(times) + 1))
            Z = model.proj(X_t)
            logits = model.forward_range(Z, Ahats, all_times, nb)
            return {t: logits[t] for t in all_times}

        def embed_fn(Z, t):
            return model.trunk(Z, Ahats[t - 1])

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # 预训练: 掩码重建, 作用于结构编码器 (Z 的掩码在输入特征上)
    Xfull = data["X"]

    def Z_of(mask):
        Xm = X_t.clone()
        Xm[mask] = 0.0
        return model.proj(Xm)
    pretrain_seq(model, data, Z_of, embed_fn, split[0], args, device)

    best = train_seq(model, data, forward_train, forward_eval, split, args, device)

    # 推理时延
    model.eval()
    t0 = time.time()
    with torch.no_grad():
        forward_eval(split[2])
    lat = (time.time() - t0) * 1000.0

    y_t, s_t = best["test_scores"]
    drift = drift_blocks_f1(data, split, y_t, s_t, best["thr"])
    peak = (torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda" else 0.0)
    return dict(model=mode, params=n_params, auc=best["auc"], ap=best["ap"],
                f1=best["f1"], p=best["p"], r=best["r"], p100=best["p100"],
                drift=drift, epoch_s=best["epoch_time"], peak_mb=peak,
                lat_ms=lat, best_epoch=best["epoch"],
                    val_ap=best["val_ap"])


# ==============================================================================
# 7. 输出: per-seed 原始 CSV (含 seed 列) + 崩溃规则汇总 + 地板配对
# ==============================================================================

CSV_KEYS = ["seed", "model", "params", "auc", "ap", "f1", "p", "r", "p100",
            "drift_w1", "drift_w2", "drift_w3", "epoch_s", "peak_mb",
            "lat_ms", "best_epoch", "val_ap"]


def save_rows_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(CSV_KEYS)
        for r in rows:
            w.writerow([r.get("seed", ""), r["model"], r["params"],
                        f"{r['auc']:.4f}", f"{r['ap']:.4f}", f"{r['f1']:.4f}",
                        f"{r['p']:.4f}", f"{r['r']:.4f}", f"{r['p100']:.4f}",
                        *(f"{x:.4f}" for x in r["drift"]),
                        f"{r['epoch_s']:.2f}", f"{r['peak_mb']:.0f}",
                        f"{r['lat_ms']:.1f}", r["best_epoch"],
                        f"{r['val_ap']:.4f}"])
    print(f"[保存] 结果已写入 {path}")


def append_rows_csv(rows, path):
    """增量追加: 每个种子跑完立即落盘, 长任务崩溃/中断不丢已完成种子。"""
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(CSV_KEYS)
        for r in rows:
            w.writerow([r.get("seed", ""), r["model"], r["params"],
                        f"{r['auc']:.4f}", f"{r['ap']:.4f}", f"{r['f1']:.4f}",
                        f"{r['p']:.4f}", f"{r['r']:.4f}", f"{r['p100']:.4f}",
                        *(f"{x:.4f}" for x in r["drift"]),
                        f"{r['epoch_s']:.2f}", f"{r['peak_mb']:.0f}",
                        f"{r['lat_ms']:.1f}", r["best_epoch"],
                        f"{r['val_ap']:.4f}"])


def load_rows_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({"seed": int(r["seed"]), "model": r["model"],
                         "ap": float(r["ap"]),
                         "w3": float(r["drift_w3"])})
    return rows


def load_full_rows_csv(path):
    """完整读回原始 CSV (续跑汇总用): 字段与 aggregate_seeds/collapse_summary
    的期望一致 (drift = [w1, w2, w3])。"""
    num = ["params", "auc", "ap", "f1", "p", "r", "p100", "epoch_s",
           "peak_mb", "lat_ms", "best_epoch"]
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            row = {"seed": int(r["seed"]), "model": r["model"],
                   "drift": [float(r["drift_w1"]), float(r["drift_w2"]),
                             float(r["drift_w3"])]}
            for k_ in num:
                row[k_] = float(r[k_])
            row["params"] = int(row["params"]); row["best_epoch"] = int(row["best_epoch"])
            rows.append(row)
    return rows


def collapse_summary(all_rows, floor_csv=None, collapse_thr=0.6, delta=0.1):
    """崩溃规则汇总: (a) w3 F1 < collapse_thr 的硬崩溃计数; (b) 若提供地板
    CSV, 按 (model, seed) 配对, 统计 Δw3 > delta 的崩溃(与主文稿规则一致)。"""
    print(f"\n{'='*78}\n崩溃规则汇总 (w3 = 尾窗 F1, 验证最优阈值)\n{'='*78}")
    by_model = {}
    for r in all_rows:
        by_model.setdefault(r["model"], []).append(r)
    floor = {}
    if floor_csv and os.path.exists(floor_csv):
        for r in load_rows_csv(floor_csv):
            fm = r["model"][:-len("_static")] if r["model"].endswith("_static") else r["model"]
            floor[(fm, r["seed"])] = r["w3"]
        print(f"[配对] 已加载地板 CSV: {floor_csv} ({len(floor)} 行)")
    elif floor_csv:
        print(f"[警告] 地板 CSV 不存在: {floor_csv}, 跳过配对分析")

    hdr = f"{'model':<16}{'n':>4}{'w3 mean±std':>20}{'min w3':>9}{'w3<0.6':>8}{'Δw3>0.1':>9}"
    print(hdr); print("-" * len(hdr))
    for m, rs in by_model.items():
        w3 = np.array([r["drift"][2] for r in rs], dtype=np.float64)
        w3 = w3[w3 == w3]
        hard = sorted(r.get("seed") for r in rs
                      if r["drift"][2] == r["drift"][2] and r["drift"][2] < collapse_thr)
        n_pair, rescued = 0, []
        for r in rs:
            key = (m, r.get("seed"))
            if key in floor:
                n_pair += 1
                if floor[key] - r["drift"][2] > delta:
                    rescued.append(r.get("seed"))
        print(f"{m:<16}{len(rs):>4}{f'{w3.mean():.4f}±{w3.std():.4f}':>20}"
              f"{w3.min():>9.4f}{str(len(hard)) + ' ' + str(hard):>8}"
              f"{(str(len(rescued)) + ' ' + str(rescued)) if floor else '-':>9}")
    print("=" * 78)
    if by_model and floor:
        print("配对口径: 崩溃 := 同种子地板 w3 - 自适应 w3 > 0.1 (主文稿规则); "
              "硬崩溃 := w3 F1 < 0.6。")


# ==============================================================================
# 8. 主流程 —— 数据准备逐行镜像 experiments.main (仅保留时间外推协议)
# ==============================================================================

def main():
    ap_ = argparse.ArgumentParser(description="外部自适应时序图模型的同协议稳定性审计")
    ap_.add_argument("--data", default="synthetic",
                     choices=["synthetic", "elliptic", "amlsim"])
    ap_.add_argument("--elliptic_dir", default="auto")
    ap_.add_argument("--amlsim_dir", default="auto")
    ap_.add_argument("--window_hours", type=int, default=4)
    ap_.add_argument("--window_mode", default="quantile",
                     choices=["quantile", "time"])
    ap_.add_argument("--n_windows", type=int, default=60)
    ap_.add_argument("--amlsim_max_nodes", type=int, default=300000)
    ap_.add_argument("--edge_fan", type=int, default=1)
    ap_.add_argument("--models", default="all",
                     help="all, 或逗号分隔: " + ",".join(EXT_MODELS))
    ap_.add_argument("--epochs", type=int, default=25)
    ap_.add_argument("--pretrain_epochs", type=int, default=3)
    ap_.add_argument("--hidden", type=int, default=64)
    ap_.add_argument("--dropout", type=float, default=0.3)
    ap_.add_argument("--lr", type=float, default=5e-3)
    ap_.add_argument("--chunk", type=int, default=8)
    ap_.add_argument("--train_end", type=int, default=34)
    ap_.add_argument("--val_end", type=int, default=39)
    ap_.add_argument("--freeze_mod_from", type=int, default=-1,
                     help="切除地板: t>=此窗口起关闭时间适应; -1=关闭")
    ap_.add_argument("--node_block", type=int, default=65536,
                     help="dysat/graphmamba 逐节点算子的节点分块大小 (6GB 显存可调小)")
    ap_.add_argument("--feature_set", default="all", choices=["all", "local"])
    ap_.add_argument("--seed", type=int, default=42)
    ap_.add_argument("--seeds", default="",
                     help="多种子: 如 '42,43,...,51'。数据划分固定(seed=42)")
    ap_.add_argument("--tag", default="")
    ap_.add_argument("--floor_csv", default="",
                     help="可选: 地板运行的 per-seed CSV, 用于崩溃配对分析")
    args = ap_.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[环境] device={device}  torch={torch.__version__}")

    data_seed = 42 if args.seeds else args.seed
    if args.data == "synthetic":
        data = make_synthetic(seed=data_seed)
    elif args.data == "amlsim":
        data = load_amlsim(args.amlsim_dir, window_hours=args.window_hours,
                           max_nodes=args.amlsim_max_nodes, seed=data_seed,
                           edge_fan=bool(args.edge_fan),
                           window_mode=args.window_mode, n_windows=args.n_windows)
        if args.train_end == 34 and args.val_end == 39:
            args.train_end = max(2, int(data["T"] * 0.6))
            args.val_end = max(args.train_end + 1, int(data["T"] * 0.8))
            print(f"[自适应] AMLSim 时间划分: 训练 1-{args.train_end} | "
                  f"验证 {args.train_end+1}-{args.val_end} | "
                  f"测试 {args.val_end+1}-{data['T']}")
        if args.chunk == 8:
            args.chunk = 4
            print("[自适应] 节点规模较大, chunk 8 -> 4 (防显存溢出)")
    else:
        data = load_elliptic(ensure_elliptic(args.elliptic_dir))

    data["train_eligible"] = np.ones(len(data["labeled"]), dtype=bool)

    if args.feature_set == "local" and data["X"].shape[1] >= 166:
        data["X"] = data["X"][:, :94]
        print("[特征] LF 设置: 仅使用前 94 维局部特征")

    # 特征标准化 (仅用训练时间片统计量, 与 experiments.main 一致)
    tr = data["node_time"] <= args.train_end
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

    valid = EXT_MODELS + FLOOR_MODELS
    modes = EXT_MODELS if args.models == "all" else [
        m.strip() for m in args.models.split(",") if m.strip() in valid]
    if not modes:
        raise SystemExit(f"[错误] 无有效模型; 可选: {valid}")
    print(f"[审计] 模型: {modes}")

    # GAT 边列表仅 dysat 需要; 提前构建一次, 所有种子共享
    edges = (build_edge_lists(data, device)
         if any(m.startswith("dysat") for m in modes) else None)

    seed_list = [int(s) for s in args.seeds.split(",")] if args.seeds else [args.seed]
    multi = len(seed_list) > 1

    tag = (f"_fz{args.freeze_mod_from}" if args.freeze_mod_from >= 0 else "") + args.tag
    base = f"TGM-Lite_审计结果_supervised_{args.data}{tag}"
    raw_path = f"{base}_seeds_raw.csv" if multi else f"{base}.csv"

    # 断点续跑: 已存在于原始 CSV 的 (模型, 种子) 组合自动跳过, 每个种子跑完即追加写盘
    done = set()
    if os.path.exists(raw_path):
        for r in load_rows_csv(raw_path):
            done.add((r["model"], r["seed"]))
        if done:
            print(f"[续跑] 检测到 {raw_path} 已有 {len(done)} 个组合, 自动跳过")

    for si, sd in enumerate(seed_list):
        args.seed = sd
        if multi:
            print(f"\n{'#'*70}\n# 种子 {sd} ({si+1}/{len(seed_list)})\n{'#'*70}")
        for m in modes:
            if (m, sd) in done:
                print(f"\n>>> 跳过 {m} (seed={sd}): 已在 {raw_path} 中")
                continue
            print(f"\n>>> 训练模型: {m} (seed={sd})")
            row = run_one_ext(m, data, Ahats, X, split, args, device, edges)
            if row is not None:
                row["seed"] = sd
                append_rows_csv([row], raw_path)
                print(f"  [结果] {m}: AP={row['ap']:.4f} F1={row['f1']:.4f} "
                      f"w3={row['drift'][2]:.4f} params={row['params']:,}  [已落盘]")

    # 汇总: 重新读回完整原始 CSV (含续跑的历史行), 保证覆盖全部种子
    all_rows = load_full_rows_csv(raw_path)
    print(f"[汇总] 共 {len(all_rows)} 行 ({raw_path})")
    if multi:
        agg = aggregate_seeds(all_rows)
        print_agg_table(agg)
        save_agg_csv(agg, f"{base}_seeds.csv")
    collapse_summary(all_rows, floor_csv=args.floor_csv or None)


if __name__ == "__main__":
    main()
