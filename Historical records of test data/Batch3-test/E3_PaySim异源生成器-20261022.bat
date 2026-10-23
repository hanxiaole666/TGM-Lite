@echo off
chcp 65001 >nul
REM ============================================================
REM  E3. 异源生成器: PaySim 派生时序图 (TIFS 决定性实验)
REM  PaySim = 与 AMLSim 不同生成器族 (移动货币模拟器, 校准自真实日志),
REM  交易级 isFraud 标签, 744 小时步, ~636 万笔 (欺诈 0.13%)。
REM
REM  第 0 步(全自动): 下方 CSV 路径的文件缺失时, paysim_adapter.py 自动经
REM           kagglehub 免登录下载 (Kaggle ealaxi/paysim1, 单文件 ~180MB 压缩,
REM           失败再整包); 已存在则直接使用。手动备选: 该数据集页下载后放到
REM           D:\TGM-Lite\data\ 下即可。
REM
REM  断点续跑: 两条 experiments.py 命令均为多种子模式, 逐行落盘 raw CSV +
REM           .progress 登记; 中途关闭后重跑同一命令自动跳过已完成组合。
REM
REM  图构建规则 (paysim_adapter.py, 确定性种子 42):
REM    12h/窗 -> T=62; 节点=交易 (保留全部欺诈, 采样至 30 万);
REM    边 = 同窗账户星形边 + 跨窗资金链边; 特征 15 维,
REM    **不含余额列** (PaySim 欺诈交易被作废, 余额字段置零会泄漏标签)。
REM
REM  切分: 自动 60%/20%/20% -> train 1-37 / val 38-49 / test 50-62;
REM  地板 t_exc=41 = 验证起点+3, 与 canonical (40=37+3) 同规则。
REM
REM  预计机时: 适配器约 10-20 min; 主审计 (gcn+evolvegcn+tgmlite) 8-10 h;
REM            地板 2.5 h (RTX 2060 6GB)。
REM ============================================================
set PY=C:\Users\apple\anaconda3\python.exe

echo s_date-time:%date%-%time%
echo %PY% paysim_adapter.py --csv "D:\TGM-Lite\data\PS_20174392719_1491204439457_log.csv" --out paysim_graph.npz --hours_per_window 12 --max_nodes 300000
%PY% paysim_adapter.py --csv "D:\TGM-Lite\data\PS_20174392719_1491204439457_log.csv" --out paysim_graph.npz --hours_per_window 12 --max_nodes 300000
echo e_date-time:%date%-%time%

echo s_date-time:%date%-%time%
echo %PY% experiments.py --data custom --custom_npz paysim_graph.npz --model gcn,evolvegcn,tgmlite --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _paysim10
%PY% experiments.py --data custom --custom_npz paysim_graph.npz --model gcn,evolvegcn,tgmlite --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _paysim10
echo e_date-time:%date%-%time%

echo s_date-time:%date%-%time%
echo %PY% experiments.py --data custom --custom_npz paysim_graph.npz --model tgmlite --epochs 100 --pretrain_epochs 5 --freeze_mod_from 41 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _paysim10fz41
%PY% experiments.py --data custom --custom_npz paysim_graph.npz --model tgmlite --epochs 100 --pretrain_epochs 5 --freeze_mod_from 41 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _paysim10fz41
echo e_date-time:%date%-%time%

echo 完成后把 *_paysim10*_seeds_raw.csv 。
echo 关键判读: 自适应 tgmlite 是否再次出现 "验证健康 + 尾窗崩溃"
echo 签名, 以及 fz41 地板是否 0 崩溃 —— 两种结果的手稿文本都已备好
echo (见 E2E3_手稿填入模板.md, 分支 A 复现 / 分支 B 不复现, 同样有价值)。

pause
