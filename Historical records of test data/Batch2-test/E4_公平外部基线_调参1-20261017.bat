@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite
REM ============================================================
REM  E4. 外部基线公平化调参 (2.5: EvolveGCN-O fairness)
REM  协议: 阶段一 3 seeds (42-44) 试点, lr in {5e-4, 5e-3, 5e-2},
REM        按 val_AP (验证窗 37-48) 选每模型最优 lr;
REM        阶段二 用选出的 lr 跑 canonical 10 seeds (42-51)。
REM  选择只看验证窗, 测试窗不参与 —— 与主文选模口径一致。
REM  预计机时: 阶段一约 2-3 h, 阶段二约 5-8 h (RTX 2060 6GB)。
REM  前置: 使用本目录已打补丁的 audit_external.py (CSV 含 val_ap 列)。
REM ============================================================
set PY=C:\Users\apple\anaconda3\python.exe

REM ---------- 阶段一: lr 网格试点 (9 组, 各 3 seeds) ----------
echo s_date-time:%date%-%time%
%PY% audit_external.py --data amlsim --models evolvegcn_full --lr 5e-4 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44 --tag _tune_lr5em4
echo s_date-time:%date%-%time%
%PY% audit_external.py --data amlsim --models evolvegcn_full --lr 5e-3 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44 --tag _tune_lr5em3
echo s_date-time:%date%-%time%
%PY% audit_external.py --data amlsim --models evolvegcn_full --lr 5e-2 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44 --tag _tune_lr5em2

echo s_date-time:%date%-%time%
%PY% audit_external.py --data amlsim --models dysat --lr 5e-4 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44 --tag _tune_lr5em4
echo s_date-time:%date%-%time%
%PY% audit_external.py --data amlsim --models dysat --lr 5e-3 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44 --tag _tune_lr5em3
echo s_date-time:%date%-%time%
%PY% audit_external.py --data amlsim --models dysat --lr 5e-2 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44 --tag _tune_lr5em2

echo s_date-time:%date%-%time%
%PY% audit_external.py --data amlsim --models graphmamba --lr 5e-4 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44 --tag _tune_lr5em4
echo s_date-time:%date%-%time%
%PY% audit_external.py --data amlsim --models graphmamba --lr 5e-3 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44 --tag _tune_lr5em3
echo s_date-time:%date%-%time%
%PY% audit_external.py --data amlsim --models graphmamba --lr 5e-2 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44 --tag _tune_lr5em2

REM ---------- 选参 (打印阶段二推荐命令) ----------
echo s_date-time:%date%-%time%
%PY% e4_select_lr.py

REM ---------- 阶段二: 按 e4_select_lr.py 输出的命令逐条执行 ----------
echo s_date-time:%date%-%time%
REM 示例 (lr 以实际选出为准):
REM %PY% audit_external.py --data amlsim --models evolvegcn_full --lr <选出> --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10
REM %PY% audit_external.py --data amlsim --models dysat          --lr <选出> --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10
REM %PY% audit_external.py --data amlsim --models graphmamba     --lr <选出> --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10

echo 完成后把 *_tune_lr*_seeds_raw.csv 与 *_tuned10_seeds_raw.csv 
echo 用于更新 Table 9 (调参后行) 与正文 RQ9 段落。

pause
