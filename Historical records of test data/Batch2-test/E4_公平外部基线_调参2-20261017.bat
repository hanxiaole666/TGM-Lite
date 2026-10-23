@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

set PY=C:\Users\apple\anaconda3\python.exe

REM ---------- 选参 (打印阶段二推荐命令) ----------
echo s_date-time:%date%-%time%
%PY% e4_select_lr.py

REM ---------- 阶段二: 按 e4_select_lr.py 输出的命令逐条执行 ----------
echo s_date-time:%date%-%time%
REM 示例 (lr 以实际选出为准):
REM %PY% audit_external.py --data amlsim --models evolvegcn_full --lr <选出> --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10
echo s_date-time:%date%-%time%
echo #C:\Users\apple\anaconda3\python.exe audit_external.py --data amlsim --models dysat --lr 5e-3 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10
C:\Users\apple\anaconda3\python.exe audit_external.py --data amlsim --models dysat --lr 5e-3 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10

REM %PY% audit_external.py --data amlsim --models dysat          --lr <选出> --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10
echo s_date-time:%date%-%time%
echo #C:\Users\apple\anaconda3\python.exe  audit_external.py --data amlsim --models evolvegcn_full --lr 5e-2 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10
C:\Users\apple\anaconda3\python.exe  audit_external.py --data amlsim --models evolvegcn_full --lr 5e-2 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10

REM %PY% audit_external.py --data amlsim --models graphmamba     --lr <选出> --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10
echo s_date-time:%date%-%time%
echo #C:\Users\apple\anaconda3\python.exe  audit_external.py --data amlsim --models graphmamba --lr 5e-2 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10
C:\Users\apple\anaconda3\python.exe  audit_external.py --data amlsim --models graphmamba --lr 5e-2 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _tuned10
echo e_date-time:%date%-%time%

echo 完成后把 *_tune_lr*_seeds_raw.csv 与 *_tuned10_seeds_raw.csv 
echo 用于更新 Table 9 (调参后行) 与正文 RQ9 段落。

pause
