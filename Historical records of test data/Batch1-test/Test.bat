@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo ## Start 10-seed 全套 PyCharm Parameters V1
echo ##种子统一 `42,43,44,45,46,47,48,49,50,51`，全部加 `--tag _10s` 保护现有 5-seed 文件。**注意：`--model all` 已自动含 nossm/nograph 两个消融，无需单独跑。

echo # 1. AMLSim 主表(时间外推) —— 含 nossm/nograph 消融 
echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model all --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
echo # C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model all --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo # 2. 跨银行 
echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model all --split_mode bank --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model all --split_mode bank --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo # 3. 5% 标签(共两条) 
echo s_date-time:%date%-%time%
echo # (第一条)C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model rf,mlp,gcn,tgmlite --label_rate 0.05 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model rf,mlp,gcn,tgmlite --label_rate 0.05 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo s_date-time:%date%-%time%
echo # (第二条)C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model evolvegcn --label_rate 0.05 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_evolvegcn
 C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model evolvegcn --label_rate 0.05 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_evolvegcn
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo # 4. Elliptic 主表 + PU (共两条) 
echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe experiments.py --data elliptic --model all --epochs 40 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
echo # (第一条) C:\Users\apple\anaconda3\python.exe experiments.py --data elliptic --model all --epochs 40 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo s_date-time:%date%-%time%
echo # (第二条)C:\Users\apple\anaconda3\python.exe experiments.py --data elliptic --task pu --epochs 40 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
C:\Users\apple\anaconda3\python.exe experiments.py --data elliptic --task pu --epochs 40 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo # 5. 超参消融(CLI 原生支持, 不改代码): 预训练/秩r/状态维m/chunk，(共七条) 
echo s_date-time:%date%-%time%
echo (第一条)#C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --pretrain_epochs 0 --epochs 100 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --pretrain_epochs 0 --epochs 100 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo s_date-time:%date%-%time%
echo (第二条)#C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --rank 4 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_r4
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --rank 4 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_r4
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo s_date-time:%date%-%time%
echo (第三条)#C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --rank 32 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_r32
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --rank 32 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_r32
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo s_date-time:%date%-%time%
echo (第四条)#C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --state 64 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_m64
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --state 64 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_m64
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo s_date-time:%date%-%time%
echo (第五条)#C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --state 256 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_m256
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --state 256 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_m256
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo s_date-time:%date%-%time%
echo (第六条)#C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --chunk 4 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_c4
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --chunk 4 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_c4
echo e_date-time:%date%-%time%
timeout /t 40 /nobreak >nul

echo s_date-time:%date%-%time%
echo (第七条)#C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --chunk 16 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_c16
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --chunk 16 --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s_v2_c16
echo e_date-time:%date%-%time%

echo END 10-seed 全套 PyCharm Parameters

pause
