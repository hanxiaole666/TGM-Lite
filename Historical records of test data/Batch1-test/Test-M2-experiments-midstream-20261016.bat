@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo #真实数据崩溃复现（M2）
echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe  experiments.py --data elliptic --model gcn,evolvegcn,tgmlite --epochs 40 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --train_end 24 --val_end 31 --tag _midstream
C:\Users\apple\anaconda3\python.exe  experiments.py --data elliptic --model gcn,evolvegcn,tgmlite --epochs 40 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --train_end 24 --val_end 31 --tag _midstream
echo e_date-time:%date%-%time%

echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe  experiments.py --data elliptic --model tgmlite --epochs 40 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --train_end 24 --val_end 31 --freeze_mod_from 32 --tag _midstreamfz32
C:\Users\apple\anaconda3\python.exe  experiments.py --data elliptic --model tgmlite --epochs 40 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --train_end 24 --val_end 31 --freeze_mod_from 32 --tag _midstreamfz32
echo e_date-time:%date%-%time%

pause
