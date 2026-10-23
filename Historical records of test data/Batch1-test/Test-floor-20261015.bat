@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --epochs 100 --pretrain_epochs 5 --seeds 42,47 --freeze_mod_from 40 --tag _floor
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --epochs 100 --pretrain_epochs 5 --seeds 42,47 --freeze_mod_from 40 --tag _floor
echo e_date-time:%date%-%time%

pause
