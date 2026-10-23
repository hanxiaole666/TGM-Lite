@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --freeze_mod_from 40 --tag _floor10
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --freeze_mod_from 40 --tag _floor10

pause
