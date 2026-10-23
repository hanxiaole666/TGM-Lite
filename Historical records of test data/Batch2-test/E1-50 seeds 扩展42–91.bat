@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo "扩展到至少 30–50 seeds." "10→50 seeds 后 Clopper–Pearson 上界从 0.26 收紧到 0/50 → 0.071，地板论证强度质变."
echo s_date-time:%date%-%time%
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model all --epochs 100 --pretrain_epochs 5 --seeds 53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91 --tag _40s_ext

echo s_date-time:%date%-%time% 地板协议同样扩展：
C:\Users\apple\anaconda3\python.exe experiments.py --data amlsim --model tgmlite --epochs 100 --pretrain_epochs 5 --freeze_mod_from 40 --seeds 53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91 --tag _fz40_floor40
echo e_date-time:%date%-%time%

pause
