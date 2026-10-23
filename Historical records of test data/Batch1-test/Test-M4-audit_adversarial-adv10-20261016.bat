@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo #对抗维度（M4）
#del TGM-Lite_对抗审计_amlsim_adv10_seeds_raw.csv TGM-Lite_对抗审计_amlsim_adv10_summary.csv

echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe  audit_adversarial.py --data amlsim --models tgmlite,gcn --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --eps 0.05,0.1,0.2 --tag _adv10
C:\Users\apple\anaconda3\python.exe audit_adversarial.py --data amlsim --models tgmlite,gcn --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --eps 0.05,0.1,0.2 --tag _adv10

echo e_date-time:%date%-%time%

pause
