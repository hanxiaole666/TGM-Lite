@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo s_date-time:%date%-%time%
echo #C:\Users\apple\anaconda3\python.exe audit_adversarial.py --data amlsim --models tgmlite --attacks evasion,drift,evasion_fz --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --eps 0.05,0.1,0.2 --tag _adv10
C:\Users\apple\anaconda3\python.exe audit_adversarial.py --data amlsim --models tgmlite --attacks evasion,drift,evasion_fz --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --eps 0.05,0.1,0.2 --tag _adv10
echo e_date-time:%date%-%time%

echo 完成后把 TGM-Lite_对抗审计_amlsim_adv10_seeds_raw.csv 与TGM-Lite_对抗审计_amlsim_adv10_summary.csv 
echo 用于更新 RQ10 / Table X (tab_adv) 与正文段落。

pause