@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo # smoke test (~1 min, CPU)
echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe audit_external.py --data synthetic --models all --epochs 3 --pretrain_epochs 1
C:\Users\apple\anaconda3\python.exe audit_external.py --data synthetic --models all --epochs 3 --pretrain_epochs 1
echo e_date-time:%date%-%time%

echo # full 10-seed audit on AMLSim
echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe audit_external.py --data amlsim --models evolvegcn_full,dysat,graphmamba --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _ext10
C:\Users\apple\anaconda3\python.exe audit_external.py --data amlsim --models evolvegcn_full,dysat,graphmamba --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _ext10
echo e_date-time:%date%-%time%

echo # matching excision floors (freeze temporal adaptation from window 40)
echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe audit_external.py --data amlsim --models evolvegcn_full,dysat,graphmamba --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --freeze_mod_from 40 --tag _ext10fz40
C:\Users\apple\anaconda3\python.exe audit_external.py --data amlsim --models evolvegcn_full,dysat,graphmamba --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --freeze_mod_from 40 --tag _ext10fz40
echo e_date-time:%date%-%time%

pause
