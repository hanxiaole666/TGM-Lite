@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo # 新增 dysat_static / graphmamba_static 模式（从零训练的静态对应物，未使用的时间模块已剔除、参数量诚实），崩溃规则改为（同种子静态对照 w3 − 自适应 w3 > 0.1）。
echo s_date-time:%date%-%time%
echo # C:\Users\apple\anaconda3\python.exe  audit_external.py --data amlsim --models dysat_static,graphmamba_static --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _static10
C:\Users\apple\anaconda3\python.exe  audit_external.py --data amlsim --models dysat_static,graphmamba_static --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _static10
echo e_date-time:%date%-%time%

pause
