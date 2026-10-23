@echo off
chcp 65001 >nul
cd /d D:\TGM-Lite

echo s_date-time:%date%-%time%
echo #C:\Users\apple\anaconda3\python.exe audit_trajectory.py --data amlsim --seeds 42,43,47,50 --epochs 100 --pretrain_epochs 5 --tag _traj
C:\Users\apple\anaconda3\python.exe audit_trajectory.py --data amlsim --seeds 42,43,47,50 --epochs 100 --pretrain_epochs 5 --tag _traj
echo e_date-time:%date%-%time%

echo s_date-time:%date%-%time%
echo #C:\Users\apple\anaconda3\python.exe fig_trajectory.py --csv TGM-Lite_轨迹审计_amlsim_traj.csv --train_end 36 --val_end 48 --floor_from 40 --out fig_trajectory
C:\Users\apple\anaconda3\python.exe fig_trajectory.py --csv TGM-Lite_轨迹审计_amlsim_traj.csv --train_end 36 --val_end 48 --floor_from 40 --out fig_trajectory
echo e_date-time:%date%-%time%

echo GM-Lite_轨迹审计_amlsim_traj.csv 与 fig_trajectory.pdf
echo 用于插入 RQ2 (机理图) 与附录。

pause
