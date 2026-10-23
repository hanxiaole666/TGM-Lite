@echo off
chcp 65001 >nul
REM ============================================================
REM 统计检验电池 (论文附录 Table: 20 项检验 + Holm/BH 校正 + 效应量)
REM 输入: results_dir 下的 12 个 canonical 逐种子 CSV (见 stats_battery.py 头部)
REM 输出: paper_assets_10s\tables\tab_stats.tex
REM ============================================================
set "PY=C:\Users\apple\anaconda3\python.exe"
cd /d "%~dp0"

if exist "results\TGM-Lite_实验结果_supervised_amlsim_10s_seeds_raw.csv" (set "RES=results") else (set "RES=.")
echo [i] 结果目录: %RES%

"%PY%" stats_battery.py --results_dir "%RES%" --out_dir paper_assets_10s
if errorlevel 1 (echo [X] 失败, 请检查 12 个 CSV 是否齐全 & pause & exit /b 1)
echo [OK] tab_stats.tex 已生成 (paper_assets_10s\tables\). 与论文附录对照即可.
pause
