@echo off & chcp 65001 >nul
setlocal EnableDelayedExpansion
rem ============================================================================
rem  统计检验升级.bat  —  TGM-Lite 论文统计强化 (审稿意见回应)
rem ----------------------------------------------------------------------------
rem  脚本: stats_upgrade.py (仅依赖 numpy/pandas/scipy)
rem  自动扫描结果目录全部 *_raw.csv, 逐组输出:
rem    描述统计(95%%CI) / Welch t / Holm+BH 校正 / 均值差 CI /
rem    Hedges g + Cliff delta 效应量 / Levene+Brown-Forsythe 方差检验 /
rem    非劣效 + TOST 等效检验 (margin 默认 0.02)
rem  输出: %RES%\stats\<组名>_stats_long.csv, <组名>_table.tex, all_stats_long.csv
rem
rem  结果组与论文内容顺序的对应 (TIFS 稿, 按文件名自动成组):
rem    Table I/II  主表+漂移   <- TGM-Lite_实验结果_supervised_amlsim_10s_seeds_raw.csv
rem    Table III   跨银行      <- ..._bank_10s_seeds_raw.csv
rem    Table IV    标签稀缺    <- lr0.05_10s + lr0.05_10s_evolvegcn (用 --pool 并为一族)
rem    Table VI    Elliptic    <- ..._supervised_elliptic_10s_seeds_raw.csv
rem    Table VII   消融        <- ..._r4/_r32/_m64/_m256/_c16/_nopre_10s_*_seeds_raw.csv
rem    Table VIII  地板        <- ..._fz40_floor10_seeds_raw.csv (仅 tgmlite, 输出描述统计)
rem    Table IX    外部审计    <- 审计结果_*_ext10/static10/fz40_ext10fz40_seeds_raw.csv
rem    附录 Table X PU         <- ..._pu_elliptic_10s_seeds_raw.csv
rem    RQ6 中段对照            <- ..._elliptic_midstream / fz32_midstreamfz32_seeds_raw.csv
rem    RQ10 对抗审计           <- 对抗审计_amlsim_adv10_seeds_raw.csv
rem  注: 目录中所有 *_raw.csv 都会被统计; 无 tgmlite 的组仅出描述统计, 属正常。
rem ============================================================================

set "PY=C:\Users\apple\anaconda3\python.exe"
cd /d "%~dp0"

rem ---- 结果目录: 默认 results, 若 10 种子主表不在其中则退回当前目录 ----
set "RES=results"
if not exist "%RES%\TGM-Lite_实验结果_supervised_amlsim_10s_seeds_raw.csv" set "RES=."
if not exist "%RES%\TGM-Lite_实验结果_supervised_amlsim_10s_seeds_raw.csv" (
    echo [错误] 找不到 TGM-Lite_实验结果_supervised_amlsim_10s_seeds_raw.csv
    echo        请把 10 种子 CSV 放入 results\ 或与本 bat 同目录, 再重跑。
    pause & exit /b 1
)
echo [使用结果目录] %RES%
echo.

rem ---- 5%% 标签稀缺: tgmlite 与 evolvegcn 在两个文件, 合并为同一比较族 ----
set "POOL="
if exist "%RES%\TGM-Lite_实验结果_supervised_amlsim_lr0.05_10s_seeds_raw.csv" (
    if exist "%RES%\TGM-Lite_实验结果_supervised_amlsim_lr0.05_10s_evolvegcn_seeds_raw.csv" (
        set "POOL=--pool TGM-Lite_实验结果_supervised_amlsim_lr0.05_10s_seeds_raw.csv+TGM-Lite_实验结果_supervised_amlsim_lr0.05_10s_evolvegcn_seeds_raw.csv:amlsim_label5_pooled"
    )
)
if defined POOL (echo [标签稀缺] 已合并两个 raw 文件为同一比较族) else (
    echo [提示] 未找到标签稀缺 10 种子 raw 文件, 跳过 pool ^(对应组将分别统计^)
)
echo.

echo ===== [1/3] 扫描 %RES%\*_raw.csv 并按组统计 (Table I - X + RQ6/RQ10 审计组) =====
echo ===== [2/3] 参考模型 tgmlite 的 Welch t + Holm/BH + 效应量 + 方差/非劣效检验 =====
echo ===== [3/3] 写出 stats\ 长表与 LaTeX 效应量表 =====
echo.

"%PY%" stats_upgrade.py --results_dir "%RES%" --ref tgmlite --margin 0.02 %POOL%

if errorlevel 1 (echo. & echo [失败] 请检查上方报错。 & pause & exit /b 1)

echo.
echo ===== 完成 =====
echo 结果: %RES%\stats\  (*_stats_long.csv 全数值 / *_table.tex 可进论文 / all_stats_long.csv 汇总)
echo 提示: n^<10 的组会打印功效警告; 论文全部表格均为 10 种子 (42-51)。
pause
