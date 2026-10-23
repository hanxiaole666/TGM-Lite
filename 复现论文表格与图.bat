@echo off & chcp 65001 >nul
setlocal EnableDelayedExpansion
rem ============================================================================
rem  复现论文表格与图.bat  —  TGM-Lite 论文全部表格 / 图的一键再生成
rem ----------------------------------------------------------------------------
rem  脚本: reproduce_paper.py (仅依赖 numpy/pandas/scipy/matplotlib)
rem  输入: 10 粒种子 (42-51) 的实验 CSV; 输出: paper_assets_10s\tables\*.tex
rem        与 paper_assets_10s\figures\*.{pdf,png}
rem
rem  与论文内容顺序的对应关系 (TIFS 稿):
rem    Table I    主表 (AMLSim 时间外推)   <- --main     amlsim_10s
rem    Table II   漂移块 F1 + 稳定性      <- 由主表 CSV 派生, 无需另跑
rem    Fig.2      漂移曲线               <- 由主表 CSV 派生
rem    Fig.3      逐种子箱线图           <- 由主表 CSV 派生
rem    Table III  跨银行泛化             <- --bank      amlsim_bank_10s
rem    Table IV   5%% 标签稀缺           <- --scarcity  lr0.05_10s (+ --scarcity_egcn)
rem    Table V    效率                   <- 由主表 raw CSV 派生
rem    Table VI   Elliptic 边界          <- --elliptic  elliptic_10s
rem    Table VII  消融 (r/m/chunk/预训练) <- 手工维护 tables\tab_ablation.tex
rem                                          (数据: 超参消融 *_r4/_r32/_m64/_m256/_c16/_nopre CSV)
rem    Table VIII + Fig.4  切除地板认证   <- --floor     fz40_floor10 (同时生成附录 tab_floor_seeds.tex)
rem    Table IX   外部机制审计 (RQ9)     <- 手工维护 tables\tab_external.tex
rem                                          (数据: 审计结果_*_ext10 / static10 / fz40_ext10fz40 CSV)
rem    附录 Table X   PU 学习            <- --pu        pu_elliptic_10s
rem    附录 Table XI  逐种子地板配对     <- 由 --floor 自动生成 (tab_floor_seeds.tex)
rem    Fig.1      架构图                 <- 论文 LaTeX 内 TikZ 矢量图, 无需数据
rem
rem  若需从原始数据重跑实验 (每格 10 种子, 对应上表顺序):
rem    %PY% experiments.py --data amlsim   --model all --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _10s
rem    %PY% experiments.py --data amlsim   --model all --epochs 100 --pretrain_epochs 5 --split_mode bank --seeds 42,...,51 --tag _bank_10s
rem    %PY% experiments.py --data amlsim   --model rf,mlp,gcn,tgmlite --label_rate 0.05 --epochs 100 --seeds 42,...,51 --tag _lr0.05_10s
rem    %PY% experiments.py --data amlsim   --model evolvegcn --label_rate 0.05 --epochs 100 --seeds 42,...,51 --tag _lr0.05_10s_evolvegcn
rem    %PY% experiments.py --data elliptic --model all --epochs 40  --pretrain_epochs 5 --seeds 42,...,51 --tag _elliptic_10s
rem    %PY% experiments.py --data elliptic --task pu    --epochs 40  --pretrain_epochs 5 --seeds 42,...,51 --tag _pu_elliptic_10s
rem    %PY% experiments.py --data amlsim   --model tgmlite --epochs 100 --pretrain_epochs 5 --seeds 42,...,51 --freeze_mod_from 40 --tag _fz40_floor10
rem ============================================================================

set "PY=C:\Users\apple\anaconda3\python.exe"
cd /d "%~dp0"

rem ---- 结果目录: 默认 results, 若 10 种子主表不在其中则退回当前目录 ----
set "RES=results"
if not exist "%RES%\TGM-Lite_实验结果_supervised_amlsim_10s_seeds.csv" set "RES=."
if not exist "%RES%\TGM-Lite_实验结果_supervised_amlsim_10s_seeds.csv" (
    echo [错误] 找不到 TGM-Lite_实验结果_supervised_amlsim_10s_seeds.csv
    echo        请把 10 种子 CSV 放入 results\ 或与本 bat 同目录, 再重跑。
    pause & exit /b 1
)
echo [使用结果目录] %RES%
echo.

echo ===== [1/4] Table I 主表 + Table II 漂移 + Table V 效率 + Fig.2 + Fig.3 =====
echo ===== [2/4] Table III 跨银行 + Table IV 标签稀缺 + Table VI Elliptic =====
echo ===== [3/4] Table VIII 地板认证 + Fig.4 (--floor) =====
echo ===== [4/4] 附录 Table X (PU 学习) =====
echo.

"%PY%" reproduce_paper.py ^
  --results_dir "%RES%" ^
  --out_dir paper_assets_10s ^
  --main          "TGM-Lite_实验结果_supervised_amlsim_10s_seeds.csv" ^
  --bank          "TGM-Lite_实验结果_supervised_amlsim_bank_10s_seeds.csv" ^
  --scarcity      "TGM-Lite_实验结果_supervised_amlsim_lr0.05_10s_seeds.csv" ^
  --scarcity_egcn "TGM-Lite_实验结果_supervised_amlsim_lr0.05_10s_evolvegcn_seeds.csv" ^
  --elliptic      "TGM-Lite_实验结果_supervised_elliptic_10s_seeds.csv" ^
  --pu            "TGM-Lite_实验结果_pu_elliptic_10s_seeds.csv" ^
  --floor         "TGM-Lite_实验结果_supervised_amlsim_fz40_floor10_seeds.csv"

if errorlevel 1 (echo. & echo [失败] 请检查上方报错 ^(常见: 某个 CSV 文件名不符^)。 & pause & exit /b 1)

echo.
echo ===== 完成 =====
echo 表格: paper_assets_10s\tables\   (tab_main^|tab_drift^|tab_bank^|tab_scarcity^|tab_eff^|tab_elliptic^|tab_floor^|tab_floor_seeds^|tab_pu.tex)
echo 图:   paper_assets_10s\figures\  (fig_drift^|fig_stability^|fig_floor 的 pdf+png)
echo 注:   Table VII 消融 / Table IX 外部审计为手工维护的 .tex,
echo       数据来源见文件头注释; Fig.1 架构图为论文内嵌 TikZ, 无需生成。
pause
