@echo off
chcp 65001 >nul
REM ============================================================
REM  E2. 配置与切分鲁棒性 (
REM      "一个生成器、一个 HI-Small 配置、一个时间切分撑不起 class phenomenon")
REM
REM  E2a 另一种时间切分: train 1-30 / val 31-42 / test 43-60 (漂移窗相对训练段前移)
REM      地板 t_exc=34 = 验证起点+3, 与 canonical 的 40=37+3 同规则
REM  E2b 另一种 AMLSim 配置: HI-Medium (中等等级洗钱模式, 交易规模更大)
REM      数据全自动: 若下方路径的文件不存在, experiments.py 会
REM        1) 先在 D:\TGM-Lite\Data\ealtman2019 下寻找;
REM        2) 再从 HI-Small 当初下载残留的 kaggle 压缩包中单文件提取(零下载);
REM        3) 最后用 kagglehub 免登录单文件下载 HI-Medium_Trans.csv(约 1-2GB,
REM           单文件接口不可用时才降级为 7.6GB 整包)。
REM      手动备选: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
REM                下载后把 HI-Medium_Trans.csv 放到下方路径即可(已存在则直接使用)
REM  E2c 另一种生成器种子: 重新生成 HI-Small (改 AMLSim 种子), 命令同 E2b 换路径
REM
REM  断点续跑(2026-10-21 起 experiments.py 原生支持):
REM    多种子模式每完成一个 (seed, model) 组合立即写入 *_seeds_raw.csv,
REM    并在同名 .progress 文件登记。中途关闭窗口/断电后, 重新运行同一条命令
REM    即可自动跳过已完成组合, 只补跑缺失的; 聚合表始终按磁盘上的完整
REM    raw CSV 重算。删除对应 .progress 行 + raw 行可强制重跑某组合。
REM    注意: 不同实验务必用不同 --tag(本 bat 已遵守), 否则会写入同一文件。
REM
REM  预计机时: E2a 约 8 h (all 模型 10 seeds) + 地板 2.5 h;
REM            E2b 同量级 (HI-Medium 节点更多, 可能 1.5x)
REM ============================================================
set PY=C:\Users\apple\anaconda3\python.exe

REM ---------- E2a: 替代切分 (HI-Small, 数据不动只动切分) ----------
echo s_date-time:%date%-%time%
echo %PY% experiments.py --data amlsim --model all --epochs 100 --pretrain_epochs 5 --train_end 30 --val_end 42 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _split3042
%PY% experiments.py --data amlsim --model all --epochs 100 --pretrain_epochs 5 --train_end 30 --val_end 42 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _split3042
echo e_date-time:%date%-%time%

echo s_date-time:%date%-%time%
echo %PY% experiments.py --data amlsim --model tgmlite --epochs 100 --pretrain_epochs 5 --train_end 30 --val_end 42 --freeze_mod_from 34 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _split3042fz34
%PY% experiments.py --data amlsim --model tgmlite --epochs 100 --pretrain_epochs 5 --train_end 30 --val_end 42 --freeze_mod_from 34 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _split3042fz34
echo e_date-time:%date%-%time%

REM ---------- E2b: HI-Medium 配置 (数据缺失时自动下载, 已存在则直接用) ----------
echo s_date-time:%date%-%time%
echo %PY% experiments.py --data amlsim --amlsim_dir "D:\TGM-Lite\Data\ealtman2019\HI-Medium_Trans.csv" --model all --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _himed10
%PY% experiments.py --data amlsim --amlsim_dir "D:\TGM-Lite\Data\ealtman2019\HI-Medium_Trans.csv" --model all --epochs 100 --pretrain_epochs 5 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _himed10
echo e_date-time:%date%-%time%

echo s_date-time:%date%-%time%
echo %PY% experiments.py --data amlsim --amlsim_dir "D:\TGM-Lite\Data\ealtman2019\HI-Medium_Trans.csv" --model tgmlite --epochs 100 --pretrain_epochs 5 --freeze_mod_from 40 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _himed10fz40
%PY% experiments.py --data amlsim --amlsim_dir "D:\TGM-Lite\Data\ealtman2019\HI-Medium_Trans.csv" --model tgmlite --epochs 100 --pretrain_epochs 5 --freeze_mod_from 40 --seeds 42,43,44,45,46,47,48,49,50,51 --tag _himed10fz40
echo e_date-time:%date%-%time%

echo.
echo 完成后把 *_split3042*_seeds_raw.csv 与 *_himed10*_seeds_raw.csv 
echo 用于生成附录 "配置与切分鲁棒性" 小节 (模板见 E2E3_手稿填入模板.md)。
pause
