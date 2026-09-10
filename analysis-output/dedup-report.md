# Dedup Report: outputs/tables/results.csv

Date: 2026-08-20. Pipeline: `results-analysis` (inventory/validate phase). Inputs: `outputs/tables/results.csv` (465 rows), Hydra job dirs (`outputs/gat_cora/*/`, `outputs/multi30k/*/`), `outputs/traces/`.

## Result

- 465 rows → **316 rows** (dropped 149)
- Clean table: `outputs/tables/results_clean.csv`
- Raw backup: `outputs/tables/results_raw_20260820.csv` (original untouched)
- Remaining duplicate keys: **0**; smoke/ft20/superseded rows: **0**

## Drop log

| Rule | Rows dropped |
|---|---|
| cora smoke (epochs=5, notebook cell-4 re-runs) | 10 |
| multi30k smoke (epochs=2) | 2 |
| cora prune pilot (finetune_epochs=20) | 60 |
| multi30k pre-val-fix early-stop runs (11/13/18/30 ep) | 8 |
| distill ft=200 budget (superseded by later ft=100 runs) | 20 |
| hidden_per_head=64 (superseded by M1 decision) | 10 |
| cora adaptive dense: local CPU copies (kept GPU 10-seed block) | 19 |
| identical re-runs (same key + same metrics, different wall_time) | 20 |

## Key decisions (evidence)

1. **cora 自适应λ稠密：保留 GPU 套（CSV 行 423–432，wall 2.3–3.6 s/200 ep）**。依据：本地机器无 CUDA（torch 1.12.1 CPU）；该套是完整 10-seed、最终代码的最新运行。两套均值差 0.0004（GPU 0.8079±0.0092 vs 本地 0.8083±0.0159）。
2. **distill 保留 finetune_epochs=100**：job 时序（173013/173455 ft=200 在 15:34/15:34，175046/175541 ft=100 在 17:50/17:55）显示 ft=100 是后来统一的口径，与 experiment_plan §4「微调 epoch 固定（GAT 100）」一致。两组最终指标完全相同（best checkpoint）。
3. **multi30k 保留 epochs=50 协议行**：11/13/18/30 ep 行为 val 修复前早停产物（对应 job 114336/121921，patience 默认值），50 ep 行为 patience=50 的最终协议。
4. **剪枝 pilot ft=20 全部删除**：job 151711–154716 为试点；最终协议 ft=100（152413–174849）。

## Seed coverage（全部符合协议，无缺口）

- GAT 主配置（adaptive × {diversity,random,gradient,magnitude} × {0.5,0.75} × {ft,ft+KD}）：0.75 档 10 seeds；0.5 档 10 seeds（diversity：job 145153 seeds 0–4 + 174849 seeds 5–9；其余 5 seeds 为 runs=5 协议）
- distill B5（0.5/0.75）：10 seeds each
- cora 稠密：B0 none 10 seeds；B1 fixed-λ 5 seeds；B1-adaptive 10 seeds（GPU 套）
- 非自适应剪枝（regularizer=none）：5 seeds（runs=5 协议）
- λ 扫描（0.01–50、output、both）：5 seeds
- citeseer/pubmed：none/λ20/adaptive 各 10 seeds
- multi30k：none 5、adaptive **6**（seeds 0–5，比计划多 1）、fixed-λ20 5

## Caveats（写论文前需处理）

1. **GPU 套来源 = Colab T4（用户确认 2026-08-20）**：CSV 行 423–432（cora adaptive dense, wall 2.3–3.6 s/200 ep）跑在 Colab T4。`outputs/gat_cora/` 中无对应 job 目录（该批次可能运行于未挂载 Drive 的会话或目录未同步）→ 复现声明可写「Colab T4」，但逐次运行的 overrides 存档缺失，建议在论文方法中只声明机器+协议。
2. **traces 已齐全（用户确认 2026-08-20）**：`outputs/traces/` 即为全部可用 trace（cora 10 个、citeseer 10、pubmed 10、multi30k 6）。cora 的 10 个 `*_t0.7_e0.05.csv` 修改时间早于 T4 批次，对应同协议较早运行；λ 动态图直接使用现有 traces，图注注明「per-epoch λ/多样性轨迹（同协议运行）」，不声称与主表逐行同一物理运行。
3. **multi30k adaptive 为 6 seeds**（计划 5）：统计直接按 6 报告即可，无删减必要。
4. **pubmed none**：job 170341（seeds 4–9 补跑）的行未进入本 CSV（表内为 160434 的 10 seeds 完整套）；multi30k fixed-λ20 seed 4 的两次补跑（job 050343/011159）行也未进入。不影响覆盖完整性，但说明 Drive 侧可能还有未合并行。
5. **无 timestamp 列**：results.csv 每行没有运行时间戳，来源判定依赖 job 目录 + wall_time 交叉比对。建议后续 pipeline 在 row 中加 `job_timestamp` 字段（`RESULT_HEADERS` + 三个 pipeline 的 `row`）。

## Next

- 分析入口：`outputs/tables/results_clean.csv`（make_figures.py / final_stats.py 需改指向或替换）
- 进入 `results-analysis` 严格统计：H1–H3 检验（配对 Wilcoxon + 效应量 + 多重比较）
