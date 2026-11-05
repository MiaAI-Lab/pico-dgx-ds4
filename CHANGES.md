# Changes in this fork

Fork of blaesus/dsgx10 by Andy Shu (MIT). Changes by Patrick Moorhead.
Base: upstream `a92b8f4` (master, 2026-10-18). All changes are in `ds41f-4xgx10-switchless/`.

Measured on four GB10 boxes (TP=4) over a 200G switch, serving DeepSeek-V4.1-Flash-1017, `--max-ctx 1048576`.
Speeds are decode tok/s (total across concurrent requests). "agent" = 8 concurrent sessions with
80K-300K-token prompts, 512 tokens per turn. qeval = 75-question quality eval, bar >= 70.

| # | Change | Files | Default | Lossless | Before -> after |
|---|---|---|---|---|---|
| 1 | 17-bit block index in `select_bins` (prefix split `nlow`, guard `bkeys_stride` <= 131072) | `engine/attn/indexer.cu` | on | yes | max context 512K -> 1M; needle at 947,826 tokens 3/3; prefill 9.2k tok/s; qeval 71/75; c1 unchanged |
| 2 | `DefaultBodyLimit` 64 MB (axum default 2 MB) | `frontend/src/server.rs` | on | yes | HTTP 413 above ~400K tokens -> 948K-token prompts accepted |
| 3 | 16 slots, big/small slot units (`--big-slots`), slot-aware admission, big-slot wait instead of die | `engine/model/*`, `engine/server/{main,serve}.cpp` | `--slots`/`--big-slots` | yes | code c16 172 -> 234-247, c8 170 -> 195-200; c1 unchanged; qeval 71 |
| 4 | 9..16-row normed GEMVs as rmsnorm + plain r16 GEMV (`DSGB10_FUSED16=1` reverts) | `engine/model/model.cu` | on | yes (same bits) | c16 234 -> 253, c4 187 -> 204, step 77.3 -> 74.8 ms |
| 5 | Batch-aware expert re-routing `DSGB10_OEA=K0,KP,MINROWS` | `engine/kernels/{common.cuh,moe.cu}` | off | **no** | with 4,12,9: c16 253 -> 286, agent 180.7 -> 203.4; qeval 71/72 |
| 6 | Cross-session draft row budget `DSGB10_BUDGET=1` | `engine/model/{spec.cu,ops.h,model.h}`, `engine/server/cmd.cpp` | off | yes | c16 253 -> 259, agent 180.7 -> 189.5 |

## End to end

| Build | chat c1 | code c1 | c4 | c8 | c16 | agent | qeval c4/c16 | max ctx |
|---|---|---|---|---|---|---|---|---|
| upstream, 256K, 8 slots | 80 | 140 | - | - | 177 | - | 71 | 256K |
| + 1 + 2, 1M, 4 slots | 78-82 | 134-140 | - | 170 | 172 | - | 71 | 1M |
| + 3, 16 slots (4 big) | 79 | 135-140 | 187 | 197 | 234 | 180.1 | 71/71 | 1M |
| + 4 + 6 (`DSGB10_BUDGET=1`) | 80.4 | 137.8 | 211 | 212 | 259 | 189.5 | 71/71 | 1M |
| + 5 (`DSGB10_OEA=4,12,9`) | 77.5-78.5 | 137.8-139.3 | 239 | 236 | 296 | 215.1 | 71/73 | 1M |

Engine args for the last row: `--max-ctx 1048576 --slots 16 --big-slots 4 --setenv DSGB10_BUDGET=1 --setenv DSGB10_OEA=4,12,9`.

## Wide step (branch wide64, now main)

| # | Change | Default | Lossless | Before -> after |
|---|---|---|---|---|
| 7 | Wide batch step, 17-64 rows (`DSGB10_WIDE=W`): prefill GEMMs + `moe_prefill`, indexer/compressor/head in 16-row groups | off | ~1 ulp | c16 296 -> 350, agent 215 -> 240 |
| 8 | Cost-aware wide gate (`DSGB10_WIDE_COST`) | on with WIDE | yes | wide only when drafts pay |
| 9 | One-pass drafter (`DSGB10_DRAFT1=6`) + coded-head GEMM (`DSGB10_HEADG`) | off / on | ~1 ulp / same bits | 48-row step 120 -> 109 ms |
| 10 | Gate re-fit + `WIDE=64` | - | yes | c16 350 -> 438, c8 295 -> 330, agent 240 -> 260; qeval 71/71; needle 3/3 at 948K |
