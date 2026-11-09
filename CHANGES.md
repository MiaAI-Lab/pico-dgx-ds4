# Changes in pico-dgx-ds4

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

## Vision

| # | Change | Files | Default | Lossless | Before -> after |
|---|---|---|---|---|---|
| 11 | Image input: side encoder (the checkpoint's `inference/vision.py` + `image_processor.py`, SDPA on contiguous 4D views: MATH -> flash, 3.1 s -> 0.39 s per 444-token image); dsfront `--vision-addr` accepts `image_url`/`image` parts, expands each placeholder into its span; SUBMIT flag 8 carries the spans' rows; prefill injects them, span rows route with `ffn.gate.bias_vl`, Engram dead on spans; decode untouched | `tools/vision/`, `frontend/src/{vision,server,protocol,main}.rs`, `engine/common/frontend_protocol.h`, `engine/model/{prefill,model,ops}.cu`, `engine/server/{serve,cmd}.cpp`, `engine/engram/hash.cpp` | off (needs `--vision-addr`) | text: same path | images refused (400) -> 15/15 image checks, ~0.26 s TTFT per image; text: chat/code c1 79.7/138.4 -> 81.9/144.4, distinct c16 367 -> 380, agent 260 -> 266, qeval 71/71 -> 71/72, needle 948K 3/3 (noise) |

## Single user (c1fast)

The served build is the exact path: chat c1 ~80-82, code c1 ~138-144 (9-run medians). Change 12 below was a lossy
experiment; the code stays in tree, off by default, and is not used.

| # | Change | Files | Default | Lossless | Before -> after |
|---|---|---|---|---|---|
| 12 | Relaxed accept for one-session greedy decode (`DSGB10_RELAX_P=P`: a draft the target gives >= P probability is kept as the row's token; `DSGB10_RELAX=G` logit-gap form) + `DSGB10_RELAX_THRESH=T` fixed verify gate for one-session greedy decodes. Batch steps and sampled requests keep exact accept and the default gate | `engine/model/sample.cu`, `engine/server/cmd.cpp` | off | **no** | **REMOVED from the build 2026-11-09 (outputs must be bit-identical; this was lossy).** Measured with P=0.003, T=0.05: chat c1 81.6 -> 102.8, code c1 140.1 -> 162.9 (our bench, 9-run medians); sparkDash prose 101 -> 145, code 166 -> 172; qeval c1/c4/c16 72/71/71; distinct c16 381, copies 434, distinct c8 311, agent 260 (vis1 same day 262), needle 948K 3/3, images 15/15 |

Method check (sparkDash 1.8.8 at 754f40a, upstream's own sdrun.mjs harness): upstream's README 95 / 151 vs our 80 / 140
was the method (one easier prompt, 256 tokens), not the engine: the exact build scores 101 / 166 on sparkDash. Full
sweep and profile: [docs/c1fast.md](ds41f-4xgx10-switchless/docs/c1fast.md).
