# wide64: a drafted batch step of up to 64 rows

Status: design, increment 1 (bench model) and increment 2 (wide `launch_batch` body, 4-rank verified) on branch `wide64`. Production stays on
cb61118 + `DSGB10_BUDGET=1 DSGB10_OEA=4,12,9` until a wide build passes every gate.

## Why

The live 16-row step (8 agent sessions at 80-300K, nsys) is ~63-68 ms: MoE 37.5 (expert-union bytes), dense
11.7, draft pass 11.5 (3 serial passes), collectives 7.5, attention + indexer 3.4. `kMaxRows = 16` caps a
step at one draft per session at S = 8 (budget 16 - S), even though agent drafts are accepted ~92 % of the
time (live log, OEA build: 1143 of 1247 drafts accepted, 14.8 tokens per 15.5-row step). More rows per step
are cheap in bytes: the union grows sub-linearly (16 sessions x k rows: 48 / 70 / 87 / 99 experts per layer at
k = 1 / 2 / 3 / 4), and the prefill kernels read the union once at 226-230 GB/s, where 16-row decode chunks
re-read shared experts and dense weights per chunk.

## Measured model (increment 1: bench_chain, one rank, synthetic weights)

`bench_chain --rows R --experts U --variants ...` now runs 17-64 rows: `base` = the decode kernels in chunks of
16, `wide` = dense projections as prefill GEMMs (rmsnorm + `gemm_fp8_p` / `gemm_woa_p` / `gemm_bf16_p`) and
the MoE as `router_topk` + `moe_prefill`, `hgemm` = the head as a GEMM (raw head), attention / indexer
stand-ins once per 16-row group, collective stand-ins scaled per row as before. U from the real route dump.

| rows | U/layer | base (16-chunks) ms | wide ms | wide+hgemm ms | ms per row |
|---|---|---|---|---|---|
| 16 | 48 | 58.1 | (= base) | - | 3.63 |
| 24 | 60 | 117.7 | 77.6 | 76.4 | 3.18 |
| 32 | 71 | 150.6 | 88.8 | 87.7 | 2.74 |
| 48 | 87 | - | 108.1 | 105.8 | 2.20 |
| 64 | 99 | - | 127.5 | 124.0 | 1.94 |

Split at 32 rows / 48 experts: base 111.9, wide 69.7, MoE-only wide 77.6, dense-only wide 108.5. So the MoE
gains most, and the dense GEMMs matter only once the MoE is wide. Marginal cost in the wide path is
~1.2 ms per row (24 -> 64 rows, union growth included) against ~3.6 ms per row in 16-row decode steps.
The live engine runs ~10 % above bench_chain at 16 rows (64 vs 58 ms).

### Measured on the engine (increment 2, 4 ranks, real weights, 2026-11-05)

`--mode batch --batch-tf` (8 sessions at 1-4.6K, teacher-forced) and `--mode bench --bench-groups` (ctx 4K,
natural routing; "rotated" = each session's routing shifted, the worst case for the union):

| rows | batch-tf ms (8 sess) | bench natural ms | experts/layer | bench rotated ms | ms per row (natural) |
|---|---|---|---|---|---|
| 16 (decode path) | 61.8 | 57.2-64.1 | 47-52 | 64.6-81.7 | 3.6-4.0 |
| 32 (wide) | 98.3 | 96.7-104.3 | 76-79 | 112.9-128.5 | 3.0-3.3 |
| 48 (wide) | 119.6 | 116.5-122.0 | 94-99 | 151.2-159.4 | 2.4-2.5 |
| 64 (wide) | 139.1 | 143.0 | 112 | 189.1 | 2.2 |

The engine runs 10-15 % above the bench_chain model (real collectives and attention). Correctness: each
session's samples are identical across 32-, 48- and 64-row steps (row results do not depend on the batch);
against the 16-row decode path 35 of 768 teacher-forced argmaxes differ (ulp-level drift flipping near-ties
on this synthetic code/doc text), golden-next-token matches 444 vs 446 of 768. qeval at c4/c16 is still the
gate once serve can reach the wide path.

### What that buys (estimate from the measured step times, not yet measured end to end)

| load | today | wide step | step ms (est.) | tokens/step (est.) | tok/s (est.) |
|---|---|---|---|---|---|
| 8 agent sessions | 16 rows, 14.8 tok, ~64 ms + host | 8 x 4 = 32 rows (measured 98 ms) | ~98 + draft ~12 | ~27-29 | ~245-265 (live 215) |
| code16 | 16 rows, 16 tok, ~54 ms (OEA) | 16 x 3 = 48 rows (measured 117-122 ms) | ~120 + draft ~23 | ~38-42 | ~270-300: no gain over live 296 without OEA on the wide MoE and a one-pass drafter |

Revised after the engine numbers: the wide step alone is a modest win for agent load and none for code16.
Code16 >= 400 needs both the wide drafter pass (one pass over 16 entries instead of 6 serial 3-entry
passes, ~23 ms today at 16 entries) and OEA on the wide MoE; it is not guaranteed by these numbers. Agent
load is the clearer win (more drafts per session at high acceptance). c1 never takes the wide path.

## Design

### Row buckets
Steps of 1..16 rows keep today's kernels, graphs and bits (c1, c4 and every 16-row step unchanged). Steps of
17..64 rows run the **wide body** of the bucket that holds them: 24, 32, 48 or 64 rows. The rows past R in a
bucket are **pad rows**: copies of the step's last real row (same token, position, slot), so they add no
experts to the union and compute the same values; every state writer masks rows >= `in->rows` (device
value), so a pad row never writes KV, compressed KV, index keys, carries, draft windows or samples.

### Capacity constants
- `config.h`: `kMaxRows` = 64 (step buffers, StepIn/SampleOut rows, Engram segments, route ring);
  new `kDecodeRows` = 16 (what the decode kernels take per call; `dsk::kMaxRows`, `attn::kMaxDecodeRows`
  stay 16); `kMaxSessRows` stays 8 (one session's rows, the attention contract); `ops::kMaxSess` decoupled
  from rows: 16 sessions.
- `dspark::kMaxHidden` -> 64 (the main_hidden tap of a wide step).
- Ring cap is 4 MiB per rank (`--ring-cap` default): 64 rows x 5120 fp32 = 1.25 MiB fits every collective.

### The wide body (forward_rows, R > 16)
| piece | 16-row path | wide path |
|---|---|---|
| attn_norm + wq_a + wkv | `gemv_fp8_norm` / rmsnorm + r16 GEMV | rmsnorm (+act_quant) + `gemm_fp8_p` (2 segs) |
| compressor rows | `compressor_project_rows` (<= 16) | the same in 16-row groups |
| all-gather + mixes | `allgather_attn_mix` RB 8/16 | RB 64 instance (or 16-row chunks, measured) |
| kv_write / compressor_emit | rows of the step | rows of the step, masked at `in->rows`, groups of 16 in order |
| q_norm + wq_b (+ idx wq_b) | normed GEMV | rmsnorm + `gemm_fp8_p` |
| indexer | `index_dense(_blocks)` <= 16 | in 16-row groups (pos, top-k, pool offset per group) |
| sparse_attn | rows of the step | rows of the step (no row cap; scratch sized 64) |
| wo_a + wo_b | `gemv_woa` + `gemv_fp8` | `gemm_woa_p` (+act_quant) + `gemm_fp8_p` |
| all-reduce + hc | `allreduce_hc` RB 8/16 | RB 64 instance |
| gate + routing | `gemv_bf16_norm` + routing inside `moe_routed` | rmsnorm + `gemm_bf16_p` + `router_topk` (OEA variant later) |
| MoE | `moe_routed` r16 | `moe_prefill` (grouped, ~1 ulp vs decode) |
| head | coded `gemv_bf16x_norm` | coded GEMV in 16-row groups (a coded-head GEMM later: -2..-3.5 ms at 48-64 rows) |
| sampler | `k_sample<8/16>` | `k_sample<64>` (grid 64, posted bytes 64 rows) |

Attention keeps the contract: <= 8 rows of one session per step, and groups of 16 run in row order, so a
session that straddles two groups is two consecutive calls (as two decode steps).

### Exactness
The 1..16-row bodies are untouched (bit-identical). A wide body's rows equal the same rows run in a 16-row
step only to ~1 bf16 ulp (prefill GEMM K-chains and moe_prefill vs the decode MoE). Each row is still
independent of the other rows of its step (no cross-row reductions), so outputs do not depend on batch
composition within the wide path. The operator accepted the ulp-level difference on condition that qeval holds
(>= 71 at c4 and c16).

### Spec (B3) changes
- `spec_batch_graph`: switch bodies 1..16 as today, plus 4 wide bodies (bucket index); `k_spec_in_batch`
  lays out pad rows and sets the bucket.
- `k_spec_gate_batch_all`: the row budget generalizes to 64. With `rbudget > 16` the gate keeps drafts by
  survival under a step-cost table (ms by bucket), i.e. it grows rows only while the marginal expected tokens
  beat the marginal ms (expert-budget-aware trimming without per-row expert knowledge: the bucket curve
  carries the union growth).
- Host: `kMaxSpecEnts` 8 -> 16 (entries), cap per entry 5 drafts, budget by load (S <= 4: 16 rows, the
  current path; S >= 5: up to 64).
- Draft pass: phase 1 keeps the 3-entry passes; phase 2 runs one pass over all entries through the prefill
  GEMMs (drafter rows 5 x entries <= 80).

### Increments (each with a test against the 16-row path)
1. bench_chain wide model (done, table above).
2. Capacity split (`kMaxRows` 64 / `kDecodeRows` 16 / `kMaxSess` 16), buffers, StepIn/SampleOut, sampler
   RB 64, fused collectives RB 64. Test: coll_tests at 17..64 rows vs the separate kernels; all existing
   tests unchanged; 1..16-row golden hashes unchanged.
3. Wide forward for `launch_batch` (no drafts), eager then graphs per bucket, row masking. Test: `--mode
   batch --batch-tf` with G x S rows (e.g. 6 x 8 = 48): each session's samples vs the 16-row run (allow
   ulp-level drift; logits max diff reported), plus `bench --ks` step times by bucket on 4 ranks.
4. Wide bodies in the B3 spec graph + generalized budget gate. Test: spec_batch_test, serve on 18901:
   abbench, agentload, qeval c4/c16, needle.
5. Wide drafter pass (one pass, GEMMs). 6. OEA routing for the wide MoE. 7. Coded-head GEMM.

### Gates to deploy
chat c1 >= 78, code c1 >= 134, code16 >= 400 (target) or a clear win at agent load, qeval >= 71 at c4 and
c16, needle 3/3 at 948K, health 200 and 17*23 = 391 after.
