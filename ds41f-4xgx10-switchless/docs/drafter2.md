# drafter2: lossless single-user speed (target chat c1 100, exact output)

STATUS (2026-11-09 15:40 UTC): nothing deployed; prod = c1c without RELAX (exact vis1 path), unchanged.
Backup made: head ~/dsgx10/serve.sh.bak-pre-drafter2 (identical copy of serve.sh).

## Infra built (branch drafter2 on rank3 ~/dsgx10-drafter2, from c1fast 507980f)
- `--draft-weights FILE` (server/main.cpp, model bind_spec): the drafter's mtp.* from a separate per-rank dsw
  file ({rank} expands); main weights untouched, embed/head stay the model's. Drafts change speed, never tokens.
  Binary: head ~/dsgx10-wide/night/bins/dsgb10_rank.d2a. Without the flag it is the c1c binary's path.
- Drafter conversion: `dsw_convert --ckpt DIR_WITH_MTP_SAFETENSORS --rank R --no-coverage --only mtp. --out ...`
  gives a 1.89 GiB rank file (115 tensors) in ~1 s.
- Identity set: head ~/drafter2/ident.py (51 greedy prompts: chat 20, code 12, math 10, zh 3, json 2, long
  8K-30K 4; plus 3 seeded temp-0.8 prompts; 256 tok, c1) + `--cmp A B`; acccat.py = per-category acceptance
  from rank0.log. rep.py = repeat a prompt N times (determinism).

## Lossless check method (lessons)
- Sequential c1 greedy is deterministic: 4/4 identical repeats on 4 prompts (prod).
- Two prod runs differed on 7/54: two owner requests at 15:00:48 batched with ours (batch rows change the
  arithmetic). Identity runs must be on :18901 or the gateway must show zero foreign requests in the window.

## Lever results
| lever | acceptance (accepted drafts/step) chat / code / math | lossless | verdict |
|---|---|---|---|
| shipped DSpark (base, d2a binary, 18901) | 1.105 / 2.923 / 2.797 | ref | |
| Avifenesh/DeepSeek-V4.1-Flash-DSpark-drafter-ft (self-distilled fine-tune, FP8 drop-in) | 1.069 / 2.964 / 2.793 | 54/54 identical to base | dead: chat -3 %, code +1.4 % |
| ursuciprian D1 (self-distill, half-strength delta) | - | - | EXL3-only overlay (trellis), not usable without requant; publishes prose +7-12 % |
GPU ms by rows (base, 18901, this set): 1: 19.21  2: 22.05  3: 25.08  4: 28.09  5: 30.58  6: 32.87.
Per-category speed base: chat 83.8, code 138.8, math 125.7, long 105.3 (median tok/s on the set).

## sparkDash 1.8.8 (exact prod build, 2026-11-09 15:24 UTC, no foreign traffic)
prose c1 100.2-100.6 (5 runs), code c1 165.2-166.0; catalogs prose 77.2 / structural 126.7 / code 153.0.

## Ceiling (measured inputs)
- Chat: 2.105 tok/step at ~22-23 ms (mix of 2-3 row steps). 100 tok/s needs ~2.4-2.5 tok/step (+16-19 %
  tok/step at today's step cost) or ~-15 % step cost, or a mix.
- Public self-distilled drafters reach -3 % (Avifenesh, here) to +7-12 % prose (D1, own engine). DeepSeek's
  shipped drafter is already trained at scale; a home self-distill is unlikely to give +16-19 % on prose.
- Free-sibling tree ceiling (upstream tree study): +22 % prose, but a sibling row costs 2.4-3.5 ms (b=1.1 ms +
  ~2 new experts x 0.71 ms); at that cost trees give +0.1-1 %.
- Engine fixed cost: 1-row step = 14.3 ms weights at 234.6 GB/s + 2.7 ms collectives + 0.8 attention; weights
  at 95 % of 273 GB/s = -1.4 ms, collectives halved = -1.3 ms: <= ~12 % at 1 row, ~8-10 % at 2-3 rows.
- Stack of best plausible: drafter +8 % x engine +10 % = chat ~84 -> ~97 (estimate, not measured).

## Next (what 100 would take)
1. Self-distill training of the 3 DSpark blocks on heretic's own greedy chat outputs, KL to target logits
   (needs: a main_hidden + target top-k logit dump in serve/spectf on 18901 at night, ~2M tokens = ~60 GB of
   layer-37..39 features; torch trainer for DSpark blocks with frozen FP4 experts; 1 night on 4 Sparks with
   prod down). Expected from public attempts: 0..+12 % prose acceptance.
2. Engine: persistent/ring GEMV for wo_a + wq_b at 2-3 rows (rows 2-3 are the chat case), fused
   allreduce_hc; each must hold the 54-prompt identity (bit-exact kernels only).
3. Cheap sibling rows (<0.5 ms) are what would make trees pay (+10 % prose per the tree study); needs rows
   that add no new experts (expert-aware sibling choice, EcoSpec-style), lossless.
