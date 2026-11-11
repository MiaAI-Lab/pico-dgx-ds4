# stack4: three small single-user levers, stacked (2026-11-10)

Goal: more single-user speed with the same output. Three published ideas were measured first, then built where they paid:
recycling rejected verify states into the drafter (Carryover Drafting), a certified LM head, and a vocabulary shortlist for
the drafter's head. What shipped is a stack of three small levers. Chat on the 320 held-out prompts: 86.2 -> 89.5 tok/s
(+3.8 %), code +2.1 %, math +2.5 %. Every number below is measured unless it says simulated or modelled.

## What is live

| # | lever | flag | step change | gain on the 320 held prompts (chat / code / math) |
|---|---|---|---|---|
| 1 | MXFP4 drafter head for one-session drafts + FP8 Markov head (upstream's formats); the FP8 head copy is kept for batch draft passes, so multi-user is unchanged | `--draft-head4 --draft-markov8 --setenv DSGB10_HEAD4_C1=1` | draft pass -0.30..-0.38 ms | +1.14 / +0.85 / +0.92 % |
| 2 | confidence MLP gate (one-session drafts): 15 scalars of the draft pass -> 64 hidden -> 5 logits | `--setenv DSGB10_CF_MLP=FILE --setenv DSGB10_CF_ALPHA=1.0` | pass +0.1 ms, accepted drafts per step +5 % on chat | +1.1 / +0.9 / +0.9 % more |
| 3 | certified LM head (one-session greedy steps): FP8 coarse read + rigorous bound + exact kernel on survivors | `--setenv DSGB10_CERT=1` | head -0.18..-0.28 ms | +1.45 / +0.3 / +0.7 % more |

Paired runs (same prompts, sequential, greedy, thinking off; tok/s = tokens / wall of the engine's own session lines):

| build | chat | code | math | all | texts equal to the exact build |
|---|---|---|---|---|---|
| exact (before) | 86.19 | 128.20 | 133.70 | 101.08 | 320/320 |
| + lever 1 | 87.17 | 129.29 | 134.93 | 102.13 | 320/320 |
| + lever 2 | 88.17 | 130.50 | 136.11 | 103.22 | 320/320 |
| + lever 3 | 89.45 | 130.88 | 137.05 | 104.15 | 320/320 |

GPU ms per step by rows (1..6), same steps:

| build | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| exact | 19.41 | 22.37 | 25.49 | 28.57 | 31.29 | 33.81 |
| + lever 1 | 19.11 | 22.08 | 25.19 | 28.25 | 31.00 | 33.51 |
| + lever 2 (other steps: the gate changed) | 19.20 | 22.18 | 25.24 | 28.27 | 30.95 | 33.61 |
| + lever 3 (the same steps as the row above) | 18.99 | 21.90 | 24.97 | 28.04 | 30.73 | 33.43 |

## Lever 1: the drafter's head in 4 bits

The draft pass ends with a GEMV over the vocabulary head. Upstream already ships two cheaper formats behind flags: the head as
MXFP4 (`--draft-head4`) and the Markov head as FP8 (`--draft-markov8`). Drafts only propose; the verify step still decides
every token with the full model. With `--draft-head4` alone a batch draft pass reads the MXFP4 head once per session, which is
slower than one FP8 GEMM for 16 sessions. `DSGB10_HEAD4_C1=1` keeps the FP8 copy (83 MB more per rank) and uses the MXFP4 head
only for one-session drafts. Accepted drafts per step: 1.705 -> 1.703.

## Lever 2: a learned verify gate

DSpark's confidence head decides how many drafts a step verifies. Its read-out is one linear layer on the drafter's hidden
state. The oracle gate ("verify exactly the drafts that will be accepted") is worth +18 % on chat; the shipped head gets little
of it. A small MLP on scalars the pass already has does better: for each of the 5 depths the shipped logit, the margin between
the two best draft logits, and the log-probability of the best one. Each rank reduces its vocabulary slice to (max, second,
sum of exponentials) per depth, one all-gather joins them, and one kernel evaluates the MLP in a fixed order so every rank gets
the same bits. Held-out log-loss .335 -> .298; on rows logged from the engine itself .429 -> .383.

Plain words about output: a gate never chooses a token. Every emitted token is still the full model's own top choice in the
step that produced it. But this engine's logits depend slightly on how positions are grouped into steps (the same token at the
same position scored log p -0.24 as a step's first row and -0.10 as its fifth). A gate that verifies different drafts changes
the grouping, and on a close call the top choice can flip. Measured: with alpha 1.0 all 374 test outputs (54-prompt identity
set + 320 held prompts) equal the exact build's; with alpha 0.9 one of the 320 differed by one word. The same effect exists
without this lever whenever another request shares a step.

## Lever 3: a certified LM head

Greedy decoding needs only the argmax of the head, not all 129,280 logits. The step now:
1. reads the FP8 copy of the head (the drafter's, already in memory) for every vocabulary row: a coarse logit C_v;
2. bounds the error of each row rigorously: |exact - C_v| <= ||x|| B_v, where B_v = the norm of the row's quantisation
   residual (Cauchy-Schwarz) plus the worst-case fp32 rounding of both kernels, computed once at start from the real weights;
3. takes the best lower bound C_v - ||x|| B_v over the four ranks (one small collective);
4. keeps every row whose upper bound C_v + ||x|| B_v reaches it, and runs the UNCHANGED exact kernel on the 16-row groups that
   hold a kept row. The other logits are -inf. The usual sampler then finds the token.

The winner always survives (its upper bound is at least its own value, which is at least any lower bound), ties included, and
its logit is computed by the same kernel as before, so the token is the one the full head gives. If the bound cannot prune
(flat distributions), more groups are read, up to all of them. Sampled requests and requests with logprobs keep the full head.
Batch (multi-user) steps keep the full head.

Why only the FP8 copy works: measured on the real head with 4,944 real final hiddens, one global threshold:

| coarse code | row residual | survivors per row (median / p99) | head bytes read vs today |
|---|---|---|---|
| FP8 copy (8 bit) | 2.66 % | 1 / 84 | 65.7 % |
| 7 bit | 2.69 % | 1 / 90 | 58.7 % |
| 6 bit | 5.40 % | 7 / 9,518 | 53-62 % |
| 5 bit | 9.21 % | 4,962 / 126,376 | > 95 % |
| MXFP4 | 12.5 % | 65 % of all rows | > 100 % |

Centroid + radius clusters (CSV-Decode) and frequency-ordered early stopping are vacuous on this head: ||w|| ||x|| is about 185
logits against a top logit of 39. Per-rank thresholds do not work either (ranks that do not hold the winner cannot prune): the
threshold has to be shared.

Proof (engine in check mode: the full head runs beside the certified one in the same step, on every rank):
| run (check mode) | rows checked on each of the 4 ranks | mismatches | notes |
|---|---|---|---|
| every draft verified, train prompts | 1,069,390 | 0 | 181,308 steps |
| serving gate, 320 held prompts + identity set + train prompts | 168,936 | 0 | identity set 54/54 |
| fallback forced every 3rd step | 106,090 | 0 | 10,336 steps ran all 2,020 groups; identity set 54/54 |
| bound scaled x4 (almost nothing can be pruned) | 43,270 | 0 | 2,018 of 2,020 groups read on average |

1,344,416 real step rows at the serving bound, zero mismatches on every rank; the largest |coarse - exact| was 8.3 % of the
bound. A row counts as a mismatch unless the certified logits give the same (maximum, lowest id) as the full head on that
rank, or the full head's maximum on that rank is below the shared threshold (then that rank cannot hold the winner under
either head). Plus: identity set 54/54 against the exact build, 320/320 held texts, qeval 71/71.

## Measured and not built

| idea | result |
|---|---|
| Carryover, training-free: use the target's output at the first rejected row as the next first draft | right 17.9 % of the time on chat against the drafter's 64.9 % at the same anchors; an oracle that always picks the right one of the two is worth +1.0 % (simulated); a trained re-ranker +0.17 % (simulated) |
| the same as a gate feature ("rejected row agrees with the fresh draft") | first-draft log-loss .3853 -> .3808: about +0.1 % |
| vocabulary shortlist for the drafter's head (FR-Spec) on top of lever 1 | static top-64k usable on 59 % of chat steps: +0.3 % (simulated); "top-N + context tokens" loses more accepted drafts than it saves |
| certified head with a 6-7 bit code | 7 % fewer head bytes than the FP8 copy: about 0.07 ms (modelled); needs a new kernel |
| 7-bit FP8 packing of the dense weights | upstream costed it: +3.5 % at one row only if decoding is free, nothing at the measured decode cost; 1-2 weeks; not attempted |

## Credits

- Engine, kernels, drafter formats (MXFP4 head, FP8 Markov head): Andy Shu, blaesus/dsgx10.
- Model, DSpark drafter and its confidence scheduling: DeepSeek (DSpark, arXiv 2607.05147).
- Carryover Drafting, arXiv 2609.14717 (Koo et al.): the rejected-state idea and the matched-anchor analysis; ReTrace
  (arXiv 2608.29748), DFlow (arXiv 2609.06498), ReAnchor (arXiv 2610.26531) for the same family.
- Certified head: certhead (github.com/nickharris808/certhead, knlp.io/lm_head_decode.html) for coarse read + bound + exact
  re-read with a dense fallback; CSV-Decode (arXiv 2511.21702) for certified sub-vocabulary decoding.
- FR-Spec (arXiv 2502.14856) for the drafter shortlist that was measured.
- TVP (arXiv 2611.00588) was read for drafter retraining; no drafter weights were retrained here.

## Tools

`DSGB10_STEP_LOG=FILE` writes every verify step (rows, each row's sample incl. rejected rows, drafts, survival, MLP inputs);
`DSGB10_FORCE_NTOP=20` adds each row's top-20; `DSGB10_CERT_CHECK=1` runs the full head beside the certified one and counts
differences (`cert:` lines in each rank's log); `DSGB10_CERT_FORCE=N` forces the full mask every Nth step;
`DSGB10_CERT_BSCALE=f` loosens the bound; `DSGB10_MH_DUMP=DIR` writes every step row's layer 37-39 inputs (data collection).


## Carryover pilot: trained hidden-state version (measured offline, not in the engine)

After the training-free check above, we tried the real thing from Carryover Drafting (arXiv 2609.14717) on this engine: the
projected layer 37-39 states of the REJECTED verify rows enter the drafter's attention as one-round read-only context, marked
by one learned embedding. We did not verify extra rows to create rejected states (a row costs about 3 ms here); only the ones
the confidence gate already produces are used. The shipped drafter weights stay frozen. Code: `tools/stack4/carryover/`.

Data: served steps on 1,942 train requests (90,636 anchors that follow a rejection); scored on 314 held requests, 12,220 such
anchors. "Chain right through depth k" = the first k drafts all equal the served tokens.

| drafter | depth 1 | depth 2 | depth 3 | accepted length (5 drafted) |
|---|---|---|---|---|
| shipped | .6975 | .4580 | .2985 | 1.7801 |
| + carried states (marker, rank-64 projection, attention bias; 1,200 steps) | .7005 | .4677 | .3105 | 1.8187 (+2.17 %) |
| + gated rank-64 adapters (2,400 steps, best checkpoint) | .7031 | .4727 | .3164 | 1.8431 (+3.54 %) |

What it is worth in speed, so far: with the confidence gate's own verify counts at those anchors the first run realised
accepted drafts +0.26 % (chat), +0.39 % (code), +0.43 % (math), about zero tok/s after a +0.05 ms pass cost. The gain sits
mostly on the 2nd and 3rd drafts, which the gate rarely verifies. The adapter run has not been replayed with the gate, and the
confidence MLP has not been re-fit to the carry drafter. Not built into the engine.

Two things worth knowing: this is the only training that improved this drafter in any of our attempts (fine-tuning the shipped
weights on up to 7.7 M tokens never did), and the paper's larger gains come from verifying a fixed 15 drafts a round, which
leaves several times more rejected rows than DSpark's confidence-scheduled verify does.

## Gates for the stack (binary s4f, all three levers; exact build = same binary without the flags, same window where shown)

| gate | stack | exact |
|---|---|---|
| identity set (54 prompts, :18901) | 54/54 | reference |
| 320 held prompts, chat / code / math tok/s | 89.45 / 130.88 / 137.05 | 86.19 / 128.20 / 133.70 |
| our bench, 27-run medians, chat / code | 80.1 / 142.4 | 79.6 / 136.6 |
| our bench, 27 more runs on the live build later that day (no other traffic) | 82.7 / 142.6 | - |
| sparkDash 1.8.8 c1 prose / code (5 runs) | 100.3-100.8 / 162.5-163.1 | 98.8-99.2 / 162.8-163.1 |
| sparkDash catalogs prose / structural / code | 79.5 / 130.7 / 154.0 | 76.3 / 125.2 / 151.2 |
| 16 distinct users / copies / 8 distinct | 365, 366 / 446 / 302 | 370, 366 / 434 / 298 |
| agent load (8 sessions, 80-300K) | 262.4 | 260.4 |
| prefill 8k / 32k tok/s | 8.8k / 10.0k | 8.9k / 10.0k |
| qeval c4 / c16 (of 75) | 71 / 71 | 71 / 72 |
| needle at 948K tokens | 3/3 | 3/3 |
| images | 15/15 | 15/15 |
| slots, context | 16 slots: 4 x 1M + 12 x 128K | same |

The 9- and 27-run medians of our bench swing by several tok/s between runs of the same build (chat runs range 74-95), so the
paired 320-prompt runs and the per-step GPU times are the proof of each lever; the medians are shown for continuity.
