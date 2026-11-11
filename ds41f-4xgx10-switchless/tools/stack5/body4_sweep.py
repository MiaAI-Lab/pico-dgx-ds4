"""body4_sweep.py: the lever C measurement of docs/stack5.md, as the block that was added to our drafter trainer.

Not runnable on its own: it runs inside the torch mirror of the DSpark drafter used for the drafter3 / stack4 work (a model
object `m` with `dense_names`, `p(name)`, `q[name]` = the fake-quantised FP8 weights as bf16, `refresh()`, and an
`evaluate()` that scores teacher-forced accuracy per depth and the accepted length on held-out sequences). The mirror and
the feature dumps are not in this repository. What the block does: after every refresh it replaces the chosen dense tensors
by their MXFP4 rounding (e2m1 values 0, 0.5, 1, 1.5, 2, 3, 4, 6 with sign; one power-of-two scale per 32 values of a row,
2^ceil(log2(amax / 6)); "+mse" picks per block the better of that scale and half of it), then evaluates.

  --body4-sweep "none@all@all+mse@attn@shared_experts@main_proj@wq_a,wq_b,wkv@wo_a,wo_b"
"""
import json, sys
import torch

SWEEP = opt("--body4-sweep", "", str)
if SWEEP:
    # stack5 lever C: the drafter's dense FP8 tensors quantised post hoc to MXFP4 (e2m1 values, one power-of-two scale per
    # 32 values of a row), scored on the held sequences. SPEC@SPEC@...: none | all | substrings of tensor names (a,b,..);
    # +mse picks per block the better of the two scales 2^ceil(log2(amax/6)) and half of it.
    FP4T = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)
    def mx4(w, mse):
        o, i = w.shape; wf = w.float().reshape(o, i // 32, 32)
        amax = wf.abs().amax(-1, keepdim=True).clamp_min(1e-30)
        e0 = torch.ceil(torch.log2(amax / 6.0))
        def q(e):
            s_ = torch.exp2(e); x = (wf.abs() / s_).clamp(max=6.0)
            idx = (x.unsqueeze(-1) - FP4T).abs().argmin(-1)
            return FP4T[idx] * s_ * wf.sign()
        best = q(e0)
        if mse:
            err = (best - wf).square().sum(-1, keepdim=True)
            c = q(e0 - 1); ec = (c - wf).square().sum(-1, keepdim=True)
            best = torch.where(ec < err, c, best)
        return best.reshape(o, i).to(w.dtype)
    base_refresh = m.refresh
    for spec in SWEEP.split("@"):
        mse = spec.endswith("+mse"); sp = spec[:-4] if mse else spec
        pick = [k for k in m.dense_names if sp != "none" and (sp == "all" or any(t_ in k for t_ in sp.split(",")))]
        def rf():
            base_refresh()
            for k in pick:
                m.q[k] = mx4(m.q[k], mse)
        m.refresh = rf
        rel = 0.0
        if pick:
            base_refresh(); num = den = 0.0
            for k in pick:
                w0 = m.q[k].float(); w1 = mx4(m.q[k], mse).float(); num += float((w1 - w0).square().sum()); den += float(w0.square().sum())
            rel = (num / den) ** 0.5
        say("sweep", spec, "tensors", len(pick), "fp8_bytes", sum(m.p(k).numel() for k in pick), "rel_err", round(rel, 4), json.dumps(evaluate()))
    sys.exit(0)
