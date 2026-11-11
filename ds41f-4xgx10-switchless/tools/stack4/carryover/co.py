#!/usr/bin/env python3
"""co.py BASE TRAIN_DIR HELD_DIR OUT [--steps N] [--k 192] [--lr 1e-3] [--rank 64] [--lora 0] [--emax 4] [--eval-only]
Carryover pilot (arXiv 2609.14717 adapted to DSpark on dsgx10): the main_hidden of the REJECTED rows of the last verify step
enters the drafter's attention as extra read-only keys for one round, at the positions they were computed, through main_proj plus
a learned low-rank correction and one learned marker embedding, with a learned attention bias per block. The shipped drafter is
frozen; only the marker, the correction, the bias (and optional gated adapters) train. Data: co_prep.py request files (served steps).
Scores, at the SAME anchors (those that follow a rejection): free-running d1/d2/d3 accuracy and accepted length, shipped vs carry."""
import sys, os, glob, json, time, random, math
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsm3 as dsm
from dsm3 import NB, WIN, HEADS, HD, NOISE, HC, OG, ORK, BF, rmsnorm, rope_tail, act_quant
a = sys.argv
def opt(k, d, f=float): return f(a[a.index(k) + 1]) if k in a else d
CK, TRD, HLD, OUT = a[1], a[2], a[3], a[4]; os.makedirs(OUT, exist_ok=True)
STEPS, KB, LR, RANK, LORA, EMAX = opt("--steps", 600, int), opt("--k", 192, int), opt("--lr", 1e-3), opt("--rank", 64, int), opt("--lora", 0, int), opt("--emax", 4, int)
EVAL_EVERY, NHELD, BIAS0 = opt("--eval-every", 150, int), opt("--nheld", 60, int), opt("--bias0", -2.0)
DW = [1, 0.78, 0.61, 0.47, 0.37]
dev = "cuda"; torch.manual_seed(1); random.seed(1)
log = open(OUT + "/log.txt", "a")
def say(*x):
    s = " ".join(str(v) for v in x); print(s, flush=True); log.write(s + "\n"); log.flush()
say("args", " ".join(a[1:]))
m = dsm.Drafter(CK, dev=dev, lazy_experts=False, nblk=3)
for p in m.parameters(): p.requires_grad_(False)
m.refresh()

class Carry(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.marker = torch.nn.Parameter(torch.zeros(dsm.DIM))
        s.A = torch.nn.Parameter(torch.randn(RANK, 3 * dsm.DIM) * 0.01); s.B = torch.nn.Parameter(torch.zeros(dsm.DIM, RANK))
        s.bias = torch.nn.Parameter(torch.full((3,), BIAS0))
        if LORA:
            s.la = torch.nn.ParameterList([torch.nn.Parameter(torch.randn(LORA, dsm.DIM) * 0.01) for _ in range(6)])
            s.lb = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(dsm.DIM, LORA)) for _ in range(6)])
C = Carry().to(dev)

def carried_kv(rmh, rpos):
    """rmh [K, E, 15360] bf16, rpos [K, E] -> per block [K, E, 512]"""
    x = m.lin8(rmh, "mtp.0.main_proj").float() + (rmh.float() @ C.A.t()) @ C.B.t()
    mx = (rmsnorm(x.to(BF), m.q["mtp.0.main_norm.weight"]).float() + C.marker).to(BF)
    out = []
    for b in range(3):
        kv = rmsnorm(m.lin8(mx, f"mtp.{b}.attn.wkv"), m.q[f"mtp.{b}.attn.kv_norm.weight"])
        out.append(act_quant(rope_tail(kv, rpos)))
    return out

def attn(h, b, anchors, mkv, apos, lo, rkv, rmask):
    K = h.size(0); pre = f"mtp.{b}.attn."
    bpos = apos.unsqueeze(1) + 1 + torch.arange(NB, device=h.device).unsqueeze(0)
    qr = rmsnorm(m.lin8(h, pre + "wq_a"), m.q[pre + "q_norm.weight"])
    q = m.lin8(qr, pre + "wq_b").unflatten(-1, (HEADS, HD)); q = rope_tail(q, bpos.unsqueeze(-1))
    kv = rmsnorm(m.lin8(h, pre + "wkv"), m.q[pre + "kv_norm.weight"]); kv = act_quant(rope_tail(kv, bpos))
    wpos = anchors.unsqueeze(1) - (WIN - 1) + torch.arange(WIN, device=h.device).unsqueeze(0)
    valid = wpos >= lo.unsqueeze(1)
    wkv = mkv[wpos.clamp_min(0)]
    parts = [wkv, kv]; masks = [valid, torch.ones(K, NB, dtype=torch.bool, device=h.device)]
    if rkv is not None: parts.append(rkv[b]); masks.append(rmask)
    keys = torch.cat(parts, 1).float(); mask = torch.cat(masks, 1)
    sc = torch.einsum("kmhd,knd->kmhn", q.float(), keys) * (HD ** -0.5)
    if rkv is not None:
        add = torch.zeros(keys.size(1), device=h.device); add[WIN + NB:] = 1.0
        sc = sc + add * C.bias[b]
    sc = sc.masked_fill(~mask[:, None, None, :], float("-inf"))
    sink = m.q[pre + "attn_sink"].view(1, 1, HEADS, 1).expand(K, NB, HEADS, 1)
    p = torch.softmax(torch.cat([sc, sink], -1), -1)[..., :-1]
    o = torch.einsum("kmhn,knd->kmhd", p, keys).to(h.dtype)
    o = rope_tail(o, bpos.unsqueeze(-1), inverse=True)
    o = act_quant(o.reshape(K, NB, OG, -1)).to(BF)
    woa = m.q[pre + "wo_a"].view(OG, ORK, -1)
    o = torch.einsum("ksgd,grd->ksgr", o, woa).flatten(2)
    return m.lin8(o, pre + "wo_b"), (p[..., WIN + NB:].sum(-1).mean().item() if rkv is not None else 0.0)

def blocks(pk, an, carry):
    """pk: a pack of sequences; an [K] indices into the pack (anchors). carry: use the anchors' rejected rows."""
    K = an.numel()
    rkv = rmask = None
    if carry:
        rkv = carried_kv(pk["rmh"][pk["ai"][an]], pk["rpos"][pk["ai"][an]]); rmask = pk["rmask"][pk["ai"][an]]
    ids = torch.full((K, NB), NOISE, dtype=torch.long); ids[:, 0] = pk["toks"][an.cpu() + 1]
    x4 = m.embed[ids].to(dev).unsqueeze(2).repeat(1, 1, HC, 1)
    pre_mix = torch.zeros(K, NB, HC, device=dev); pre_mix[..., 0] = 1.0
    att = 0.0
    for b in range(3):
        res = x4
        a_pre, a_post, a_comb = m.hc_mixes(x4, b, "attn")
        h = rmsnorm(m.hc_pre(x4, pre_mix), m.q[f"mtp.{b}.attn_norm.weight"])
        ao, at = attn(h, b, an, pk["mkv"][b], pk["pos"][an], pk["lo"][an], rkv, rmask); att += at / 3
        if LORA and carry: ao = (ao.float() + (h.float() @ C.la[2 * b].t()) @ C.lb[2 * b].t()).to(ao.dtype)
        x4 = m.hc_post(ao, res, a_post, a_comb)
        res = x4
        f_pre, f_post, f_comb = m.hc_mixes(x4, b, "ffn")
        h = rmsnorm(m.hc_pre(x4, a_pre), m.q[f"mtp.{b}.ffn_norm.weight"])
        fo = m.ffn(h, b)
        if LORA and carry: fo = (fo.float() + (h.float() @ C.la[2 * b + 1].t()) @ C.lb[2 * b + 1].t()).to(fo.dtype)
        x4 = m.hc_post(fo, res, f_post, f_comb)
        pre_mix = f_pre
    return m.hc_pre(x4, pre_mix), att

def load_pack(files):
    """several requests side by side: toks, mh, pos (position in its own sequence), lo (the sequence's first index), the anchors
    that follow a rejection (index into the pack) with their rejected rows padded to EMAX."""
    toks, mh, pos, lo, an, rmh, rpos, rmask, meta = [], [], [], [], [], [], [], [], []
    off = 0
    for f in files:
        d = np.load(f); T = len(d["toks"])
        toks.append(d["toks"].astype(np.int64)); mh.append(d["mh"]); pos.append(np.arange(T)); lo.append(np.full(T, off))
        for (aa, k0, n) in d["an"]:
            if n < 1 or aa + 2 >= T: continue
            e = min(int(n), EMAX)
            r = np.zeros((EMAX, 15360), np.uint16); r[:e] = d["rej_mh"][k0:k0 + e]
            p = np.zeros(EMAX, np.int64); p[:e] = d["rej_pos"][k0:k0 + e]
            mk = np.zeros(EMAX, bool); mk[:e] = True
            an.append(off + int(aa)); rmh.append(r); rpos.append(p); rmask.append(mk); meta.append((int(d["qi"]), int(aa), T - int(aa) - 3))
        off += T
    if not an: return None
    pk = dict(toks=torch.from_numpy(np.concatenate(toks)), pos=torch.from_numpy(np.concatenate(pos)).to(dev), lo=torch.from_numpy(np.concatenate(lo)).to(dev),
              an=torch.tensor(an, device=dev), rmh=torch.from_numpy(np.stack(rmh).view(np.int16)).view(torch.bfloat16).to(dev) if False else None, meta=meta)
    mhb = torch.from_numpy(np.concatenate(mh).view(np.int16)).to(dev).view(torch.bfloat16)
    pk["rmh"] = torch.from_numpy(np.stack(rmh).view(np.int16)).to(dev).view(torch.bfloat16)
    pk["rpos"] = torch.from_numpy(np.stack(rpos)).to(dev); pk["rmask"] = torch.from_numpy(np.stack(rmask)).to(dev)
    pk["ai"] = torch.full((off,), -1, dtype=torch.long, device=dev); pk["ai"][pk["an"]] = torch.arange(len(an), device=dev)
    with torch.no_grad():
        mkv = []
        for c0 in range(0, off, 2048):       # main_kv in chunks (memory)
            part = m.main_kv(mhb[c0:c0 + 2048], pk["pos"][c0:c0 + 2048]); mkv.append(part)
        pk["mkv"] = [torch.cat([p[b] for p in mkv]) for b in range(3)]
    pk["toksd"] = pk["toks"].to(dev)
    pk["left"] = torch.tensor([x[2] for x in meta], device=dev)
    return pk

AR = torch.arange(NB, device=dev)
def tf_loss(pk, an, carry):
    xh, att = blocks(pk, an, carry)
    lg = m.logits(xh)
    tp = an.unsqueeze(1) + 1 + AR
    ok = (AR.unsqueeze(0) <= pk["left"][pk["ai"][an]].unsqueeze(1))
    tpc = tp.clamp_max(pk["toksd"].numel() - 2)
    bias, _ = m.markov(pk["toksd"][tpc])
    lp = torch.log_softmax((lg + bias).float(), -1)
    label = pk["toksd"][tpc + 1]
    ce = -lp.gather(-1, label.unsqueeze(-1)).squeeze(-1)
    w = ok.float() * torch.tensor(DW, device=dev)
    hit = ((lg + bias).argmax(-1) == label).float()
    return (ce * w).sum() / w.sum().clamp_min(1), [float((hit[:, j] * ok[:, j]).sum() / ok[:, j].sum().clamp_min(1)) for j in range(3)], att

@torch.no_grad()
def chain(pk, an, carry):
    xh, _ = blocks(pk, an, carry)
    lg = m.logits(xh)
    prev = pk["toksd"][an + 1]; ds = []
    for j in range(NB):
        bias, _ = m.markov(prev); prev = (lg[:, j] + bias).argmax(-1); ds.append(prev)
    return torch.stack(ds, 1)

def evaluate(files, tag, dump=None):
    C.eval(); tot = np.zeros((2, NB)); n = np.zeros(NB); lead = np.zeros(2); N = 0; rows = []
    for i in range(0, len(files), 6):
        pk = load_pack(files[i:i + 6])
        if pk is None: continue
        for j0 in range(0, pk["an"].numel(), 256):
            an = pk["an"][j0:j0 + 256]; K = an.numel()
            tp = (an.unsqueeze(1) + 2 + AR).clamp_max(pk["toksd"].numel() - 1)
            truth = pk["toksd"][tp]; ok = AR.unsqueeze(0) <= pk["left"][pk["ai"][an]].unsqueeze(1)
            D = [chain(pk, an, False), chain(pk, an, True)]
            for v in range(2):
                hit = (D[v] == truth) & ok; cum = hit.cumprod(1)
                tot[v] += cum.sum(0).cpu().numpy(); lead[v] += cum.sum().item()
            n += ok.sum(0).cpu().numpy(); N += K
            if dump is not None:
                l0 = ((D[0] == truth) & ok).cumprod(1).sum(1).cpu().tolist(); l1 = ((D[1] == truth) & ok).cumprod(1).sum(1).cpu().tolist()
                for q, (mt, x0, x1) in enumerate(zip(pk["meta"][j0:j0 + K], l0, l1)): rows.append((mt[0], mt[1], int(pk["rmask"][pk["ai"][an[q]]].sum()), x0, x1))
    C.train()
    say(f"{tag}: {N} anchors after a rejection | chain right through depth 1/2/3: shipped " + "/".join("%.4f" % (tot[0][j] / max(1, n[j])) for j in range(3)) + " carry " + "/".join("%.4f" % (tot[1][j] / max(1, n[j])) for j in range(3)) + f" | accepted length (of 5 drafted) shipped {lead[0]/max(1,N):.4f} carry {lead[1]/max(1,N):.4f} ({100*(lead[1]/max(1e-9,lead[0])-1):+.2f} %)")
    if dump is not None: json.dump(rows, open(dump, "w"))
    return lead[1] / max(1e-9, lead[0])

trf = sorted(glob.glob(TRD + "/q*.npz")); hf = sorted(glob.glob(HLD + "/q*.npz"))
random.Random(0).shuffle(hf); hsel = sorted(hf[:NHELD])
say("train requests", len(trf), "held requests", len(hf), "eval on", len(hsel))
if os.path.exists(OUT + "/carry.pt") and "--resume" in a: C.load_state_dict(torch.load(OUT + "/carry.pt"))
if "--eval-only" in a:
    evaluate(sorted(hf), "held, all", OUT + "/held_rows.json"); sys.exit(0)
evaluate(hsel, "step 0")
params = [dict(params=[C.marker, C.A, C.B], lr=LR), dict(params=[C.bias], lr=LR * 10)]
if LORA: params.append(dict(params=list(C.la) + list(C.lb), lr=LR))
optz = torch.optim.AdamW(params, weight_decay=0.0)
sched = torch.optim.lr_scheduler.OneCycleLR(optz, max_lr=[g["lr"] for g in params], total_steps=STEPS, pct_start=0.1)
order = list(trf); random.shuffle(order); cur = 0; t0 = time.time(); best = 0; run = []
for step in range(1, STEPS + 1):
    if cur + 4 > len(order): random.shuffle(order); cur = 0
    pk = load_pack(order[cur:cur + 4]); cur += 4
    if pk is None: continue
    sel = pk["an"][torch.randperm(pk["an"].numel(), device=dev)[:KB]]
    loss, hit, att = tf_loss(pk, sel, True)
    optz.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(C.parameters(), 1.0); optz.step(); sched.step()
    run.append((loss.item(), hit[0], att))
    if step % 25 == 0:
        r = np.mean(run[-25:], 0); say(f"step {step} loss {r[0]:.4f} tf d1 {r[1]:.4f} attention on carried rows {r[2]:.4f} bias {[round(x, 2) for x in C.bias.tolist()]} |marker| {C.marker.norm().item():.3f} {time.time()-t0:.0f}s")
    if step % EVAL_EVERY == 0 or step == STEPS:
        g = evaluate(hsel, f"step {step}")
        if g > best: best = g; torch.save(C.state_dict(), OUT + "/carry.pt")
C.load_state_dict(torch.load(OUT + "/carry.pt"))
evaluate(sorted(hf), "FINAL held, all requests (best checkpoint)", OUT + "/held_rows.json")
