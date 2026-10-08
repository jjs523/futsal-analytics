"""Per-match self-supervised appearance head on the cached OSNet features (research_plan.md V4 step 4).

Pedestrian ReID (OSNet, MSMT17) separates the two teams but barely separates two teammates in the same bib. A small
MLP that re-weights the frozen 512-d vector for THIS match can learn the cues that differ between teammates (shorts,
socks, build) without decoding any video. Labels come for free from pure tracklets:
  - crops of one pure conservative fused tracklet are one person (positives; >= 1 s apart so the head cannot win by
    learning pose continuity, plus the other camera's crops of the same fused tracklet for view invariance);
  - tracklets that coexist in time are different people (negatives; same-team ones are the hard case and weigh 3x).
Tracklets that never coexist may be the same person and are never used as negatives.

Evaluation is the plan's ReID proxy AUC on the held-out time half: positives = crops of one pure tracklet >= 2 s
apart, negatives = crops of coexisting tracklets (all, and same-team only).

Back ends reuse the head unchanged through `with_embedding(data, load_or_train(data))`, which swaps CamData.reid.
Registered A/B variants wrap other modules' registered trackers: ref_<name> (raw OSNet), ssl_<name> (head everywhere),
ssllink_<name> / mixlink_<name> (head, or head mixed 1:1 with OSNet, in the linker; tracklet source on raw OSNet).

    python experiments/ssl_embed.py                                  # head on minutes 0-3, AUC report on minutes 3-6
    python experiments/harness.py ssllink_gta_xv_conservative ssl_embed
    python experiments/metrics2.py ref_gta_xv_conservative ssllink_gta_xv_conservative --parts first,second
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import time
import weakref
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import CACHE, Data, Track, register
from blocks import clean_masks, dbscan_split, normalise, team_vote

SSL_DEFAULTS: dict = dict(
    hidden=256, out=128, skip=False,       # MLP d_in -> hidden -> out; skip adds a linear d_in -> out branch
    use_color=False,                       # append the 30-d sqrt colour histogram to the OSNet input
    loss="infonce", tau=0.1, margin=0.3,   # 'infonce' (multi-positive, weighted negatives) or 'triplet' (batch-hard)
    hard_w=3.0,                            # weight of same-team coexisting negatives
    min_dt_s=1.0, cross_view=True,         # positive pairs: same tracklet >= min_dt_s apart, or the other camera
    epochs=30, P=16, K=8, lr=1e-3, wd=1e-4, in_drop=0.1,
    train_lo=0, train_hi=1800,             # grid frames used for training (first half by default)
    min_span_s=2.0, min_clean=5, eps=0.55,  # pure-tracklet filter (on blocks' default clean crops)
    crop_min_h=60.0, crop_max_iou=0.1,     # which crops of a pure tracklet are trained on (blocks.clean_masks)
    seed=0,
)

_SOURCE: "weakref.WeakKeyDictionary[Data, list[Track]]" = weakref.WeakKeyDictionary()
_RAW: "weakref.WeakKeyDictionary[Data, dict[str, np.ndarray]]" = weakref.WeakKeyDictionary()   # OSNet while swapped


# ---------------------------------------------------------------------------------------------------------------
# Pure tracklets and their clean crops
# ---------------------------------------------------------------------------------------------------------------

def source_tracklets(data: Data) -> list[Track]:
    """xview(source='conservative') fused tracklets on raw OSNet (also when called inside with_embedding), computed
    once per Data (about 6 s)."""
    if data not in _SOURCE:
        from xview import xview
        with with_embedding(data, _RAW[data], pad_to=0) if data in _RAW else contextlib.nullcontext():
            _SOURCE[data] = xview(data, "conservative")
    return _SOURCE[data]


def pure_tracklets(data: Data, lo: int, hi: int, min_span_s: float = 2.0, min_clean: int = 5,
                   eps: float = 0.55) -> list[Track]:
    """Source tracklets clipped to [lo, hi) that span >= min_span_s, have >= min_clean clean crops and stay one DBSCAN
    cluster (dbscan_split run down to min_span_s, not its 5 s default, so short tracklets are tested too). Clipping
    keeps training and evaluation halves disjoint."""
    clean = clean_masks(data)
    out = []
    for t in source_tracklets(data):
        t = {k: p for k, p in t.items() if lo <= k < hi}
        if not t:
            continue
        ks = sorted(t)
        if ks[-1] - ks[0] + 1 < min_span_s * data.rate:
            continue
        if sum(bool(clean[c][i]) for p in t.values() for c, i in p.boxes) < min_clean:
            continue
        if len(dbscan_split(t, data, clean, eps=eps, min_len_s=min_span_s)) != 1:
            continue
        out.append(t)
    return out


@dataclass
class CropSet:
    """Every clean crop (both cameras) of a list of tracklets, as flat arrays."""
    cam: np.ndarray          # (n,) camera position in data.cams order
    idx: np.ndarray          # (n,) box index
    k: np.ndarray            # (n,) grid frame
    tid: np.ndarray          # (n,) tracklet index
    xy: np.ndarray           # (n, 2) the tracklet's pitch position at k
    team: np.ndarray         # (T,) 'Y' / 'N' / 'U' per tracklet
    span: np.ndarray         # (T, 2) first and last grid frame per tracklet
    co: np.ndarray           # (T, T) True where two tracklets are surely two different people

    def same_team(self) -> np.ndarray:
        t = self.team
        return (t[:, None] == t[None]) & (t[:, None] != "U")


def coexisting(tracks: list[Track], dup_m: float = 1.0) -> np.ndarray:
    """(T, T) True where two tracklets share frames and are NOT the same person seen twice: the cross-view pairing
    leaves some cam1-only and cam2-only tracklets of one player unfused, and those run side by side at < dup_m
    median distance. Teammates crossing are close only briefly, so they stay negatives."""
    T = len(tracks)
    span = np.array([[min(t), max(t)] for t in tracks], int).reshape(-1, 2)
    m = (span[:, 0][:, None] <= span[:, 1][None]) & (span[:, 0][None] <= span[:, 1][:, None])
    np.fill_diagonal(m, False)
    for a, b in zip(*np.nonzero(np.triu(m))):
        ks = tracks[a].keys() & tracks[b].keys()
        d = [np.linalg.norm(tracks[a][k].xy - tracks[b][k].xy) for k in ks]
        if not d or np.median(d) < dup_m:
            m[a, b] = m[b, a] = False
    return m


def crop_set(tracks: list[Track], data: Data, clean: dict[str, np.ndarray] | None = None) -> CropSet:
    """`clean`: which boxes count as crops (blocks.clean_masks defaults when None)."""
    clean = clean_masks(data) if clean is None else clean
    pos = {cam: q for q, cam in enumerate(data.cams)}
    rows = [(pos[c], i, k, j) for j, t in enumerate(tracks) for k, p in t.items() for c, i in p.boxes if clean[c][i]]
    r = np.array(rows, int).reshape(-1, 4)
    xy = np.array([tracks[j][k].xy for k, j in zip(r[:, 2], r[:, 3])], float).reshape(-1, 2)
    return CropSet(r[:, 0], r[:, 1], r[:, 2], r[:, 3], xy, np.array([team_vote(t, data)[0] for t in tracks]),
                   np.array([[min(t), max(t)] for t in tracks], int).reshape(-1, 2), coexisting(tracks))


def gather(feats: dict[str, np.ndarray], data: Data, cam: np.ndarray, idx: np.ndarray) -> np.ndarray:
    names = list(data.cams)
    out = np.zeros((len(idx), next(iter(feats.values())).shape[1]), np.float32)
    for q, name in enumerate(names):
        m = cam == q
        out[m] = feats[name][idx[m]]
    return out


# ---------------------------------------------------------------------------------------------------------------
# Feature sets: raw OSNet and the cheap colour baselines
# ---------------------------------------------------------------------------------------------------------------

def raw_feats(data: Data) -> dict[str, np.ndarray]:
    return {cam: normalise(c.reid.astype(np.float32)) for cam, c in data.cams.items()}


def colour_feats(data: Data, part: str = "lower") -> dict[str, np.ndarray]:
    """Hellinger-mapped (sqrt, then L2) HSV histograms: 'upper' (10-50 % of the box: the bib), 'lower' (55-85 %:
    shorts and legs) or 'full' (both). See futsal.pipeline.detect.appearance for the 2 x 15 layout."""
    sl = {"upper": slice(0, 15), "lower": slice(15, 30), "full": slice(0, 30)}[part]
    return {cam: normalise(np.sqrt(c.color[:, sl]).astype(np.float32)) for cam, c in data.cams.items()}


def concat_feats(data: Data, w: float = 0.5, part: str = "full", base: dict[str, np.ndarray] | None = None
                 ) -> dict[str, np.ndarray]:
    """[ReID, w * colour] re-normalised: cos = (cos_reid + w^2 cos_colour) / (1 + w^2)."""
    base = raw_feats(data) if base is None else base
    col = colour_feats(data, part)
    return {cam: normalise(np.hstack([normalise(base[cam]), w * col[cam]])) for cam in data.cams}


def mix_feats(a: dict[str, np.ndarray], b: dict[str, np.ndarray], w: float = 1.0) -> dict[str, np.ndarray]:
    """normalise([a, w * b]) per camera: cos = (cos_a + w^2 cos_b) / (1 + w^2). Mixing the head with raw OSNet pulls
    its distances back towards the OSNet scale that back-end thresholds were tuned on."""
    return {cam: normalise(np.hstack([normalise(a[cam]), w * normalise(b[cam])])) for cam in a}


def _inputs(data: Data, use_color: bool) -> dict[str, np.ndarray]:
    raw = raw_feats(data)
    if not use_color:
        return raw
    col = {cam: np.sqrt(c.color).astype(np.float32) for cam, c in data.cams.items()}
    return {cam: np.hstack([raw[cam], col[cam]]) for cam in data.cams}


# ---------------------------------------------------------------------------------------------------------------
# Head and training
# ---------------------------------------------------------------------------------------------------------------

def _make_head(d_in: int, hidden: int, out: int, skip: bool, in_drop: float):
    import torch.nn as nn
    import torch.nn.functional as F

    class Head(nn.Module):
        """MLP d_in -> hidden -> out, L2-normalised; `skip` adds a linear branch so the head can stay close to the
        OSNet geometry and only re-weight it."""

        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(in_drop)
            self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Linear(hidden, out))
            self.lin = nn.Linear(d_in, out, bias=False) if skip else None

        def forward(self, x):
            x = self.drop(x)
            z = self.mlp(x)
            if self.lin is not None:
                z = z + self.lin(x)
            return F.normalize(z, dim=1)

    return Head()


def _batch(cs: CropSet, by_tid: list[np.ndarray], P: int, K: int, rng: np.random.Generator) -> np.ndarray | None:
    """P tracklets alive at one random instant (so nearly every cross-tracklet pair is a negative; the loss drops the
    rest through CropSet.co), K crops each (with replacement when a tracklet has fewer)."""
    k0 = cs.k[rng.integers(len(cs.k))]
    act = np.flatnonzero((cs.span[:, 0] <= k0) & (cs.span[:, 1] >= k0))
    if len(act) < 2:
        return None
    act = rng.choice(act, min(P, len(act)), replace=False)
    return np.concatenate([rng.choice(by_tid[t], K, replace=len(by_tid[t]) < K) for t in act])


def _loss(z, tid, k, cam, co, team_same, gap: float, p: dict):
    """Weighted multi-positive InfoNCE or batch-hard triplet over one batch. Negatives are only pairs of surely
    different people (`co`); other cross-tracklet pairs are ignored."""
    import torch
    import torch.nn.functional as F
    same = tid[:, None] == tid[None]
    eye = torch.eye(len(tid), dtype=torch.bool, device=z.device)
    far = (k[:, None] - k[None]).abs() >= gap
    if p["cross_view"]:
        far |= cam[:, None] != cam[None]
    pos = same & far & ~eye
    neg = co[tid][:, tid]
    w = 1.0 + (p["hard_w"] - 1.0) * team_same[tid][:, tid].float()
    s = z @ z.T
    if p["loss"] == "infonce":
        logits = s / p["tau"] + torch.log(w)
        lse_neg = torch.logsumexp(logits.masked_fill(~neg, -1e9), dim=1, keepdim=True)
        st = s / p["tau"]
        per = -st + torch.logaddexp(st, lse_neg.expand_as(st))
        m = pos & neg.any(1, keepdim=True)
        return per[m].mean() if m.any() else s.sum() * 0
    d = 1.0 - s
    dp = d.masked_fill(~pos, -1.0).max(1).values
    hard = team_same[tid][:, tid] & neg
    easy = ~team_same[tid][:, tid] & neg
    dn_h = d.masked_fill(~hard, 9.0).min(1).values
    dn_e = d.masked_fill(~easy, 9.0).min(1).values
    has_p, has_h, has_e = pos.any(1), hard.any(1), easy.any(1)
    l = p["hard_w"] * F.relu(dp - dn_h + p["margin"]) * has_h + F.relu(dp - dn_e + p["margin"]) * has_e
    wsum = p["hard_w"] * has_h + has_e
    m = has_p & (wsum > 0)
    return (l[m] / wsum[m]).mean() if m.any() else s.sum() * 0


def train_head(data: Data, **params) -> tuple[dict[str, np.ndarray], dict]:
    """Train the head on pure tracklets inside [train_lo, train_hi) and embed every box of both cameras.
    Returns ({cam: (n_boxes, out) L2-normalised}, training stats)."""
    import torch
    p = SSL_DEFAULTS | params
    t0 = time.time()
    torch.manual_seed(p["seed"])
    rng = np.random.default_rng(p["seed"])
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tracks = pure_tracklets(data, p["train_lo"], p["train_hi"], p["min_span_s"], p["min_clean"], p["eps"])
    cs = crop_set(tracks, data, clean_masks(data, min_h=p["crop_min_h"], max_iou=p["crop_max_iou"]))
    X = _inputs(data, p["use_color"])
    Xt = torch.from_numpy(gather(X, data, cs.cam, cs.idx)).to(dev)
    tid_t = torch.from_numpy(cs.tid).to(dev)
    k_t = torch.from_numpy(cs.k).to(dev)
    cam_t = torch.from_numpy(cs.cam).to(dev)
    team_same = torch.from_numpy(cs.same_team()).to(dev)
    co_t = torch.from_numpy(cs.co).to(dev)
    by_tid = [np.flatnonzero(cs.tid == t) for t in range(len(tracks))]
    head = _make_head(Xt.shape[1], p["hidden"], p["out"], p["skip"], p["in_drop"]).to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=p["lr"], weight_decay=p["wd"])
    steps_per_epoch = max(20, len(cs.k) // (p["P"] * p["K"]))
    total = p["epochs"] * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=p["lr"], total_steps=total, pct_start=0.1)
    head.train()
    losses = []
    step = misses = 0
    while step < total:
        b = _batch(cs, by_tid, p["P"], p["K"], rng)
        if b is None:
            misses += 1
            if misses > 1000:
                raise RuntimeError("no instant with two coexisting pure tracklets: nothing to contrast")
            continue
        misses = 0
        bt = torch.from_numpy(b).to(dev)
        loss = _loss(head(Xt[bt]), tid_t[bt], k_t[bt], cam_t[bt], co_t, team_same, p["min_dt_s"] * data.rate, p)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        step += 1
    head.eval()
    out = {}
    with torch.no_grad():
        for cam, x in X.items():
            xt = torch.from_numpy(x).to(dev)
            out[cam] = torch.cat([head(xt[i:i + 8192]) for i in range(0, len(xt), 8192)]).cpu().numpy().astype(np.float32)
    stats = {"tracklets": len(tracks), "crops": int(len(cs.k)), "crops_cam2": int((cs.cam == 1).sum()),
             "steps": total, "loss_first": round(float(np.mean(losses[:50])), 4),
             "loss_last": round(float(np.mean(losses[-50:])), 4), "train_s": round(time.time() - t0, 1),
             "device": str(dev)}
    return out, stats


def _tag(data: Data, p: dict) -> str:
    key = json.dumps(p, sort_keys=True) + "|" + ",".join(f"{c}:{len(d.t)}" for c, d in data.cams.items())
    return hashlib.md5(key.encode()).hexdigest()[:10]


def load_or_train(data: Data, tag: str | None = None, cache: str = CACHE, **params) -> dict[str, np.ndarray]:
    """{cam: (n_boxes, d) L2-normalised} head embeddings of every box, cached as experiments/cache/ssl_<tag>_<cam>.npy
    (tag = hash of the parameters and the box counts unless given). Training settings and stats go to
    ssl_<tag>.json next to them."""
    p = SSL_DEFAULTS | params
    tag = tag or _tag(data, p)
    paths = {cam: os.path.join(cache, f"ssl_{tag}_{cam}.npy") for cam in data.cams}
    if all(os.path.exists(f) for f in paths.values()):
        feats = {cam: np.load(f) for cam, f in paths.items()}
        if all(len(feats[cam]) == len(c.t) for cam, c in data.cams.items()):
            return feats
    feats, stats = train_head(data, **p)
    for cam, f in paths.items():
        np.save(f, feats[cam])
    json.dump({"params": p, "stats": stats}, open(os.path.join(cache, f"ssl_{tag}.json"), "w"), indent=1)
    return feats


@contextlib.contextmanager
def with_embedding(data: Data, feats: dict[str, np.ndarray], pad_to: int = 512):
    """Temporarily replace every CamData.reid by `feats` (L2-normalised, float32), so any back end that reads
    CamData.reid runs unchanged. Zero-padding to `pad_to` columns keeps every cosine identical while code that
    assumes 512-d (blocks.clean_feats' empty case, boxmot's dummy embedder) keeps working; pad_to=0 disables it.
    Distances change scale, so appearance gates tuned on OSNet (app_gate, app_distance thresholds) need retuning:
    see `gate_like_raw`."""
    saved = {cam: c.reid for cam, c in data.cams.items()}
    outer = data not in _RAW
    if outer:
        _RAW[data] = saved
    try:
        for cam, c in data.cams.items():
            f = normalise(np.asarray(feats[cam], np.float32))
            if len(f) != len(c.t):
                raise ValueError(f"{cam}: {len(f)} features for {len(c.t)} boxes")
            if pad_to and f.shape[1] < pad_to:
                f = np.hstack([f, np.zeros((len(f), pad_to - f.shape[1]), np.float32)])
            c.reid = f
        yield data
    finally:
        for cam, c in data.cams.items():
            c.reid = saved[cam]
        if outer:
            del _RAW[data]


# ---------------------------------------------------------------------------------------------------------------
# ReID proxy AUC (research_plan.md, label-free metric 7)
# ---------------------------------------------------------------------------------------------------------------

@dataclass
class ProxyPairs:
    """Fixed crop pairs (and 2 s chunk pairs) so every embedding is scored on exactly the same evidence."""
    a: tuple[np.ndarray, np.ndarray]       # (cam, idx) of the first crop of each pair
    b: tuple[np.ndarray, np.ndarray]
    label: np.ndarray                      # 1 = same tracklet >= min_dt apart, 2 = ... and >= far_m apart on the
                                           # pitch (location-shortcut control), 0 = coexisting tracklets
    same_team: np.ndarray                  # negatives of two tracklets with the same clear team vote
    chunks: list[tuple[np.ndarray, np.ndarray]]   # (cam, idx) crops of each 2 s chunk
    chunk_pairs: np.ndarray                # (m, 2) chunk indices
    chunk_label: np.ndarray
    chunk_same_team: np.ndarray
    tracklets: int


def proxy_pairs(data: Data, lo: int, hi: int, min_dt_s: float = 2.0, max_pos: int = 200, max_neg: int = 50,
                far_m: float = 5.0, chunk_s: float = 2.0, seed: int = 0, crops: dict[str, np.ndarray] | None = None,
                **pure) -> ProxyPairs:
    """Positives: crops of one pure tracklet >= min_dt_s apart (either camera), at most max_pos per tracklet.
    Negatives: crops of two coexisting tracklets, at most max_neg per tracklet pair. The caps keep a few long
    tracklets from dominating. A second positive set keeps only pairs >= far_m apart on the pitch: OSNet vectors
    also encode background, and a head could score well by learning image location, which would not help linking
    across gaps. Chunk level: the mean of each tracklet's clean crops per chunk_s window, which is the
    evidence a linker actually compares. `crops` picks the boxes to pair (default: blocks' clean crops), e.g. the
    small or overlapped boxes that the default mask rejects."""
    rng = np.random.default_rng(seed)
    tracks = pure_tracklets(data, lo, hi, **pure)
    cs = crop_set(tracks, data, crops)
    by = [np.flatnonzero(cs.tid == t) for t in range(len(tracks))]
    co, st = cs.co, cs.same_team()
    A, B, L, S = [], [], [], []
    gap = min_dt_s * data.rate
    for t, ix in enumerate(by):
        i, j = np.triu_indices(len(ix), 1)
        ok = np.abs(cs.k[ix[i]] - cs.k[ix[j]]) >= gap
        i, j = i[ok], j[ok]
        far = np.linalg.norm(cs.xy[ix[i]] - cs.xy[ix[j]], axis=1) >= far_m
        for sel, lab in ((np.arange(len(i)), 1), (np.flatnonzero(far), 2)):
            if len(sel) > max_pos:
                sel = rng.choice(sel, max_pos, replace=False)
            A.append(ix[i[sel]]); B.append(ix[j[sel]]); L.append(np.full(len(sel), lab)); S.append(np.zeros(len(sel), bool))
    for t, u in zip(*np.nonzero(np.triu(co))):
        n = min(max_neg, len(by[t]) * len(by[u]))
        A.append(rng.choice(by[t], n)); B.append(rng.choice(by[u], n))
        L.append(np.zeros(n)); S.append(np.full(n, bool(st[t, u])))
    none = [np.zeros(0)]
    A, B = np.concatenate(A or none).astype(int), np.concatenate(B or none).astype(int)
    L, S = np.concatenate(L or none), np.concatenate(S or none).astype(bool)
    # chunks
    chunks, owner, ck = [], [], []
    w = int(chunk_s * data.rate)
    for t, ix in enumerate(by):
        for c in np.unique(cs.k[ix] // w):
            m = ix[cs.k[ix] // w == c]
            chunks.append((cs.cam[m], cs.idx[m])); owner.append(t); ck.append(c)
    owner, ck = np.array(owner, int), np.array(ck, int)
    i, j = np.triu_indices(len(chunks), 1)
    pos = (owner[i] == owner[j]) & (np.abs(ck[i] - ck[j]) >= 1)
    neg = co[owner[i], owner[j]]
    keep = pos | neg
    i, j = i[keep], j[keep]
    return ProxyPairs((cs.cam[A], cs.idx[A]), (cs.cam[B], cs.idx[B]), L, S,
                      chunks, np.stack([i, j], 1), (owner[i] == owner[j]).astype(float),
                      st[owner[i], owner[j]] & (owner[i] != owner[j]), len(tracks))


def proxy_auc(data: Data, feats: dict[str, np.ndarray], pp: ProxyPairs) -> dict:
    """AUC of cosine similarity, positives vs all coexisting negatives and vs same-team negatives only, for single
    crops and for 2 s chunk means; plus the median cosine distances, which back ends need to re-set their gates."""
    from sklearn.metrics import roc_auc_score
    fa = normalise(gather(feats, data, *pp.a))
    fb = normalise(gather(feats, data, *pp.b))
    s = (fa * fb).sum(1)
    pos, far, neg = pp.label == 1, pp.label == 2, pp.label == 0
    hard = neg & pp.same_team

    def auc(sp, sn):
        if not len(sp) or not len(sn):
            return float("nan")
        return round(float(roc_auc_score(np.r_[np.ones(len(sp)), np.zeros(len(sn))], np.r_[sp, sn])), 4)

    cs_ = np.zeros(len(pp.chunk_pairs))
    if len(pp.chunk_pairs):
        cm = normalise(np.stack([gather(feats, data, c, i).mean(0) for c, i in pp.chunks]))
        cs_ = (cm[pp.chunk_pairs[:, 0]] * cm[pp.chunk_pairs[:, 1]]).sum(1)
    cpos, cneg = pp.chunk_label == 1, pp.chunk_label == 0
    chard = cneg & pp.chunk_same_team
    return {"auc": auc(s[pos], s[neg]), "auc_same_team": auc(s[pos], s[hard]),
            "auc_far_same_team": auc(s[far], s[hard]) if far.any() else float("nan"),
            "chunk_auc": auc(cs_[cpos], cs_[cneg]), "chunk_auc_same_team": auc(cs_[cpos], cs_[chard]),
            "d_pos_med": round(float(np.median(1 - s[pos])), 3) if pos.any() else float("nan"),
            "d_neg_same_med": round(float(np.median(1 - s[hard])), 3) if hard.any() else float("nan"),
            "n_pos": int(pos.sum()), "n_pos_far": int(far.sum()), "n_neg": int(neg.sum()), "n_neg_same": int(hard.sum())}


def closed_set_check(data: Data, feats: dict[str, np.ndarray], lo: int, hi: int, ks: tuple[int, ...] = (5, 6),
                     **pure) -> dict:
    """Label-free identity test that does not saturate like the AUC: cluster each team's pure tracklets (mean of
    their clean crops) into K groups (average linkage, cosine) and report the conflict share = time-overlap of
    coexisting tracklets that land in one group / all coexisting overlap. An embedding that knows the 5 players
    of a team gives ~0 conflicts at K=5 and five ~0.2 size shares; chance is ~1/K."""
    from sklearn.cluster import AgglomerativeClustering
    tracks = pure_tracklets(data, lo, hi, **pure)
    cs = crop_set(tracks, data)
    X = gather(feats, data, cs.cam, cs.idx)
    M = normalise(np.stack([X[cs.tid == t].mean(0) for t in range(len(tracks))]))
    out = {}
    for team in ("Y", "N"):
        ix = np.flatnonzero(cs.team == team)
        shared = np.array([[len(tracks[a].keys() & tracks[b].keys()) if cs.co[a, b] else 0 for b in ix] for a in ix])
        size = np.array([len(tracks[t]) for t in ix], float)
        for K in ks:
            if len(ix) <= K:
                continue
            lab = AgglomerativeClustering(n_clusters=K, metric="cosine", linkage="average").fit_predict(M[ix])
            out[f"conflict_{team}{K}"] = round(float((shared * (lab[:, None] == lab[None])).sum() / max(shared.sum(), 1)), 3)
            share = np.sort(np.bincount(lab, weights=size, minlength=K))[::-1] / size.sum()
            out[f"sizes_{team}{K}"] = [round(float(x), 2) for x in share]
    return out


def gate_like_raw(data: Data, feats: dict[str, np.ndarray], pp: ProxyPairs, raw_gate: float) -> dict:
    """Carry an OSNet-tuned cosine-distance gate over to `feats` by quantile matching on the proxy positives: the
    returned gate accepts the same share of same-person pairs as `raw_gate` does on raw OSNet, and the dict also says
    how many same-team negatives each gate lets through."""
    def dist(f, m):
        fa, fb = normalise(gather(f, data, *pp.a)), normalise(gather(f, data, *pp.b))
        return 1 - (fa * fb).sum(1)[m]
    pos, hard = pp.label == 1, (pp.label == 0) & pp.same_team
    raw = raw_feats(data)
    q = float((dist(raw, pos) <= raw_gate).mean())
    gate = float(np.quantile(dist(feats, pos), q))
    return {"gate": round(gate, 3), "pos_accepted": round(q, 3),
            "neg_same_accepted_raw": round(float((dist(raw, hard) <= raw_gate).mean()), 4),
            "neg_same_accepted": round(float((dist(feats, hard) <= gate).mean()), 4)}


# ---------------------------------------------------------------------------------------------------------------
# Registered variants: other back ends re-run unchanged with the head (trained on the first half only, so the
# second-half metrics are held out). 'ref_*' re-runs the same back end on raw OSNet under a separate result name,
# so the A/B uses one code version and never overwrites the owner's results.
# ---------------------------------------------------------------------------------------------------------------

BACKENDS = {                     # registered name -> module that registers it
    "xv_conservative": "xview",
    "lc_xvcons": "link_closed",
    "gta_xv_conservative": "link_gta",
}


@contextlib.contextmanager
def raw_tracklet_sources(data: Data):
    """Inside with_embedding, keep xview.xview (the per-camera tracklet builders + cross-view pairing) on raw OSNet.
    The builders' appearance gates also judge small and overlapped boxes, which the head never saw in training: with
    the head they cut ~20 % more tracklets (ssl_xv_conservative). Back ends import xview.xview at call time, so
    swapping the module attribute reaches them without touching their code."""
    import xview as xv
    orig = xv.xview

    def xview_raw(d: Data, *a, **kw):
        if d is not data or d not in _RAW:
            return orig(d, *a, **kw)
        with with_embedding(d, _RAW[d], pad_to=0):
            return orig(d, *a, **kw)
    xv.xview = xview_raw
    try:
        yield
    finally:
        xv.xview = orig


def backend_feats(data: Data, kind: str, **params) -> dict[str, np.ndarray]:
    """'ssl': the head; 'mix': the head mixed 1:1 with raw OSNet (mix_feats)."""
    f = load_or_train(data, **params)
    return mix_feats(f, raw_feats(data), 1.0) if kind == "mix" else f


def run_backend(data: Data, name: str, module: str, mode: str, kind: str = "ssl", **params) -> list[Track]:
    """Run the tracker registered as `name` in `module` with mode 'raw' (OSNet), 'ssl' (the `kind` features
    everywhere) or 'link' (the `kind` features in the linker, raw OSNet in the xview tracklet source)."""
    import importlib
    import harness
    importlib.import_module(module)
    fn = harness.TRACKERS[name]
    if mode == "raw":
        return fn(data)
    feats = backend_feats(data, kind, **params)
    with contextlib.ExitStack() as stack:
        if mode == "link":
            stack.enter_context(raw_tracklet_sources(data))
        stack.enter_context(with_embedding(data, feats))
        return fn(data)


for _name, _mod in BACKENDS.items():
    for _prefix, _mode, _kind in (("ref", "raw", ""), ("ssl", "ssl", "ssl"), ("ssllink", "link", "ssl"),
                                  ("mixlink", "link", "mix")):
        if _mod == "xview" and _mode == "link":             # nothing but the source there
            continue
        register(f"{_prefix}_{_name}")(lambda data, _n=_name, _m=_mod, _o=_mode, _k=_kind: run_backend(data, _n, _m, _o, _k))


# ---------------------------------------------------------------------------------------------------------------
# CLI: AUC report, head trained on minutes 0-3, everything scored on minutes 3-6
# ---------------------------------------------------------------------------------------------------------------

def report(data: Data, configs: dict[str, dict] | None = None, swap: bool = False) -> dict[str, dict]:
    """Proxy AUC + closed-set check on the held-out half for raw OSNet, the colour baselines and heads trained on the
    other half (first -> second by default; swap=True trains on the second and scores the first)."""
    configs = configs if configs is not None else {"ssl_default": {}}
    half = data.n // 2
    (tlo, thi), (elo, ehi) = ((half, data.n), (0, half)) if swap else ((0, half), (half, data.n))
    test = proxy_pairs(data, elo, ehi)
    feats = {"raw_osnet": raw_feats(data), "colour_lower": colour_feats(data, "lower"),
             "colour_full": colour_feats(data, "full")}
    for w in (0.5, 1.0):
        feats[f"concat_w{w}"] = concat_feats(data, w)
    for name, p in configs.items():
        feats[name] = load_or_train(data, **({"train_lo": tlo, "train_hi": thi} | p))
        feats[f"{name}+colour_w0.5"] = concat_feats(data, 0.5, base=feats[name])
    feats["mix(ssl_default,raw)"] = mix_feats(feats[next(iter(configs))], feats["raw_osnet"])
    return {name: proxy_auc(data, f, test) | closed_set_check(data, f, elo, ehi) for name, f in feats.items()}


if __name__ == "__main__":
    _data = Data()
    _keys = ["auc", "auc_same_team", "auc_far_same_team", "chunk_auc_same_team", "d_pos_med", "d_neg_same_med",
             "conflict_Y5", "conflict_Y6", "conflict_N5", "conflict_N6"]
    for _swap in (False, True):
        _res = report(_data, {"ssl_default": {}, "ssl_out64": {"out": 64}, "ssl_triplet": {"loss": "triplet"},
                              "ssl_color_in": {"use_color": True}}, swap=_swap)
        print()
        print(f"{'train 2nd, score 1st' if _swap else 'train 1st, score 2nd':26s}" +"".join(f"{k[:19]:>20s}" for k in _keys))
        for _n, _r in _res.items():
            print(f"{_n:26s}" + "".join(f"{_r.get(k, ''):>20}" for k in _keys))
            print(f"{'':26s}  sizes Y5 {_r.get('sizes_Y5')}  N5 {_r.get('sizes_N5')}  N6 {_r.get('sizes_N6')}")
