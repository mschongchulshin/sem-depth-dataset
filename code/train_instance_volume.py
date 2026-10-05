import argparse, json, os, collections
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = "/Volumes/One Touch/em-depth-dataset"
SPHERE = 4.0 / 3.0 / np.sqrt(np.pi)
GEO = {"blockface": ("em", "inst_face"), "thinsection": ("image", "inst_section"),
       "surface": ("se", "inst_front")}


class InstanceCrops(Dataset):

    def __init__(self, rows, geometry, size=256, augment=False, max_inst=48):
        self.geometry, self.size, self.augment, self.max_inst = geometry, size, augment, max_inst
        by = collections.defaultdict(list)
        for r in rows:
            by[r["file"]].append(r)
        self.files = sorted(by)
        self.by = by
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        imk, ik = GEO[self.geometry]
        d = np.load(os.path.join(ROOT, f), allow_pickle=False)
        a = np.asarray(d[imk], dtype=np.float32)
        lo, hi = (float(v) for v in np.percentile(a, [1, 99]))
        img = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        ids = d[ik]
        recs = self.by[f]
        if len(recs) > self.max_inst:
            recs = list(self.rng.choice(recs, self.max_inst, replace=False))

        H, W = ids.shape
        s = self.size
        if H >= s and W >= s:
            y0 = (H - s) // 2 if not self.augment else int(self.rng.integers(0, H - s + 1))
            x0 = (W - s) // 2 if not self.augment else int(self.rng.integers(0, W - s + 1))
            img, ids = img[y0:y0 + s, x0:x0 + s], ids[y0:y0 + s, x0:x0 + s]
        px = float(recs[0]["px_nm"])

        keep = [r for r in recs if (ids == r["instance"]).sum() >= 16]
        if not keep:
            return None
        masks = np.stack([(ids == r["instance"]).astype(np.float32) for r in keep])
        y = np.array([np.log(r["volume_nm3"]) for r in keep], dtype=np.float32)
        areas = np.array([float(m.sum()) * px * px for m in masks], dtype=np.float32)
        base = np.log(np.maximum(areas.astype(np.float64), 1.0) ** 1.5 * SPHERE).astype(np.float32)
        pxc = np.full((1, *img.shape), float(np.log2(px / 8.0)), dtype=np.float32)
        x = np.concatenate([img[None], pxc], 0).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(masks), torch.from_numpy(y),
                torch.from_numpy(base),
                [r["organelle"] for r in keep], [r["dataset"] for r in keep],
                [r.get("tier", "?") for r in keep])


def collate(b):
    return [x for x in b if x is not None]


class Net(nn.Module):
    def __init__(self, cin=2, w=(32, 64, 128)):
        super().__init__()
        layers, c = [], cin
        for ww in w:
            layers += [nn.Conv2d(c, ww, 3, padding=1), nn.GroupNorm(8, ww), nn.GELU(),
                       nn.Conv2d(ww, ww, 3, padding=1), nn.GroupNorm(8, ww), nn.GELU()]
            c = ww
        self.enc = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(c + 2, 128), nn.GELU(),
                                  nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, x, masks, log_area, log_px):
        fmap = self.enc(x[None])[0]
        m = masks / masks.sum((1, 2), keepdim=True).clamp(min=1)
        pooled = torch.einsum("chw,nhw->nc", fmap, m)
        z = torch.cat([pooled, log_area[:, None], log_px.expand(len(pooled))[:, None]], 1)
        return self.head(z)[:, 0]


def r2(pred, true):
    p, t = np.asarray(pred), np.asarray(true)
    ss = ((t - p) ** 2).sum()
    tot = ((t - t.mean()) ** 2).sum()
    return float(1 - ss / tot) if tot > 0 else float("nan")


def run_fold(rows, held, a, dev):
    tr = [r for r in rows if r["dataset"] != held]
    te = [r for r in rows if r["dataset"] == held]
    if len(te) < a.min_test or not tr:
        return None
    if a.max_train_files:
        seen, keep = set(), []
        for r in tr:
            if r["file"] not in seen and len(seen) >= a.max_train_files:
                continue
            seen.add(r["file"]); keep.append(r)
        tr = keep
    import collections as _c
    _by = _c.defaultdict(list); _byc = _c.defaultdict(list); _all = []
    for r in tr:
        d = np.log(r["volume_nm3"]) - np.log(r["v_stereo"])
        _by[(r["organelle"], r.get("tier", "?"))].append(d)
        _byc[r["organelle"]].append(d); _all.append(d)
    CAL = {k: float(np.median(v)) for k, v in _by.items()}
    CALC = {k: float(np.median(v)) for k, v in _byc.items()}
    GCAL = float(np.median(_all)) if _all else 0.0
    def cal_of(org, tier):
        return CAL.get((org, tier), CALC.get(org, GCAL))

    dtr = InstanceCrops(tr, a.geometry, a.size, augment=True)
    dte = InstanceCrops(te, a.geometry, a.size)
    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, collate_fn=collate)
    lte = DataLoader(dte, batch_size=a.batch, shuffle=False, collate_fn=collate)

    model = Net().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    for ep in range(a.epochs):
        model.train()
        for batch in ltr:
            opt.zero_grad()
            loss = 0.0
            for x, masks, y, base, orgs, _, tiers in batch:
                off = torch.tensor([cal_of(o, t) for o, t in zip(orgs, tiers)],
                                   dtype=torch.float32)
                x, masks, y, base, off = (t.to(dev) for t in (x, masks, y, base, off))
                bc = base + off
                resid = model(x, masks, bc, x[1, 0, 0].reshape(1))
                loss = loss + F.smooth_l1_loss(bc + resid, y)
            if isinstance(loss, float):
                continue
            (loss / max(len(batch), 1)).backward()
            opt.step()

    model.eval()
    P, T, B, C, O = [], [], [], [], []
    with torch.no_grad():
        for batch in lte:
            for x, masks, y, base, orgs, _, tiers in batch:
                off = torch.tensor([cal_of(o, t) for o, t in zip(orgs, tiers)],
                                   dtype=torch.float32)
                x, masks, base, off = (t.to(dev) for t in (x, masks, base, off))
                bc = base + off
                p = (bc + model(x, masks, bc, x[1, 0, 0].reshape(1))).cpu().numpy()
                P += list(p); T += list(y.numpy())
                B += list(base.cpu().numpy()); C += list(bc.cpu().numpy()); O += orgs
    P, T, B, C = np.array(P), np.array(T), np.array(B), np.array(C)
    err = lambda lp: float(np.median(np.abs(np.exp(lp) - np.exp(T)) / np.exp(T)))
    out = dict(held=held, n=len(T), model_err=err(P), stereo_err=err(B), cal_err=err(C),
               model_r2=r2(P, T), stereo_r2=r2(B, T), cal_r2=r2(C, T), per_class={})
    for org in sorted(set(O)):
        s = np.array([o == org for o in O])
        if s.sum() >= 30:
            out["per_class"][org] = dict(
                n=int(s.sum()),
                model_err=float(np.median(np.abs(np.exp(P[s]) - np.exp(T[s])) / np.exp(T[s]))),
                stereo_err=float(np.median(np.abs(np.exp(B[s]) - np.exp(T[s])) / np.exp(T[s]))),
                model_r2=r2(P[s], T[s]), stereo_r2=r2(B[s], T[s]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--geometry", default="blockface", choices=sorted(GEO))
    ap.add_argument("--holdout", default=None, help="comma list, default every volume")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-test", type=int, default=200)
    ap.add_argument("--max-train-files", type=int, default=600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.table)]
    rows = [r for r in rows if r.get("unbiased")]
    ds = sorted({r["dataset"] for r in rows})
    holds = a.holdout.split(",") if a.holdout else ds
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{len(rows):,} unbiased records, {len(ds)} volumes, device={dev}", flush=True)

    res = []
    for h in holds:
        r = run_fold(rows, h, a, dev)
        if not r:
            print(f"  {h}: skipped", flush=True); continue
        res.append(r)
        print(f"  {h:24s} n={r['n']:6,d}  model {r['model_err']:.3f}"
              f"  cal {r['cal_err']:.3f}  raw {r['stereo_err']:.3f}"
              f"   R2 {r['model_r2']:+.3f}/{r['cal_r2']:+.3f}"
              f"   {'WIN' if r['model_err'] < r['cal_err'] else 'lose'}", flush=True)
    if res:
        print(f"\nMACRO over {len(res)} held-out volumes")
        print(f"  model              {np.mean([r['model_err'] for r in res]):.3f}   "
              f"R2 {np.mean([r['model_r2'] for r in res]):+.3f}")
        print(f"  class x tier calib {np.mean([r['cal_err'] for r in res]):.3f}   "
              f"R2 {np.mean([r['cal_r2'] for r in res]):+.3f}   <- the number to beat")
        print(f"  raw stereology     {np.mean([r['stereo_err'] for r in res]):.3f}   "
              f"R2 {np.mean([r['stereo_r2'] for r in res]):+.3f}")
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in res:
            for org, d in r["per_class"].items():
                for k, v in d.items():
                    agg[org][k].append(v)
        print(f"\n{'class':12s} {'folds':>5s} {'model err':>10s} {'stereo err':>11s} "
              f"{'model R2':>9s} {'stereo R2':>10s}")
        for org in sorted(agg, key=lambda o: -sum(agg[o]["n"])):
            d = agg[org]
            print(f"{org:12s} {len(d['n']):5d} {np.mean(d['model_err']):10.3f} "
                  f"{np.mean(d['stereo_err']):11.3f} {np.mean(d['model_r2']):+9.3f} "
                  f"{np.mean(d['stereo_r2']):+10.3f}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
