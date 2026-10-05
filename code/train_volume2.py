import argparse, collections, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader


def boundary_features(img, mask, gmag):
    m = mask > 0.5
    if m.sum() < 8:
        return np.zeros(NBOUND, dtype=np.float32)
    er = ndimage.binary_erosion(m, iterations=1)
    di = ndimage.binary_dilation(m, iterations=2)
    band = di & ~er
    out = di & ~m
    ins = er
    g = gmag[band] if band.any() else np.array([0.0], np.float32)
    ii = img[ins] if ins.any() else img[m]
    oo = img[out] if out.any() else img[m]
    im = img[m]
    f = [g.mean(), g.std(), np.percentile(g, 90),
         float(band.sum()) / max(float(m.sum()), 1.0),
         ii.mean() - oo.mean(), ii.std(), oo.std(),
         im.std(), float(np.percentile(im, 90) - np.percentile(im, 10))]
    return np.nan_to_num(np.array(f, dtype=np.float32))

SPHERE = 4.0 / 3.0 / np.sqrt(np.pi)

GEO = {"blockface":   dict(img="em",    inst="inst_face",
                           shape=("face_feret_nm",)),
       "thinsection": dict(img="image", inst="inst_section",
                           shape=("profile_feret_nm",)),
       "surface":     dict(img="se",    inst="inst_front",
                           shape=("proj_feret_nm", "proj_major_nm", "proj_minor_nm",
                                  "proj_perimeter_nm"))}
MAXSHAPE = 4
NBOUND = 9
NFEAT = 2 + 2 * MAXSHAPE + NBOUND


class Views(Dataset):

    def __init__(self, rows, vocab, geometry="blockface", size=256, augment=False,
                 max_inst=48, balance=False):
        self.size, self.augment, self.max_inst = size, augment, max_inst
        self.vocab, self.geo = vocab, GEO[geometry]
        self.target = "volume_nm3"
        by = collections.defaultdict(list)
        for r in rows:
            by[r["file"]].append(r)
        self.files, self.by = sorted(by), by
        self.rng = np.random.default_rng(0)
        c = collections.Counter(r["organelle"] for r in rows)
        self.w = {k: (1.0 / v) ** 0.5 for k, v in c.items()} if balance else None

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        d = np.load(os.path.join(self.root, f), allow_pickle=False)
        a = np.asarray(d[self.geo["img"]], dtype=np.float32)
        if a.ndim == 3:
            a = a[0]
        lo, hi = (float(v) for v in np.percentile(a, [1, 99]))
        img = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        raw = (a / 255.0).astype(np.float32)
        gy, gx = np.gradient(img)
        gmag = np.sqrt(gy * gy + gx * gx).astype(np.float32)
        ids = d[self.geo["inst"]]
        recs = self.by[f]
        if len(recs) > self.max_inst:
            if self.w:
                p = np.array([self.w.get(r["organelle"], 1.0) for r in recs], dtype=np.float64)
                p /= p.sum()
                recs = [recs[j] for j in self.rng.choice(len(recs), self.max_inst,
                                                         replace=False, p=p)]
            else:
                recs = [recs[j] for j in self.rng.choice(len(recs), self.max_inst, replace=False)]

        H, W = ids.shape
        s = self.size
        if H >= s and W >= s:
            y0 = int(self.rng.integers(0, H - s + 1)) if self.augment else (H - s) // 2
            x0 = int(self.rng.integers(0, W - s + 1)) if self.augment else (W - s) // 2
            sl = (slice(y0, y0 + s), slice(x0, x0 + s))
            img, raw, gmag, ids = img[sl], raw[sl], gmag[sl], ids[sl]
        px = float(recs[0]["px_nm"])
        keep = [r for r in recs if (ids == r["instance"]).sum() >= 16]
        if not keep:
            return None
        masks = np.stack([(ids == r["instance"]).astype(np.float32) for r in keep])
        ph, pw = (-img.shape[0]) % 8, (-img.shape[1]) % 8
        if ph or pw:
            img, raw, gmag = (np.pad(z, ((0, ph), (0, pw))) for z in (img, raw, gmag))
            masks = np.pad(masks, ((0, 0), (0, ph), (0, pw)))
        y = np.array([np.log(r[self.target]) for r in keep], dtype=np.float32)
        areas = np.array([float(m.sum()) * px * px for m in masks], dtype=np.float32)
        base = np.log(np.maximum(areas.astype(np.float64), 1.0) ** 1.5 * SPHERE).astype(np.float32)
        la = np.log(np.maximum(areas, 1.0))
        cols = [la, np.full(len(keep), np.log2(px / 8.0), dtype=np.float32)]
        for j in range(MAXSHAPE):
            keys = self.geo["shape"]
            if j < len(keys):
                v = np.maximum(np.array([float(r.get(keys[j]) or 0.0) for r in keep],
                                        dtype=np.float32), px)
                cols += [np.log(v), np.log(v) - 0.5 * la]
            else:
                z = np.zeros(len(keep), dtype=np.float32)
                cols += [z, z]
        feat = np.stack(cols, 1).astype(np.float32)
        bf = np.stack([boundary_features(img, mk, gmag) for mk in masks])
        feat = np.concatenate([feat, bf], 1).astype(np.float32)
        cls = np.array([self.vocab.get(r["organelle"], 0) for r in keep], dtype=np.int64)
        return (torch.from_numpy(np.stack([img, raw, gmag])), torch.from_numpy(masks),
                torch.from_numpy(y),
                torch.from_numpy(base), torch.from_numpy(feat), torch.from_numpy(cls),
                [r["organelle"] for r in keep])


def collate(b):
    return [x for x in b if x is not None]


def blk(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())


class Net(nn.Module):

    def __init__(self, ncls, w=(32, 64, 128, 256), nfeat=NFEAT, cin=3):
        super().__init__()
        self.e1, self.e2, self.e3, self.e4 = blk(cin, w[0]), blk(w[0], w[1]), \
            blk(w[1], w[2]), blk(w[2], w[3])
        self.pool = nn.MaxPool2d(2)
        c = sum(w)
        self.trunk = nn.Sequential(nn.Linear(c + nfeat, 256), nn.GELU(),
                                   nn.Linear(256, 128), nn.GELU())
        self.head = nn.Linear(128, 1)
        self.cls = nn.Linear(128, ncls)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def start_at(self, log_offset):
        with torch.no_grad():
            self.head.bias.fill_(float(log_offset))

    def forward(self, x, masks, feat):
        f1 = self.e1(x[None] if x.dim() == 3 else x)
        f2 = self.e2(self.pool(f1))
        f3 = self.e3(self.pool(f2))
        f4 = self.e4(self.pool(f3))
        vecs = []
        for f in (f1, f2, f3, f4):
            m = F.interpolate(masks[None], size=f.shape[-2:], mode="area")[0]
            m = m / m.sum((1, 2), keepdim=True).clamp(min=1e-6)
            vecs.append(torch.einsum("chw,nhw->nc", f[0], m))
        z = self.trunk(torch.cat(vecs + [feat], 1))
        return self.head(z)[:, 0], self.cls(z)


def stats(pred_log, true_log):
    p, t = np.exp(pred_log), np.exp(true_log)
    rel = np.abs(p - t) / t
    return dict(med=float(np.median(rel)), p75=float(np.percentile(rel, 75)),
                p90=float(np.percentile(rel, 90)),
                mdlog=float(np.median(np.abs(pred_log - true_log))), n=len(t))


def run_fold(rows, vocab, held, a, dev, root):
    tr = [r for r in rows if r["dataset"] != held]
    te = [r for r in rows if r["dataset"] == held]
    if len(te) < a.min_test or not tr:
        return None
    rng = np.random.default_rng(0)
    for split, cap in ((tr, a.max_train_files), (te, a.max_test_files)):
        pass
    def cap_files(rs, n):
        fs = sorted({r["file"] for r in rs})
        if len(fs) <= n:
            return rs
        keep = set(rng.choice(fs, n, replace=False))
        return [r for r in rs if r["file"] in keep]
    tr, te = cap_files(tr, a.max_train_files), cap_files(te, a.max_test_files)

    files = sorted({r["file"] for r in tr})
    nval = max(1, int(0.20 * len(files)))
    vfiles = set(np.random.default_rng(1).choice(files, nval, replace=False))
    trn = [r for r in tr if r["file"] not in vfiles]
    val = [r for r in tr if r["file"] in vfiles]
    if not trn or not val:
        trn, val = tr, tr

    dtr = Views(trn, vocab, a.geometry, a.size, True, a.max_inst, balance=not a.no_balance)
    dva = Views(val, vocab, a.geometry, a.size, False, 512)
    dte = Views(te, vocab, a.geometry, a.size, False, 512)
    for d in (dtr, dva, dte):
        d.root, d.target = root, a.target
    lva = DataLoader(dva, batch_size=a.batch, collate_fn=collate)
    ltr = DataLoader(dtr, batch_size=a.batch, shuffle=True, collate_fn=collate)
    lte = DataLoader(dte, batch_size=a.batch, collate_fn=collate)

    off = float(np.median([np.log(r[a.target]) - np.log(max(r["v_stereo"], 1.0))
                           for r in trn]))
    model = Net(len(vocab) + 1).to(dev)
    model.start_at(off)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=2)
    wt = torch.ones(len(vocab) + 1)
    if not a.no_balance:
        cc = collections.Counter(r["organelle"] for r in trn)
        for k, v in cc.items():
            if k in vocab:
                wt[vocab[k]] = (1.0 / v) ** 0.5
        wt = wt / wt.mean()
    wt = wt.to(dev)

    def val_err():
        model.eval()
        P, T = [], []
        with torch.no_grad():
            for batch in lva:
                for img, masks, y, base, feat, cls, _ in batch:
                    img, masks, base, feat = (t.to(dev) for t in (img, masks, base, feat))
                    res, _ = model(img, masks, feat)
                    P += list((base + res).cpu().numpy()); T += list(y.numpy())
        if not P:
            return float("inf")
        P, T = np.array(P), np.array(T)
        return float(np.median(np.abs(np.exp(P) - np.exp(T)) / np.exp(T)))

    best, best_state, bad, best_ep = float("inf"), None, 0, 0
    for ep in range(a.epochs):
        model.train()
        for batch in ltr:
            if not batch:
                continue
            loss = 0.0
            for img, masks, y, base, feat, cls, _ in batch:
                img, masks, y, base, feat, cls = (t.to(dev) for t in
                                                  (img, masks, y, base, feat, cls))
                res, logit = model(img, masks, feat)
                w = wt[cls]
                loss = loss + ((base + res - y).abs() * w).sum() / w.sum().clamp(min=1e-6) \
                    + a.cls_weight * F.cross_entropy(logit, cls, weight=wt)
            opt.zero_grad(); (loss / max(len(batch), 1)).backward(); opt.step()
        v = val_err()
        sched.step(v)
        if v < best * (1.0 - a.min_delta):
            best, best_ep, bad = v, ep + 1, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        print(f"    ep{ep + 1:3d}  val {v:.4f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= a.patience:
            print(f"    early stop at epoch {ep + 1}, best was {best_ep} ({best:.4f})",
                  flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    P, T, B, O = [], [], [], []
    hit = tot = 0
    with torch.no_grad():
        for batch in lte:
            for img, masks, y, base, feat, cls, orgs in batch:
                img, masks, base, feat = (t.to(dev) for t in (img, masks, base, feat))
                res, logit = model(img, masks, feat)
                P += list((base + res).cpu().numpy()); T += list(y.numpy())
                B += list(base.cpu().numpy()); O += orgs
                hit += int((logit.argmax(1).cpu() == cls).sum()); tot += len(cls)
    P, T, B = np.array(P), np.array(T), np.array(B)
    out = dict(held=held, n=len(T), cls_acc=hit / max(tot, 1),
               best_epoch=best_ep, val_err=best,
               model=stats(P, T), stereo=stats(B, T), per_class={},
               preds=dict(pred=[round(float(x), 5) for x in P],
                          true=[round(float(x), 5) for x in T],
                          base=[round(float(x), 5) for x in B], org=list(O)))
    for org in sorted(set(O)):
        s = np.array([o == org for o in O])
        if s.sum() >= 30:
            out["per_class"][org] = dict(model=stats(P[s], T[s]), stereo=stats(B[s], T[s]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--table", default="cache/inst_bf_organelle.jsonl")
    ap.add_argument("--geometry", default="blockface", choices=sorted(GEO))
    ap.add_argument("--target", default="volume_nm3",
                    help="volume_nm3 for the whole organelle, v_below_nm3 for only what "
                         "remains below the cut plane, which is the part the electrons "
                         "actually sampled")
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--min-delta", type=float, default=0.002,
                    help="relative improvement that counts, so epoch-to-epoch noise on a "
                         "small validation split does not exhaust patience")
    ap.add_argument("--patience", type=int, default=8,
                    help="epochs without validation improvement before stopping")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-inst", type=int, default=48)
    ap.add_argument("--min-test", type=int, default=200)
    ap.add_argument("--max-train-files", type=int, default=2500)
    ap.add_argument("--max-test-files", type=int, default=600)
    ap.add_argument("--cls-weight", type=float, default=0.3)
    ap.add_argument("--no-balance", action="store_true",
                    help="also drop class-frequency sampling and loss weights, so the run "
                         "touches organelle labels nowhere at all")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(os.path.join(a.root, a.table))]
    rows = [r for r in rows if r.get("unbiased") and r.get(a.target, 0) > 0]
    names = sorted({r["organelle"] for r in rows})
    vocab = {n: i + 1 for i, n in enumerate(names)}
    ds = sorted({r["dataset"] for r in rows})
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{len(rows):,} instances, {len(ds)} volumes, {len(names)} classes, "
          f"geometry={a.geometry}, target={a.target}, device={dev}", flush=True)
    holds = a.holdout.split(",") if a.holdout else ds

    res = []
    for h in holds:
        r = run_fold(rows, vocab, h, a, dev, a.root)
        if not r:
            print(f"  {h:24s} skipped", flush=True); continue
        res.append(r)
        print(f"  {h:24s} n={r['n']:6,d} ep{r.get('best_epoch', 0):02d} "
              f"model {r['model']['med']:.3f}  stereo {r['stereo']['med']:.3f}  "
              f"{'WIN' if r['model']['med'] < r['stereo']['med'] else 'lose'}", flush=True)

    if res:
        mm = np.mean([r["model"]["med"] for r in res])
        ss = np.mean([r["stereo"]["med"] for r in res])
        print(f"\nMACRO over {len(res)} volumes   model {mm:.3f}   stereo {ss:.3f}   "
              f"clsAcc {np.mean([r['cls_acc'] for r in res]):.3f}")
        agg = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in res:
            for o, d in r["per_class"].items():
                agg[o]["model"].append(d["model"]["med"])
                agg[o]["stereo"].append(d["stereo"]["med"])
                agg[o]["n"].append(d["model"]["n"])
        print(f"\n{'class':12s} {'folds':>5s} {'n':>8s} {'model':>8s} {'stereo':>8s}")
        for o in sorted(agg, key=lambda k: -sum(agg[k]["n"])):
            d = agg[o]
            print(f"{o:12s} {len(d['n']):5d} {sum(d['n']):8,d} "
                  f"{np.mean(d['model']):8.3f} {np.mean(d['stereo']):8.3f}")
        cm = np.mean([np.mean(d["model"]) for d in agg.values()])
        cs = np.mean([np.mean(d["stereo"]) for d in agg.values()])
        print(f"\nequal-class macro   model {cm:.3f}   stereo {cs:.3f}   "
              f"<- the average the first attempt was hiding")
    if a.out:
        json.dump(res, open(os.path.join(a.root, a.out), "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
