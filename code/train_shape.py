import argparse, collections, glob, json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader

G = 32


def load_shapes(root):
    keys, arrs = [], []
    for f in sorted(glob.glob(os.path.join(root, "cache", "shapes_*.npz"))):
        z = np.load(f)
        keys.append(z["keys"]); arrs.append(z["shapes"])
    if not keys:
        return {}
    K = np.concatenate(keys); A = np.concatenate(arrs)
    return {k: i for i, k in enumerate(K.tolist())}, A


class Shapes(Dataset):

    def __init__(self, rows, index, packed, root, size=256, augment=False, max_inst=16):
        self.root, self.size, self.augment, self.max_inst = root, size, augment, max_inst
        self.index, self.packed = index, packed
        by = collections.defaultdict(list)
        for r in rows:
            by[r["file"]].append(r)
        self.files, self.by = sorted(by), by
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        d = np.load(os.path.join(self.root, f), allow_pickle=False)
        a = np.asarray(d["em"], dtype=np.float32)
        lo, hi = (float(v) for v in np.percentile(a, [1, 99]))
        img = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        ids = d["inst_face"]
        recs = [r for r in self.by[f] if f"{f}|{r['instance']}" in self.index]
        if len(recs) > self.max_inst:
            recs = [recs[j] for j in self.rng.choice(len(recs), self.max_inst, replace=False)]
        px = float(recs[0]["px_nm"]) if recs else 8.0

        keep, shapes = [], []
        for r in recs:
            m = ids == r["instance"]
            if m.sum() < 16:
                continue
            sh = np.unpackbits(self.packed[self.index[f"{f}|{r['instance']}"]])[:G ** 3]
            keep.append(r); shapes.append(sh.reshape(G, G, G).astype(np.float32))
        if not keep:
            return None

        crops, masks, vols, areas = [], [], [], []
        s = self.size
        H, W = ids.shape
        for r in keep:
            m = (ids == r["instance"])
            yy, xx = np.nonzero(m)
            cy, cx = int(yy.mean()), int(xx.mean())
            y0 = int(np.clip(cy - s // 2, 0, max(H - s, 0)))
            x0 = int(np.clip(cx - s // 2, 0, max(W - s, 0)))
            sl = (slice(y0, y0 + s), slice(x0, x0 + s))
            ci, cm = img[sl], m[sl].astype(np.float32)
            if ci.shape != (s, s):
                ph, pw = s - ci.shape[0], s - ci.shape[1]
                ci = np.pad(ci, ((0, ph), (0, pw)))
                cm = np.pad(cm, ((0, ph), (0, pw)))
            crops.append(np.stack([ci, cm, ci * cm]))
            vols.append(np.log(r["volume_nm3"]))
            areas.append(float(m.sum()) * px * px)
        areas = np.array(areas, dtype=np.float32)
        feat = np.stack([np.log(np.maximum(areas, 1.0)),
                         np.full(len(keep), np.log2(px / 8.0), dtype=np.float32)], 1)
        return (torch.from_numpy(np.stack(crops)),
                torch.from_numpy(np.stack(shapes)),
                torch.tensor(vols, dtype=torch.float32),
                torch.from_numpy(feat),
                torch.tensor(areas, dtype=torch.float32))


def collate(b):
    return [x for x in b if x is not None]


def blk2(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())


def blk3(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())


def up3(t):
    return t.repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)


class ShapeNet(nn.Module):

    def __init__(self, cin=3, w=(32, 64, 128, 256), lat=192):
        super().__init__()
        self.enc = nn.Sequential(blk2(cin, w[0]), nn.MaxPool2d(2),
                                 blk2(w[0], w[1]), nn.MaxPool2d(2),
                                 blk2(w[1], w[2]), nn.MaxPool2d(2),
                                 blk2(w[2], w[3]), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.to_lat = nn.Sequential(nn.Linear(w[3] + 2, 256), nn.GELU(), nn.Linear(256, lat))
        self.seed = nn.Linear(lat, 128 * 4 * 4 * 4)
        self.d1, self.d2, self.d3 = blk3(128, 64), blk3(64, 32), blk3(32, 16)
        self.occ = nn.Conv3d(16, 1, 1)
        self.depth = nn.Sequential(nn.Linear(lat, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, feat):
        z = self.to_lat(torch.cat([self.enc(x), feat], 1))
        v = self.seed(z).view(-1, 128, 4, 4, 4)
        v = self.d3(up3(self.d2(up3(self.d1(up3(v))))))
        return self.occ(v)[:, 0], self.depth(z)[:, 0]


def dice_bce(logit, target):
    p = torch.sigmoid(logit)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return dice.mean() + F.binary_cross_entropy_with_logits(logit, target)


def run_fold(rows, index, packed, held, a, dev):
    tr = [r for r in rows if r["dataset"] != held]
    te = [r for r in rows if r["dataset"] == held]
    if len(te) < a.min_test or not tr:
        return None
    rng = np.random.default_rng(0)

    def cap(rs, n):
        fs = sorted({r["file"] for r in rs})
        if len(fs) <= n:
            return rs
        keep = set(rng.choice(fs, n, replace=False))
        return [r for r in rs if r["file"] in keep]
    tr, te = cap(tr, a.max_train_files), cap(te, a.max_test_files)
    vf = set(np.random.default_rng(1).choice(sorted({r["file"] for r in tr}),
                                             max(1, int(0.2 * len({r["file"] for r in tr}))),
                                             replace=False))
    trn = [r for r in tr if r["file"] not in vf]
    val = [r for r in tr if r["file"] in vf] or tr

    mk = lambda rs, aug: DataLoader(Shapes(rs, index, packed, a.root, a.size, aug, a.max_inst),
                                    batch_size=a.batch, shuffle=aug, collate_fn=collate)
    ltr, lva, lte = mk(trn, True), mk(val, False), mk(te, False)

    model = ShapeNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=2)

    def volume_from(occ_logit, box_log, area):
        frac = torch.sigmoid(occ_logit).mean((1, 2, 3)).clamp(min=1e-4)
        return torch.log(frac) + box_log

    def evaluate(loader):
        model.eval(); P, T, I, I0 = [], [], [], []
        with torch.no_grad():
            for batch in loader:
                for crops, shapes, vols, feat, areas in batch:
                    crops, feat, areas = crops.to(dev), feat.to(dev), areas.to(dev)
                    sh = shapes.to(dev)
                    occ, dep = model(crops, feat)
                    P += list(volume_from(occ, dep, areas).cpu().numpy()); T += list(vols.numpy())
                    b = (torch.sigmoid(occ) > 0.5).float()
                    inter = (b * sh).sum((1, 2, 3))
                    union = ((b + sh) > 0).float().sum((1, 2, 3)).clamp(min=1)
                    I += list((inter / union).cpu().numpy())
                    I0 += list((sh.mean((1, 2, 3))).cpu().numpy())
        if not P:
            return float("inf"), None, None, 0.0, 0.0
        P, T = np.array(P), np.array(T)
        return (float(np.median(np.abs(np.exp(P) - np.exp(T)) / np.exp(T))), P, T,
                float(np.mean(I)), float(np.mean(I0)))

    best, best_state, bad, best_ep = float("inf"), None, 0, 0
    for ep in range(a.epochs):
        model.train()
        for batch in ltr:
            if not batch:
                continue
            loss = 0.0
            for crops, shapes, vols, feat, areas in batch:
                crops, shapes, vols = crops.to(dev), shapes.to(dev), vols.to(dev)
                feat, areas = feat.to(dev), areas.to(dev)
                occ, dep = model(crops, feat)
                frac = shapes.mean((1, 2, 3)).clamp(min=1e-4)
                tgt_box = vols - torch.log(frac)
                loss = loss + dice_bce(occ, shapes) \
                    + a.depth_weight * F.l1_loss(dep, tgt_box) \
                    + a.vol_weight * F.l1_loss(volume_from(occ, dep, areas), vols)
            opt.zero_grad(); (loss / max(len(batch), 1)).backward(); opt.step()
        v = evaluate(lva)[0]
        sched.step(v)
        if v < best * (1 - a.min_delta):
            best, best_ep, bad = v, ep + 1, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        print(f"    ep{ep + 1:3d}  val {v:.4f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= a.patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    err, P, T, iou, iou0 = evaluate(lte)
    return dict(held=held, n=0 if P is None else len(P), best_epoch=best_ep, val_err=best,
                med=err, iou=iou, iou_allones=iou0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--table", default="cache/inst_bf_organelle.jsonl")
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-inst", type=int, default=16)
    ap.add_argument("--min-test", type=int, default=200)
    ap.add_argument("--max-train-files", type=int, default=1200)
    ap.add_argument("--max-test-files", type=int, default=400)
    ap.add_argument("--depth-weight", type=float, default=1.0)
    ap.add_argument("--vol-weight", type=float, default=1.0,
                    help="weight on the volume the model actually reports, so the shape and "
                         "depth heads are trained towards agreeing on it")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    index, packed = load_shapes(a.root)
    rows = [json.loads(l) for l in open(os.path.join(a.root, a.table))]
    rows = [r for r in rows if r.get("unbiased") and f"{r['file']}|{r['instance']}" in index]
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ds = sorted({r["dataset"] for r in rows})
    print(f"{len(rows):,} instances with shapes, {len(ds)} volumes, device={dev}", flush=True)
    holds = a.holdout.split(",") if a.holdout else ds

    res = []
    for h in holds:
        r = run_fold(rows, index, packed, h, a, dev)
        if not r:
            print(f"  {h:24s} skipped", flush=True); continue
        res.append(r)
        print(f"  {h:24s} ep{r['best_epoch']:02d}  test {r['med']:.3f}  "
              f"shape IoU {r['iou']:.3f}  (all-ones grid {r['iou_allones']:.3f})", flush=True)
    if res:
        print(f"\nMACRO over {len(res)} volumes   {np.mean([r['med'] for r in res]):.3f}")
    if a.out:
        json.dump(res, open(os.path.join(a.root, a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
