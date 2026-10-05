import argparse, collections, json, os, re
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

COMPACT = ("mito", "nucleus", "endo", "lyso", "ld", "nucleolus")
from torch.utils.data import Dataset, DataLoader

def dataset_of(path):
    parts = path.replace("\\", "/").split("/")
    box = parts[1] if len(parts) > 2 else parts[-1]
    return box.split("__")[0]


def load_compact(table):
    by = collections.defaultdict(set)
    for line in open(table):
        r = json.loads(line)
        if r["organelle"] in COMPACT:
            by[r["file"]].add(int(r["instance"]))
    return by


class Faces(Dataset):

    def __init__(self, files, keep, root, size=192, augment=False):
        self.files, self.keep, self.root = files, keep, root
        self.size, self.augment = size, augment
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        d = np.load(os.path.join(self.root, f), allow_pickle=False)
        em = np.asarray(d["em"], dtype=np.float32)
        lo, hi = (float(v) for v in np.percentile(em, [1, 99]))
        img = np.clip((em - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        dep = np.asarray(d["depth_below_nm"], dtype=np.float32)
        inst = np.asarray(d["inst_face"], dtype=np.int32)
        m = np.asarray(d["mask"], dtype=bool)
        if "clipped" in d.files:
            m &= ~np.asarray(d["clipped"], dtype=bool)
        m &= np.isfinite(dep) & (dep > 0)
        ok = self.keep.get(f, ())
        m &= np.isin(inst, list(ok)) if ok else False

        edge = np.zeros_like(m)
        edge[:-1, :] |= inst[:-1, :] != inst[1:, :]
        edge[1:, :] |= inst[:-1, :] != inst[1:, :]
        edge[:, :-1] |= inst[:, :-1] != inst[:, 1:]
        edge[:, 1:] |= inst[:, :-1] != inst[:, 1:]
        px0 = float(d["px_nm"]) if "px_nm" in d.files else 8.0
        dist = (distance_transform_edt(~edge) * px0).astype(np.float32)

        s = self.size
        H, W = img.shape
        if H < s or W < s:
            pad = ((0, max(0, s - H)), (0, max(0, s - W)))
            img, dep, dist = (np.pad(a, pad) for a in (img, dep, dist))
            m = np.pad(m, pad, constant_values=False)
            H, W = img.shape
        y0 = int(self.rng.integers(0, H - s + 1)) if self.augment else (H - s) // 2
        x0 = int(self.rng.integers(0, W - s + 1)) if self.augment else (W - s) // 2
        sl = (slice(y0, y0 + s), slice(x0, x0 + s))
        img, dep, dist, m = img[sl], dep[sl], dist[sl], m[sl]

        scale = np.full_like(img, np.float32(np.log2(px0 / 8.0)))
        x = np.stack([img, scale, np.log1p(dist).astype(np.float32)], 0)
        ld = np.log(np.maximum(dep, 1.0)).astype(np.float32)
        return (torch.from_numpy(x), torch.from_numpy(ld), torch.from_numpy(dist),
                torch.from_numpy(m))


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.GELU())


class UNet(nn.Module):

    def __init__(self, cin=3, w=(32, 64, 128, 256)):
        super().__init__()
        self.e1, self.e2, self.e3 = block(cin, w[0]), block(w[0], w[1]), block(w[1], w[2])
        self.bott = block(w[2], w[3])
        self.d3 = block(w[3] + w[2], w[2])
        self.d2 = block(w[2] + w[1], w[1])
        self.d1 = block(w[1] + w[0], w[0])
        self.out = nn.Conv2d(w[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def start_at(self):
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        up = lambda t, r: F.interpolate(t, size=r.shape[-2:], mode="nearest")
        d3 = self.d3(torch.cat([up(b, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return self.out(d1)[:, 0]


def metrics(pred_log, true_log, m):
    if m.sum() == 0:
        return None
    p, t = pred_log[m].exp(), true_log[m].exp()
    ratio = torch.maximum(p / t, t / p)
    return dict(mae_nm=float((p - t).abs().mean()),
                rmse_nm=float(torch.sqrt(((p - t) ** 2).mean())),
                absrel=float(((p - t).abs() / t).mean()),
                rmse_log=float(torch.sqrt(((pred_log[m] - true_log[m]) ** 2).mean())),
                d1=float((ratio < 1.25).float().mean()),
                d2=float((ratio < 1.25 ** 2).float().mean()),
                d3=float((ratio < 1.25 ** 3).float().mean()),
                n=int(m.sum()))


def agg(rows):
    rows = [r for r in rows if r]
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    return {k: sum(r[k] * r["n"] for r in rows) / n for k in rows[0] if k != "n"} | {"n": n}


DIST_EDGES = np.array([0, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 1e9], dtype=np.float32)


def bins(dist):
    return torch.from_numpy(np.digitize(np.asarray(dist), DIST_EDGES) - 1).clamp(
        0, len(DIST_EDGES) - 1)


def fit_baselines(loader):
    allv, per = [], collections.defaultdict(list)
    for _, ld, dist, m in loader:
        if m.sum() == 0:
            continue
        allv.append(ld[m])
        b = bins(dist[m].numpy())
        for k in torch.unique(b).tolist():
            per[int(k)].append(ld[m][b == k])
    if not allv:
        return None
    gl = float(torch.cat(allv).median())
    lut = torch.full((len(DIST_EDGES),), gl)
    for k, v in per.items():
        if 0 <= k < len(lut):
            lut[k] = float(torch.cat(v).median())
    return gl, lut


def run_fold(files, keep, held, a, dev):
    tr = [f for f in files if dataset_of(f) != held]
    te = [f for f in files if dataset_of(f) == held]
    if len(te) < 20 or not tr:
        return None
    rng = np.random.default_rng(0)
    if len(tr) > a.max_train:
        tr = list(rng.choice(tr, a.max_train, replace=False))
    if len(te) > a.max_test:
        te = list(rng.choice(te, a.max_test, replace=False))
    mk = lambda fl, aug: DataLoader(Faces(fl, keep, a.root, a.size, aug),
                                    batch_size=a.batch, shuffle=aug)
    ltr, lte = mk(tr, True), mk(te, False)
    fit = fit_baselines(ltr)
    if not fit:
        return None
    gl, lut = fit
    lut = lut.to(dev)

    model = UNet().to(dev)
    model.start_at()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr,
                                                total_steps=max(1, a.epochs * len(ltr)))
    for _ in range(a.epochs):
        model.train()
        for x, ld, dist, m in ltr:
            x, ld, m = x.to(dev), ld.to(dev), m.to(dev)
            if m.sum() == 0:
                continue
            base = lut[bins(dist).to(dev)]
            loss = F.l1_loss((base + model(x))[m], ld[m])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()

    model.eval()
    arms = {k: [] for k in ("model", "geom2d", "const")}
    with torch.no_grad():
        for x, ld, dist, m in lte:
            x, ld, m = x.to(dev), ld.to(dev), m.to(dev)
            if m.sum() == 0:
                continue
            base = lut[bins(dist).to(dev)]
            arms["model"].append(metrics(base + model(x), ld, m))
            arms["geom2d"].append(metrics(base, ld, m))
            arms["const"].append(metrics(torch.full_like(ld, gl), ld, m))
    return dict(held=held, n_train=len(tr), n_test=len(te),
                depth={k: agg(v) for k, v in arms.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--table", default="cache/inst_blockface.jsonl")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-train", type=int, default=1200)
    ap.add_argument("--max-test", type=int, default=300)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import glob
    keep = load_compact(os.path.join(a.root, a.table))
    files = sorted(f for f in (os.path.relpath(p, a.root) for p in
                   glob.glob(os.path.join(a.root, "blockface", "*", "faces", "*.npz")))
                   if keep.get(f))
    by = collections.Counter(dataset_of(f) for f in files)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{len(files):,} block-face images with compact instances, "
          f"{len(by)} source volumes, classes {'/'.join(COMPACT)}, device={dev}", flush=True)
    holds = a.holdout.split(",") if a.holdout else [d for d, n in by.most_common() if n >= 40]

    res = []
    for h in holds:
        r = run_fold(files, keep, h, a, dev)
        if not r:
            print(f"  {h:24s} skipped", flush=True); continue
        res.append(r)
        d, g, c = (r["depth"][k] for k in ("model", "geom2d", "const"))
        print(f"  {h:24s} test={r['n_test']:4d} | MAE nm  model {d['mae_nm']:7,.0f}  "
              f"geom2d {g['mae_nm']:7,.0f}  const {c['mae_nm']:7,.0f} | AbsRel model "
              f"{d['absrel']:.3f} geom2d {g['absrel']:.3f} | d1 model {d['d1']:.3f} "
              f"geom2d {g['d1']:.3f}", flush=True)

    if res:
        print(f"\nMACRO over {len(res)} held-out source volumes")
        print("  [depth_below_nm]")
        for arm in ("model", "geom2d", "const"):
            v = [r["depth"][arm] for r in res if r["depth"][arm]]
            mn = lambda k: np.mean([z[k] for z in v])
            print(f"    {arm:7s} MAE {mn('mae_nm'):8,.0f} nm  RMSE {mn('rmse_nm'):8,.0f} nm"
                  f"  AbsRel {mn('absrel'):.3f}  d1 {mn('d1'):.3f}  d2 {mn('d2'):.3f}"
                  f"  d3 {mn('d3'):.3f}")
        for rival in ("geom2d", "const"):
            w = sum(r["depth"]["model"]["mae_nm"] < r["depth"][rival]["mae_nm"] for r in res)
            print(f"    model beats {rival:7s} on {w}/{len(res)} folds  (MAE)")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
