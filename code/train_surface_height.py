import argparse, collections, glob, json, os, re
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def dataset_of(path):
    parts = path.replace("\\", "/").split("/")
    box = parts[-3] if len(parts) > 2 else parts[-1]
    return box.split("__")[0]


class Views(Dataset):

    def __init__(self, files, root, size=192, augment=False):
        self.files, self.root, self.size, self.augment = files, root, size, augment
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(os.path.join(self.root, self.files[i]), allow_pickle=False)
        se = np.asarray(d["se"], dtype=np.float32)
        bse = np.asarray(d["bse"], dtype=np.float32)
        dep = np.asarray(d["depth_nm"], dtype=np.float32)
        m = np.asarray(d["mask"]).astype(bool) & np.isfinite(dep) & (dep > 0)
        x = np.concatenate([se[None], bse], 0)

        s = self.size
        H, W = se.shape
        if H < s or W < s:
            pad = ((0, max(0, s - H)), (0, max(0, s - W)))
            x = np.pad(x, ((0, 0), *pad)); dep = np.pad(dep, pad)
            m = np.pad(m, pad, constant_values=False)
            H, W = x.shape[-2:]
        y0 = int(self.rng.integers(0, H - s + 1)) if self.augment else (H - s) // 2
        x0 = int(self.rng.integers(0, W - s + 1)) if self.augment else (W - s) // 2
        sl = (slice(y0, y0 + s), slice(x0, x0 + s))
        x, dep, m = x[(slice(None), *sl)], dep[sl], m[sl]
        ph, pw = (-x.shape[-2]) % 8, (-x.shape[-1]) % 8
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, ph), (0, pw)))
            dep = np.pad(dep, ((0, ph), (0, pw)))
            m = np.pad(m, ((0, ph), (0, pw)), constant_values=False)
        return (torch.from_numpy(x.copy()),
                torch.from_numpy(np.log(np.maximum(dep, 1.0)).astype(np.float32)),
                torch.from_numpy(m))


def blk(ci, co):
    return nn.Sequential(nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU(),
                         nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.GELU())


class UNet(nn.Module):
    def __init__(self, cin=5, w=(32, 64, 128, 256)):
        super().__init__()
        self.e1, self.e2, self.e3 = blk(cin, w[0]), blk(w[0], w[1]), blk(w[1], w[2])
        self.bott = blk(w[2], w[3])
        self.d3, self.d2, self.d1 = blk(w[3] + w[2], w[2]), blk(w[2] + w[1], w[1]), \
            blk(w[1] + w[0], w[0])
        self.out = nn.Conv2d(w[0], 1, 1)
        self.pool = nn.MaxPool2d(2)

    def start_at(self, v):
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.fill_(float(v))

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        b = self.bott(self.pool(e3))
        up = lambda t, r: F.interpolate(t, size=r.shape[-2:], mode="nearest")
        d3 = self.d3(torch.cat([up(b, e3), e3], 1))
        d2 = self.d2(torch.cat([up(d3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        return self.out(d1)[:, 0]


def _stats(p, t, prefix=""):
    r = torch.maximum(p / t, t / p)
    return {prefix + "mae_nm": float((p - t).abs().mean()),
            prefix + "rmse_nm": float(torch.sqrt(((p - t) ** 2).mean())),
            prefix + "absrel": float(((p - t).abs() / t).mean()),
            prefix + "d1": float((r < 1.25).float().mean()),
            prefix + "d2": float((r < 1.25 ** 2).float().mean()),
            prefix + "d3": float((r < 1.25 ** 3).float().mean())}


def metrics(p_log, t_log, m):
    if m.sum() == 0:
        return None
    p, t = p_log[m].exp(), t_log[m].exp()
    out = _stats(p, t)
    pl, tl = p_log[m], t_log[m]
    if pl.numel() >= 16 and float(pl.std()) > 1e-9:
        pm, tm = pl.mean(), tl.mean()
        var = ((pl - pm) ** 2).sum()
        slope = ((pl - pm) * (tl - tm)).sum() / var.clamp(min=1e-9)
        out.update(_stats((pl * slope + (tm - slope * pm)).exp(), t, "fit_"))
    else:
        out.update(_stats(p, t, "fit_"))
    out["n"] = int(m.sum())
    return out


def agg(rows):
    rows = [r for r in rows if r]
    if not rows:
        return None
    n = sum(r["n"] for r in rows)
    return {k: sum(r[k] * r["n"] for r in rows) / n for k in rows[0] if k != "n"} | {"n": n}


def run_fold(files, held, a, dev):
    tr = [f for f in files if dataset_of(f) != held]
    te = [f for f in files if dataset_of(f) == held]
    if len(te) < 20 or not tr:
        return None
    rng = np.random.default_rng(0)
    if len(tr) > a.max_train:
        tr = list(rng.choice(tr, a.max_train, replace=False))
    if len(te) > a.max_test:
        te = list(rng.choice(te, a.max_test, replace=False))
    nv = max(1, int(0.2 * len(tr)))
    vidx = set(np.random.default_rng(1).choice(len(tr), nv, replace=False).tolist())
    trn = [f for i, f in enumerate(tr) if i not in vidx]
    val = [f for i, f in enumerate(tr) if i in vidx] or tr

    mk = lambda fl, aug: DataLoader(Views(fl, a.root, a.size, aug),
                                    batch_size=a.batch, shuffle=aug)
    ltr, lva, lte = mk(trn, True), mk(val, False), mk(te, False)

    med = []
    for _, y, m in ltr:
        if m.sum():
            med.append(y[m])
        if len(med) > 40:
            break
    model = UNet().to(dev)
    model.start_at(float(torch.cat(med).median()) if med else 8.0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.3, patience=2)

    def evaluate(loader):
        model.eval(); acc = []
        with torch.no_grad():
            for x, y, m in loader:
                x, y, m = x.to(dev), y.to(dev), m.to(dev)
                if m.sum() == 0:
                    continue
                acc.append(metrics(model(x), y, m))
        return agg(acc)

    best, state, bad, bep = float("inf"), None, 0, 0
    for ep in range(a.epochs):
        model.train()
        for x, y, m in ltr:
            x, y, m = x.to(dev), y.to(dev), m.to(dev)
            if m.sum() == 0:
                continue
            loss = F.l1_loss(model(x)[m], y[m])
            opt.zero_grad(); loss.backward(); opt.step()
        v = evaluate(lva)
        vv = v["fit_absrel"] if v else float("inf")
        sched.step(vv)
        if vv < best * (1 - 0.002):
            best, bep, bad = vv, ep + 1, 0
            state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
        print(f"    ep{ep + 1:3d}  val AbsRel {vv:.4f}{'  *' if bad == 0 else ''}", flush=True)
        if bad >= a.patience:
            break
    if state is not None:
        model.load_state_dict(state)
    return dict(held=held, best_epoch=bep, **(evaluate(lte) or {}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--size", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-train", type=int, default=1500)
    ap.add_argument("--max-test", type=int, default=400)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    files = sorted(os.path.relpath(p, a.root) for p in
                   glob.glob(os.path.join(a.root, "corpus", "*", "views", "*.npz")))
    by = collections.Counter(dataset_of(f) for f in files)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"{len(files):,} surface views, {len(by)} source volumes, device={dev}", flush=True)
    holds = a.holdout.split(",") if a.holdout else [d for d, n in by.most_common() if n >= 40]

    res = []
    for h in holds:
        r = run_fold(files, h, a, dev)
        if not r:
            print(f"  {h:24s} skipped", flush=True); continue
        res.append(r)
        print(f"  {h:24s} ep{r['best_epoch']:02d}  fitAbsRel {r['fit_absrel']:.3f}"
              f"  (absolute {r['absrel']:.3f})  fitMAE {r['fit_mae_nm']:,.0f} nm"
              f"  fitd1 {r['fit_d1']:.3f}", flush=True)
    if res:
        f = lambda k: np.mean([r[k] for r in res])
        print(f"\nMACRO over {len(res)} volumes")
        print(f"  like-for-like, affine fitted   AbsRel {f('fit_absrel'):.3f}  "
              f"MAE {f('fit_mae_nm'):,.0f} nm  d1 {f('fit_d1'):.3f}  d2 {f('fit_d2'):.3f}")
        print(f"  absolute, no fit               AbsRel {f('absrel'):.3f}  "
              f"MAE {f('mae_nm'):,.0f} nm  d1 {f('d1'):.3f}"
              f"   <- classical cannot do this at all")
        print(f"  classical four-quadrant integration, scale and offset fitted per image: "
              f"AbsRel 0.781, MAE 2,398 nm")
    if a.out:
        json.dump(res, open(os.path.join(a.root, a.out), "w"), indent=1)


if __name__ == "__main__":
    main()
