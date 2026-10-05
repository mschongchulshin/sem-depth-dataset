import argparse, collections, glob, json, os, time
import numpy as np

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_index(limit_per_vol=900, min_fg=0.04, crop=256):
    cache = f"{ROOT}/cache/depth_index.json"
    if os.path.exists(cache):
        return json.load(open(cache))
    files = sorted(f for f in glob.glob(f"{ROOT}/blockface/*/faces/*.npz") if "/._" not in f)
    print(f"scanning {len(files):,} planes", flush=True)
    per_vol = collections.Counter()
    keep = []
    for i, f in enumerate(files):
        if i % 3000 == 0 and i:
            print(f"  {i:,}  kept {len(keep):,}", flush=True)
        box = os.path.basename(os.path.dirname(os.path.dirname(f)))
        vol = box.split("__")[0]
        if per_vol[vol] >= limit_per_vol:
            continue
        try:
            z = np.load(f, allow_pickle=False)
            msk = z["mask"]
            if msk.shape[0] < crop or msk.shape[1] < crop:
                continue
            if float(msk.mean()) < min_fg:
                continue
            m = json.loads(str(z["meta"]))
        except Exception:
            continue
        per_vol[vol] += 1
        keep.append(dict(file=os.path.relpath(f, ROOT), box=box, dataset=vol,
                         px_nm=float(m.get("pixel_size_nm", 0) or 0),
                         z_step_nm=float(m.get("z_step_nm", 0) or 0)))
    json.dump(keep, open(cache, "w"))
    print(f"kept {len(keep):,} planes over {len(per_vol)} source volumes")
    return keep


def load_plane(rec, crop=256, rng=None):
    z = np.load(f"{ROOT}/{rec['file']}", allow_pickle=False)
    em, dep, msk = z["em"], z["depth_below_nm"], z["mask"]
    clipped = z["clipped"] if "clipped" in z.files else np.zeros_like(msk)
    H, W = msk.shape
    if rng is None:
        y0, x0 = (H - crop) // 2, (W - crop) // 2
    else:
        y0 = int(rng.integers(0, H - crop + 1)); x0 = int(rng.integers(0, W - crop + 1))
    sl = (slice(y0, y0 + crop), slice(x0, x0 + crop))
    d = np.nan_to_num(dep[sl].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = msk[sl] & np.isfinite(dep[sl]) & (d > 0) & ~clipped[sl]
    return em[sl].astype(np.float32) / 255.0, d, valid, rec["px_nm"]


def metrics(pred, true):
    r = pred / true
    return dict(absrel=float(np.mean(np.abs(pred - true) / true)),
                medrel=float(np.median(np.abs(pred - true) / true)),
                rmse=float(np.sqrt(np.mean((pred - true) ** 2))),
                d1=float(np.mean(np.maximum(r, 1 / r) < 1.25)),
                d2=float(np.mean(np.maximum(r, 1 / r) < 1.25 ** 2)),
                n=int(true.size))


def make_unet(torch, nn):
    class Block(nn.Module):
        def __init__(s, a, b):
            super().__init__()
            s.f = nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b),
                                nn.ReLU(inplace=True),
                                nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b),
                                nn.ReLU(inplace=True))
        def forward(s, x): return s.f(x)

    class UNet(nn.Module):
        def __init__(s, w=(24, 48, 96, 160)):
            super().__init__()
            s.e1, s.e2, s.e3 = Block(1, w[0]), Block(w[0], w[1]), Block(w[1], w[2])
            s.b = Block(w[2], w[3])
            s.u3 = nn.Conv2d(w[3], w[2], 1); s.d3 = Block(w[2] * 2, w[2])
            s.u2 = nn.Conv2d(w[2], w[1], 1); s.d2 = Block(w[1] * 2, w[1])
            s.u1 = nn.Conv2d(w[1], w[0], 1); s.d1 = Block(w[0] * 2, w[0])
            s.out = nn.Conv2d(w[0], 1, 1)
            s.pool = nn.MaxPool2d(2)
        def forward(s, x):
            e1 = s.e1(x); e2 = s.e2(s.pool(e1)); e3 = s.e3(s.pool(e2))
            b = s.b(s.pool(e3))
            up = nn.functional.interpolate
            d3 = s.d3(torch.cat([up(s.u3(b), scale_factor=2, mode="nearest"), e3], 1))
            d2 = s.d2(torch.cat([up(s.u2(d3), scale_factor=2, mode="nearest"), e2], 1))
            d1 = s.d1(torch.cat([up(s.u1(d2), scale_factor=2, mode="nearest"), e1], 1))
            return s.out(d1)
    return UNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5, help="how many source volumes to hold out")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--steps", type=int, default=220, help="batches per epoch")
    ap.add_argument("--val-planes", type=int, default=120)
    ap.add_argument("--test-planes", type=int, default=300)
    a = ap.parse_args()

    import torch
    from torch import nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device {dev}", flush=True)

    idx = build_index(crop=a.crop)
    by_vol = collections.defaultdict(list)
    for r in idx:
        by_vol[r["dataset"]].append(r)
    vols = sorted(by_vol, key=lambda v: -len(by_vol[v]))
    print("planes per source volume: " +
          ", ".join(f"{v.replace('jrc_','').replace('aic_','')} {len(by_vol[v])}"
                    for v in vols[:20]), flush=True)
    folds = vols[:a.folds]

    UNet = make_unet(torch, nn)
    results = {}

    for held in folds:
        t0 = time.time()
        tr_vols = [v for v in vols if v != held]
        tr = [r for v in tr_vols for r in by_vol[v]]
        te = by_vol[held]
        rng = np.random.default_rng(0)
        rng.shuffle(tr)
        val, tr = tr[:a.val_planes], tr[a.val_planes:]
        te_s = te[:a.test_planes]
        print(f"\n{'='*74}\nhold out {held}   train {len(tr):,}  val {len(val)}  "
              f"test {len(te_s)}", flush=True)

        samp = [load_plane(r, a.crop) for r in tr[:400]]
        allд = np.concatenate([d[v] for _, d, v, _ in samp if v.any()])
        const = float(np.median(allд))
        print(f"  training median depth {const:.1f} nm  over {allд.size:,} pixels", flush=True)

        net = UNet().to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=2)
        best, bad, best_state = 1e9, 0, None
        rng_t = np.random.default_rng(1)

        def batch(pool, n, train=True):
            xs, ys, ms = [], [], []
            for _ in range(n):
                r = pool[int(rng_t.integers(len(pool)))]
                em, dep, val_m, _ = load_plane(r, a.crop, rng_t if train else None)
                if not val_m.any():
                    continue
                xs.append(em); ys.append(np.log(np.maximum(dep, 1.0)).astype(np.float32)); ms.append(val_m)
            if not xs:
                return None
            X = torch.from_numpy(np.stack(xs)[:, None]).to(dev)
            Y = torch.from_numpy(np.stack(ys)[:, None]).to(dev)
            M = torch.from_numpy(np.stack(ms)[:, None]).to(dev)
            return X, Y, M

        for ep in range(a.epochs):
            net.train(); tot = k = 0
            for _ in range(a.steps):
                b = batch(tr, a.batch, True)
                if b is None:
                    continue
                X, Y, M = b
                p = net(X)
                loss = (torch.abs(p - Y) * M).sum() / M.sum().clamp(min=1)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                opt.step()
                tot += float(loss); k += 1
            net.eval(); vt = vk = 0
            with torch.no_grad():
                for _ in range(20):
                    b = batch(val, a.batch, False)
                    if b is None:
                        continue
                    X, Y, M = b
                    p = net(X)
                    vt += float((torch.abs(p - Y) * M).sum() / M.sum().clamp(min=1)); vk += 1
            v = vt / max(vk, 1)
            sched.step(v)
            flag = ""
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k2: t.detach().cpu().clone() for k2, t in net.state_dict().items()}
                flag = " *"
            else:
                bad += 1
            print(f"    epoch {ep:2d}  train {tot/max(k,1):.4f}  val {v:.4f}{flag}", flush=True)
            if bad >= a.patience:
                print(f"    early stop at epoch {ep}", flush=True)
                break
        if best_state:
            net.load_state_dict(best_state)

        net.eval()
        P, T, AREA = [], [], []
        with torch.no_grad():
            for r in te_s:
                em, dep, val_m, px = load_plane(r, a.crop)
                if not val_m.any():
                    continue
                X = torch.from_numpy(em[None, None]).to(dev)
                p = np.exp(net(X)[0, 0].cpu().numpy())
                P.append(p[val_m]); T.append(dep[val_m])
                AREA.append(np.full(int(val_m.sum()), float(val_m.mean())))
        P = np.concatenate(P); T = np.concatenate(T); AREA = np.concatenate(AREA)
        arms = {"constant": np.full_like(T, const), "unet": P}
        res = {k: metrics(v, T) for k, v in arms.items()}
        results[held] = res
        print(f"  {'arm':<10s} {'AbsRel':>8s} {'MedRel':>8s} {'RMSE nm':>9s} "
              f"{'delta1':>8s} {'delta2':>8s}")
        for k, m in res.items():
            print(f"  {k:<10s} {m['absrel']:>8.4f} {m['medrel']:>8.4f} {m['rmse']:>9.1f} "
                  f"{m['d1']:>8.4f} {m['d2']:>8.4f}")
        print(f"  pixels {res['unet']['n']:,}   {time.time()-t0:.0f}s", flush=True)
        json.dump(results, open(f"{ROOT}/cache/depth_baseline.json", "w"), indent=1)

    print(f"\n{'='*74}\nmean over {len(results)} held-out source volumes")
    print(f"  {'arm':<10s} {'AbsRel':>8s} {'MedRel':>8s} {'delta1':>8s} {'delta2':>8s}")
    for k in ("constant", "unet"):
        print(f"  {k:<10s} "
              f"{np.mean([r[k]['absrel'] for r in results.values()]):>8.4f} "
              f"{np.mean([r[k]['medrel'] for r in results.values()]):>8.4f} "
              f"{np.mean([r[k]['d1'] for r in results.values()]):>8.4f} "
              f"{np.mean([r[k]['d2'] for r in results.values()]):>8.4f}")
    print(f"-> cache/depth_baseline.json")


if __name__ == "__main__":
    main()
