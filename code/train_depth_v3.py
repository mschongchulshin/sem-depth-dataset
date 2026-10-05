import argparse, collections, json, os, time
import numpy as np

CACHE = ("/private/tmp/claude-501/-Users-hongchulshin/"
         "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache")
ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG = {
    "A": dict(norm="div255", scale_ch=False, loss="logl1", target="nm",    edt_ch=True),
    "B": dict(norm="zscore", scale_ch=False, loss="logl1", target="nm",    edt_ch=True),
    "C": dict(norm="zscore", scale_ch=True,  loss="logl1", target="nm",    edt_ch=True),
    "D": dict(norm="zscore", scale_ch=True,  loss="silog", target="nm",    edt_ch=True),
    "E": dict(norm="zscore", scale_ch=True,  loss="silog", target="steps", edt_ch=True),
    "F": dict(norm="zscore", scale_ch=True,  loss="silog", target="steps", edt_ch=False),

    "M": dict(norm="zscore", scale_ch=True,  loss="silog", target="steps", edt_ch=False,
              blank_img=True),
    "S": dict(norm="zscore", scale_ch=True,  loss="silog", target="steps", edt_ch=True,
              blank_img=True),
    "P": dict(norm="zscore", scale_ch=True,  loss="silog", target="steps", edt_ch=False),
    "B": dict(norm="zscore", scale_ch=True,  loss="silog", target="steps", edt_ch=True),
}


def metrics(pred_nm, true_nm):
    r = pred_nm / true_nm
    lg = np.log(np.maximum(pred_nm, 1)) - np.log(np.maximum(true_nm, 1))
    return dict(absrel=float(np.mean(np.abs(pred_nm - true_nm) / true_nm)),
                medrel=float(np.median(np.abs(pred_nm - true_nm) / true_nm)),
                rmse=float(np.sqrt(np.mean((pred_nm - true_nm) ** 2))),
                d1=float(np.mean(np.maximum(r, 1 / r) < 1.25)),
                d2=float(np.mean(np.maximum(r, 1 / r) < 1.25 ** 2)),
                silog=float(np.sqrt(np.mean(lg ** 2) - np.mean(lg) ** 2)),
                n=int(true_nm.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A,B,C,D,E")
    ap.add_argument("--folds", type=int, default=4, help="0 means every source volume")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--out", default="cache/depth_v3.json")
    a = ap.parse_args()

    import torch
    from torch import nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    meta = json.load(open(f"{CACHE}/meta.json"))
    N, CROP = meta["n"], meta["crop"]
    EM = np.load(f"{CACHE}/em.npy", mmap_mode="r")[:N]
    DP = np.load(f"{CACHE}/depth.npy", mmap_mode="r")[:N]
    VM = np.load(f"{CACHE}/valid.npy", mmap_mode="r")[:N]
    ds = np.array([m["dataset"] for m in meta["meta"]])
    PX = np.array([float(m["px_nm"]) for m in meta["meta"]], np.float32)
    ZS = np.array([float(m.get("z_step_nm", 0) or 0) for m in meta["meta"]], np.float32)
    ZS[ZS <= 0] = PX[ZS <= 0]
    print(f"device {dev}   staged {N:,} planes at {CROP} px", flush=True)
    print(f"pixel sizes {sorted(set(PX.tolist()))}", flush=True)

    edt_path = f"{CACHE}/edt.npy"
    if os.path.exists(edt_path):
        EDT = np.load(edt_path, mmap_mode="r")[:N]
    else:
        from scipy import ndimage
        print("computing distance transforms", flush=True)
        EDT = np.lib.format.open_memmap(edt_path, mode="w+", dtype=np.float32,
                                        shape=(N, CROP, CROP))
        for i in range(N):
            if i % 1000 == 0 and i:
                print(f"  {i:,}/{N:,}", flush=True)
            EDT[i] = ndimage.distance_transform_edt(np.asarray(VM[i]))
        EDT.flush()

    counts = collections.Counter(ds.tolist())
    folds = ([v for v, _ in counts.most_common()] if a.folds == 0
             else [v for v, _ in counts.most_common(a.folds)])
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    print(f"arms {arms}   folds {len(folds)}", flush=True)

    class Block(nn.Module):
        def __init__(s, i, o):
            super().__init__()
            s.f = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o),
                                nn.SiLU(inplace=True),
                                nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o),
                                nn.SiLU(inplace=True))
        def forward(s, x): return s.f(x)

    class UNet(nn.Module):
        def __init__(s, cin, w=(32, 64, 128, 192)):
            super().__init__()
            s.e1, s.e2, s.e3 = Block(cin, w[0]), Block(w[0], w[1]), Block(w[1], w[2])
            s.b = Block(w[2], w[3])
            s.u3 = nn.Conv2d(w[3], w[2], 1); s.d3 = Block(w[2] * 2, w[2])
            s.u2 = nn.Conv2d(w[2], w[1], 1); s.d2 = Block(w[1] * 2, w[1])
            s.u1 = nn.Conv2d(w[1], w[0], 1); s.d1 = Block(w[0] * 2, w[0])
            s.out = nn.Conv2d(w[0], 1, 1); s.pool = nn.MaxPool2d(2)
        def forward(s, x):
            up = nn.functional.interpolate
            e1 = s.e1(x); e2 = s.e2(s.pool(e1)); e3 = s.e3(s.pool(e2))
            b = s.b(s.pool(e3))
            d3 = s.d3(torch.cat([up(s.u3(b), scale_factor=2, mode="nearest"), e3], 1))
            d2 = s.d2(torch.cat([up(s.u2(d3), scale_factor=2, mode="nearest"), e2], 1))
            d1 = s.d1(torch.cat([up(s.u1(d2), scale_factor=2, mode="nearest"), e1], 1))
            return s.out(d1)

    results = collections.defaultdict(dict)

    def make_batch(idx, cfg, rg, aug):
        x = np.asarray(EM[idx], np.float32)
        m = np.asarray(VM[idx])
        y_nm = np.asarray(DP[idx], np.float32)
        e = np.asarray(EDT[idx], np.float32)
        zs = ZS[idx][:, None, None]
        px = PX[idx][:, None, None]
        if cfg["norm"] == "div255":
            x = x / 255.0
        else:
            mu = x.mean(axis=(1, 2), keepdims=True)
            sd = x.std(axis=(1, 2), keepdims=True)
            x = (x - mu) / np.maximum(sd, 1e-3)
        if cfg.get("blank_img"):
            x = np.zeros_like(x)
        ch = [x]
        if cfg.get("edt_ch", True):
            ch.append(e / 40.0)
        if cfg["scale_ch"]:
            ch.append(np.broadcast_to(np.log(px / 16.0), x.shape).astype(np.float32))
            ch.append(np.broadcast_to(np.log(zs / 16.0), x.shape).astype(np.float32))
        X = np.stack(ch, 1)
        y = y_nm / zs if cfg["target"] == "steps" else y_nm
        Y = np.log(np.maximum(y, 1e-3))[:, None]
        if aug:
            if rg.random() < 0.5:
                X, Y, m = X[..., ::-1, :], Y[..., ::-1, :], m[:, ::-1]
            if rg.random() < 0.5:
                X, Y, m = X[..., ::-1], Y[..., ::-1], m[..., ::-1]
        return (np.ascontiguousarray(X), np.ascontiguousarray(Y),
                np.ascontiguousarray(m)[:, None], zs)

    for held in folds:
        te_i = np.flatnonzero(ds == held)
        tr_i = np.flatnonzero(ds != held)
        rng = np.random.default_rng(0); rng.shuffle(tr_i)
        va_i, tr_i = tr_i[:180], tr_i[180:]
        T = np.concatenate([np.asarray(DP[i], np.float32)[np.asarray(VM[i])] for i in te_i])
        print(f"\n{'=' * 78}\nhold out {held}   train {len(tr_i):,}  test {len(te_i):,}  "
              f"scored pixels {T.size:,}", flush=True)

        sub = tr_i[:900]
        dtr = np.concatenate([np.asarray(DP[i])[np.asarray(VM[i])] for i in sub]).astype(np.float32)
        etr = np.concatenate([np.asarray(EDT[i])[np.asarray(VM[i])] for i in sub]).astype(np.float32)
        const = float(np.median(dtr))
        edges = np.unique(np.quantile(etr, np.linspace(0, 1, 41)))
        binned = np.zeros(len(edges) + 1, np.float32)
        bi = np.digitize(etr, edges)
        for b in range(len(edges) + 1):
            mm = bi == b
            binned[b] = np.median(dtr[mm]) if mm.sum() > 50 else const
        E_te = np.concatenate([np.asarray(EDT[i], np.float32)[np.asarray(VM[i])] for i in te_i])
        results[held]["constant"] = metrics(np.full_like(T, const), T)
        results[held]["edt"] = metrics(binned[np.digitize(E_te, edges)], T)

        for arm in arms:
            cfg = CONFIG[arm]
            t0 = time.time()
            cin = 1 + (1 if cfg.get("edt_ch", True) else 0) + (2 if cfg["scale_ch"] else 0)
            net = UNet(cin).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=3)
            best, bad, state = 1e9, 0, None
            rg = np.random.default_rng(7)

            def loss_of(p, Y, M):
                d = (p - Y) * M
                if cfg["loss"] == "logl1":
                    return d.abs().sum() / M.sum().clamp(min=1)
                n = M.sum().clamp(min=1)
                m1 = d.sum() / n
                m2 = (d * d).sum() / n
                return torch.sqrt((m2 - 0.5 * m1 * m1).clamp(min=1e-8))

            for ep in range(a.epochs):
                net.train()
                for _ in range(a.steps):
                    idx = rg.choice(tr_i, size=a.batch, replace=False)
                    X, Y, M, _ = make_batch(idx, cfg, rg, True)
                    X = torch.from_numpy(X).to(dev); Y = torch.from_numpy(Y).to(dev)
                    M = torch.from_numpy(M).to(dev)
                    l = loss_of(net(X), Y, M)
                    if not torch.isfinite(l):
                        continue
                    opt.zero_grad(); l.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
                net.eval(); vt = vk = 0
                with torch.no_grad():
                    for _ in range(10):
                        idx = rg.choice(va_i, size=a.batch, replace=False)
                        X, Y, M, _ = make_batch(idx, cfg, rg, False)
                        X = torch.from_numpy(X).to(dev); Y = torch.from_numpy(Y).to(dev)
                        M = torch.from_numpy(M).to(dev)
                        vt += float(loss_of(net(X), Y, M)); vk += 1
                v = vt / max(vk, 1); sched.step(v)
                if v < best - 1e-4:
                    best, bad = v, 0
                    state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
                else:
                    bad += 1
                if bad >= a.patience:
                    break
            if state:
                net.load_state_dict(state)
            net.eval()
            P = []
            with torch.no_grad():
                for i in te_i:
                    X, _, _, zs = make_batch(np.array([i]), cfg, rg, False)
                    p = np.exp(net(torch.from_numpy(X).to(dev))[0, 0].cpu().numpy())
                    if cfg["target"] == "steps":
                        p = p * float(zs[0, 0, 0])
                    P.append(p[np.asarray(VM[i])])
            m = metrics(np.concatenate(P), T)
            results[held][arm] = m
            print(f"  {arm}  AbsRel {m['absrel']:>8.3f}  MedRel {m['medrel']:>7.3f}  "
                  f"d1 {m['d1']:.3f}  SIlog {m['silog']:.3f}  val {best:.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)
            json.dump(results, open(f"{ROOT}/{a.out}", "w"), indent=1)

    print(f"\n{'=' * 78}\nmean over {len(results)} held-out source volumes")
    print(f"  {'arm':<10s}{'AbsRel':>9s}{'MedRel':>9s}{'d1':>8s}{'d2':>8s}{'SIlog':>8s}")
    for k in ["constant", "edt"] + arms:
        vals = [r[k] for r in results.values() if k in r]
        if not vals:
            continue
        print(f"  {k:<10s}"
              f"{np.mean([v['absrel'] for v in vals]):>9.3f}"
              f"{np.mean([v['medrel'] for v in vals]):>9.3f}"
              f"{np.mean([v['d1'] for v in vals]):>8.3f}"
              f"{np.mean([v['d2'] for v in vals]):>8.3f}"
              f"{np.mean([v['silog'] for v in vals]):>8.3f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
