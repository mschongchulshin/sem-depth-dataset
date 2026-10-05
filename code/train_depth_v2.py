import argparse, collections, json, os, time
import numpy as np

CACHE = ("/private/tmp/claude-501/-Users-hongchulshin/"
         "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache")
ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def metrics(pred, true):
    r = pred / true
    return dict(absrel=float(np.mean(np.abs(pred - true) / true)),
                medrel=float(np.median(np.abs(pred - true) / true)),
                rmse=float(np.sqrt(np.mean((pred - true) ** 2))),
                d1=float(np.mean(np.maximum(r, 1 / r) < 1.25)),
                d2=float(np.mean(np.maximum(r, 1 / r) < 1.25 ** 2)),
                n=int(true.size))


def build_edt(valid):
    from scipy import ndimage
    return ndimage.distance_transform_edt(valid).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=150)
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
    print(f"device {dev}   staged {N:,} planes at {CROP} px", flush=True)

    edt_path = f"{CACHE}/edt.npy"
    if os.path.exists(edt_path):
        EDT = np.load(edt_path, mmap_mode="r")[:N]
    else:
        print("computing distance transforms", flush=True)
        EDT = np.lib.format.open_memmap(edt_path, mode="w+", dtype=np.float32,
                                        shape=(N, CROP, CROP))
        for i in range(N):
            if i % 1000 == 0 and i:
                print(f"  {i:,}/{N:,}", flush=True)
            EDT[i] = build_edt(np.asarray(VM[i]))
        EDT.flush()

    counts = collections.Counter(ds.tolist())
    folds = [v for v, _ in counts.most_common(a.folds)]
    print("held-out folds: " + ", ".join(folds), flush=True)

    class Block(nn.Module):
        def __init__(s, i, o):
            super().__init__()
            s.f = nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o),
                                nn.SiLU(inplace=True),
                                nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o),
                                nn.SiLU(inplace=True))
        def forward(s, x): return s.f(x)

    class UNet(nn.Module):
        def __init__(s, cin=1, w=(32, 64, 128, 192)):
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

    results = {}
    for held in folds:
        t0 = time.time()
        te_i = np.flatnonzero(ds == held)
        tr_i = np.flatnonzero(ds != held)
        rng = np.random.default_rng(0); rng.shuffle(tr_i)
        va_i, tr_i = tr_i[:200], tr_i[200:]
        print(f"\n{'=' * 74}\nhold out {held}   train {len(tr_i):,}  val {len(va_i)}  "
              f"test {len(te_i):,}", flush=True)

        sub = tr_i[:1200]
        dtr = np.concatenate([np.asarray(DP[i])[np.asarray(VM[i])] for i in sub]).astype(np.float32)
        etr = np.concatenate([np.asarray(EDT[i])[np.asarray(VM[i])] for i in sub]).astype(np.float32)
        const = float(np.median(dtr))
        edges = np.unique(np.quantile(etr, np.linspace(0, 1, 41)))
        binned = np.zeros(len(edges) + 1, np.float32)
        bi = np.digitize(etr, edges)
        for b in range(len(edges) + 1):
            m = bi == b
            binned[b] = np.median(dtr[m]) if m.sum() > 50 else const
        print(f"  training median depth {const:.1f} nm over {dtr.size:,} pixels", flush=True)

        def run(cin, use_edt, tag):
            net = UNet(cin=cin).to(dev)
            opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=3)
            best, bad, state = 1e9, 0, None
            rg = np.random.default_rng(7)

            def batch(pool, n, aug):
                k = rg.choice(pool, size=n, replace=False)
                x = np.asarray(EM[k], np.float32) / 255.0
                y = np.asarray(DP[k], np.float32)
                m = np.asarray(VM[k])
                e = np.asarray(EDT[k], np.float32) / 40.0
                if aug:
                    if rg.random() < 0.5:
                        x, y, m, e = x[:, ::-1], y[:, ::-1], m[:, ::-1], e[:, ::-1]
                    if rg.random() < 0.5:
                        x, y, m, e = x[..., ::-1], y[..., ::-1], m[..., ::-1], e[..., ::-1]
                ch = [x] + ([e] if use_edt else [])
                X = torch.from_numpy(np.ascontiguousarray(np.stack(ch, 1))).to(dev)
                Y = torch.from_numpy(np.ascontiguousarray(np.log(np.maximum(y, 1.0)))[:, None]).to(dev)
                M = torch.from_numpy(np.ascontiguousarray(m)[:, None]).to(dev)
                return X, Y, M

            for ep in range(a.epochs):
                net.train(); tot = k2 = 0
                for _ in range(a.steps):
                    X, Y, M = batch(tr_i, a.batch, True)
                    p = net(X)
                    loss = (torch.abs(p - Y) * M).sum() / M.sum().clamp(min=1)
                    if not torch.isfinite(loss):
                        continue
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
                    opt.step(); tot += float(loss); k2 += 1
                net.eval(); vt = vk = 0
                with torch.no_grad():
                    for _ in range(12):
                        X, Y, M = batch(va_i, a.batch, False)
                        p = net(X)
                        vt += float((torch.abs(p - Y) * M).sum() / M.sum().clamp(min=1)); vk += 1
                v = vt / max(vk, 1); sched.step(v)
                flag = ""
                if v < best - 1e-4:
                    best, bad = v, 0
                    state = {kk: t.detach().cpu().clone() for kk, t in net.state_dict().items()}
                    flag = " *"
                else:
                    bad += 1
                if ep % 4 == 0 or flag:
                    print(f"    {tag} ep {ep:2d}  train {tot/max(k2,1):.4f}  "
                          f"val {v:.4f}{flag}", flush=True)
                if bad >= a.patience:
                    print(f"    {tag} early stop at {ep}", flush=True); break
            if state:
                net.load_state_dict(state)
            net.eval()
            P = []
            with torch.no_grad():
                for i in te_i:
                    x = np.asarray(EM[i], np.float32)[None, None] / 255.0
                    ch = [x]
                    if use_edt:
                        ch.append(np.asarray(EDT[i], np.float32)[None, None] / 40.0)
                    X = torch.from_numpy(np.concatenate(ch, 1)).to(dev)
                    P.append(np.exp(net(X)[0, 0].cpu().numpy())[np.asarray(VM[i])])
            return np.concatenate(P)

        T = np.concatenate([np.asarray(DP[i], np.float32)[np.asarray(VM[i])] for i in te_i])
        E = np.concatenate([np.asarray(EDT[i], np.float32)[np.asarray(VM[i])] for i in te_i])
        arms = {"constant": np.full_like(T, const),
                "edt": binned[np.digitize(E, edges)]}
        arms["unet"] = run(1, False, "unet")
        arms["unet+edt"] = run(2, True, "unet+edt")

        res = {k: metrics(v, T) for k, v in arms.items()}
        results[held] = res
        print(f"\n  {'arm':<10s} {'AbsRel':>8s} {'MedRel':>8s} {'RMSE nm':>9s} "
              f"{'delta1':>8s} {'delta2':>8s}")
        for k, m in res.items():
            print(f"  {k:<10s} {m['absrel']:>8.4f} {m['medrel']:>8.4f} {m['rmse']:>9.1f} "
                  f"{m['d1']:>8.4f} {m['d2']:>8.4f}")
        print(f"  scored pixels {res['unet']['n']:,}   {time.time()-t0:.0f}s", flush=True)
        json.dump(results, open(f"{ROOT}/cache/depth_baseline.json", "w"), indent=1)

    print(f"\n{'=' * 74}\nmean over {len(results)} held-out source volumes")
    print(f"  {'arm':<10s} {'AbsRel':>8s} {'MedRel':>8s} {'delta1':>8s} {'delta2':>8s}")
    for k in ("constant", "edt", "unet", "unet+edt"):
        if not all(k in r for r in results.values()):
            continue
        print(f"  {k:<10s} "
              f"{np.mean([r[k]['absrel'] for r in results.values()]):>8.4f} "
              f"{np.mean([r[k]['medrel'] for r in results.values()]):>8.4f} "
              f"{np.mean([r[k]['d1'] for r in results.values()]):>8.4f} "
              f"{np.mean([r[k]['d2'] for r in results.values()]):>8.4f}")
    print("-> cache/depth_baseline.json")


if __name__ == "__main__":
    main()
