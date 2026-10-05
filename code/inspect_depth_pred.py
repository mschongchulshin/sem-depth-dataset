import json, os, time
import numpy as np

CACHE = ("/private/tmp/claude-501/-Users-hongchulshin/"
         "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache")
ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HELD = "jrc_choroid-plexus-2"
OUT = f"{CACHE}/pred_{HELD}.npz"


def main():
    import torch
    from torch import nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    meta = json.load(open(f"{CACHE}/meta.json"))
    N, CROP = meta["n"], meta["crop"]
    EM = np.load(f"{CACHE}/em.npy", mmap_mode="r")[:N]
    DP = np.load(f"{CACHE}/depth.npy", mmap_mode="r")[:N]
    VM = np.load(f"{CACHE}/valid.npy", mmap_mode="r")[:N]
    EDT = np.load(f"{CACHE}/edt.npy", mmap_mode="r")[:N]
    ds = np.array([m["dataset"] for m in meta["meta"]])

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

    te_i = np.flatnonzero(ds == HELD)
    tr_i = np.flatnonzero(ds != HELD)
    rng = np.random.default_rng(0); rng.shuffle(tr_i)
    va_i, tr_i = tr_i[:200], tr_i[200:]
    print(f"hold out {HELD}  train {len(tr_i):,}  test {len(te_i):,}", flush=True)

    sub = tr_i[:1200]
    dtr = np.concatenate([np.asarray(DP[i])[np.asarray(VM[i])] for i in sub]).astype(np.float32)
    const = float(np.median(dtr))
    print(f"training median depth {const:.1f} nm", flush=True)

    def train_arm(cin, use_edt, tag, epochs=60, patience=8, steps=150, bs=16):
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
                if rg.random() < 0.5: x, y, m, e = x[:, ::-1], y[:, ::-1], m[:, ::-1], e[:, ::-1]
                if rg.random() < 0.5: x, y, m, e = x[..., ::-1], y[..., ::-1], m[..., ::-1], e[..., ::-1]
            ch = [x] + ([e] if use_edt else [])
            X = torch.from_numpy(np.ascontiguousarray(np.stack(ch, 1))).to(dev)
            Y = torch.from_numpy(np.ascontiguousarray(np.log(np.maximum(y, 1.0)))[:, None]).to(dev)
            M = torch.from_numpy(np.ascontiguousarray(m)[:, None]).to(dev)
            return X, Y, M

        for ep in range(epochs):
            net.train()
            for _ in range(steps):
                X, Y, M = batch(tr_i, bs, True)
                loss = (torch.abs(net(X) - Y) * M).sum() / M.sum().clamp(min=1)
                if not torch.isfinite(loss): continue
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            net.eval(); vt = vk = 0
            with torch.no_grad():
                for _ in range(12):
                    X, Y, M = batch(va_i, bs, False)
                    vt += float((torch.abs(net(X) - Y) * M).sum() / M.sum().clamp(min=1)); vk += 1
            v = vt / max(vk, 1); sched.step(v)
            if v < best - 1e-4:
                best, bad = v, 0
                state = {k2: t.detach().cpu().clone() for k2, t in net.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                print(f"  {tag} early stop at {ep}, best val {best:.4f}", flush=True); break
        if state: net.load_state_dict(state)
        net.eval()
        P = []
        with torch.no_grad():
            for i in te_i:
                x = np.asarray(EM[i], np.float32)[None, None] / 255.0
                ch = [x]
                if use_edt: ch.append(np.asarray(EDT[i], np.float32)[None, None] / 40.0)
                P.append(np.exp(net(torch.from_numpy(np.concatenate(ch, 1)).to(dev))[0, 0]
                                .cpu().numpy())[np.asarray(VM[i])])
        return np.concatenate(P), best

    t0 = time.time()
    p_img, v_img = train_arm(1, False, "photo only")
    p_both, v_both = train_arm(2, True, "photo + shape")
    T = np.concatenate([np.asarray(DP[i], np.float32)[np.asarray(VM[i])] for i in te_i])
    np.savez_compressed(OUT, truth=T, photo=p_img, both=p_both, const=const)
    print(f"saved {OUT}   {time.time()-t0:.0f}s", flush=True)

    arms = {"constant": np.full_like(T, const), "photo": p_img, "photo+shape": p_both}

    print(f"\ntruth depth, nm:  min {T.min():.0f}  p1 {np.percentile(T,1):.0f}  "
          f"p25 {np.percentile(T,25):.0f}  median {np.median(T):.0f}  "
          f"p75 {np.percentile(T,75):.0f}  p99 {np.percentile(T,99):.0f}  max {T.max():.0f}")
    print(f"\n{'arm':<13s}{'min':>8s}{'p1':>8s}{'p25':>8s}{'median':>9s}{'p75':>8s}"
          f"{'p99':>9s}{'max':>10s}")
    for k, p in arms.items():
        print(f"{k:<13s}{p.min():>8.0f}{np.percentile(p,1):>8.0f}{np.percentile(p,25):>8.0f}"
              f"{np.median(p):>9.0f}{np.percentile(p,75):>8.0f}{np.percentile(p,99):>9.0f}"
              f"{p.max():>10.0f}")

    print(f"\nAbsRel by decile of TRUE depth. The denominator is the true value, so the")
    print(f"shallow deciles are where an over-prediction is punished hardest.")
    q = np.quantile(T, np.linspace(0, 1, 11))
    print(f"  {'true depth range':<22s}{'n':>10s}" + "".join(f"{k:>13s}" for k in arms))
    for b in range(10):
        m = (T >= q[b]) & (T < q[b + 1] if b < 9 else T <= q[b + 1])
        if m.sum() < 100: continue
        row = "".join(f"{np.mean(np.abs(p[m]-T[m])/T[m]):>13.3f}" for p in arms.values())
        print(f"  {q[b]:>7.0f} to {q[b+1]:<11.0f}{int(m.sum()):>10,d}{row}")

    print(f"\nhow much of each arm's AbsRel comes from the shallowest decile alone")
    m0 = T < q[1]
    for k, p in arms.items():
        whole = float(np.mean(np.abs(p - T) / T))
        part = float(np.sum(np.abs(p[m0] - T[m0]) / T[m0]) / T.size)
        print(f"  {k:<13s} AbsRel {whole:>7.3f}   from the shallowest 10% {part:>7.3f} "
              f"({100*part/whole:.0f}%)")

    print(f"\nthe same ranking under metrics without a small denominator")
    print(f"  {'arm':<13s}{'AbsRel':>9s}{'MedRel':>9s}{'log10 RMSE':>12s}{'RMSE nm':>10s}"
          f"{'delta1':>9s}{'SIlog':>9s}")
    for k, p in arms.items():
        lg = np.log(np.maximum(p, 1)) - np.log(np.maximum(T, 1))
        r = p / T
        print(f"  {k:<13s}{np.mean(np.abs(p-T)/T):>9.3f}"
              f"{np.median(np.abs(p-T)/T):>9.3f}"
              f"{np.sqrt(np.mean((np.log10(np.maximum(p,1))-np.log10(np.maximum(T,1)))**2)):>12.3f}"
              f"{np.sqrt(np.mean((p-T)**2)):>10.0f}"
              f"{np.mean(np.maximum(r,1/r)<1.25):>9.3f}"
              f"{np.sqrt(np.mean(lg**2)-np.mean(lg)**2):>9.3f}")
    print(f"\n  best val loss, log-space L1:  photo only {v_img:.4f}   photo+shape {v_both:.4f}")


if __name__ == "__main__":
    main()
