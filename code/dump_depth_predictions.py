import json, os, sys
import numpy as np

CACHE = ("/private/tmp/claude-501/-Users-hongchulshin/"
         "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache")
ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FOLDS = ["jrc_mus-thymus-1", "jrc_choroid-plexus-2"]
NKEEP = 14


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
    PX = np.array([float(m["px_nm"]) for m in meta["meta"]], np.float32)
    ZS = np.array([float(m.get("z_step_nm", 0) or 0) for m in meta["meta"]], np.float32)
    ZS[ZS <= 0] = PX[ZS <= 0]

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

    def chans(idx):
        x = np.asarray(EM[idx], np.float32)
        mu = x.mean(axis=(1, 2), keepdims=True); sd = x.std(axis=(1, 2), keepdims=True)
        x = (x - mu) / np.maximum(sd, 1e-3)
        e = np.asarray(EDT[idx], np.float32) / 40.0
        px = PX[idx][:, None, None]; zs = ZS[idx][:, None, None]
        return np.stack([x, e,
                         np.broadcast_to(np.log(px / 16.0), x.shape).astype(np.float32),
                         np.broadcast_to(np.log(zs / 16.0), x.shape).astype(np.float32)], 1)

    out = {}
    for held in FOLDS:
        te_i = np.flatnonzero(ds == held)
        tr_i = np.flatnonzero(ds != held)
        rng = np.random.default_rng(0); rng.shuffle(tr_i)
        va_i, tr_i = tr_i[:180], tr_i[180:]
        print(f"\nhold out {held}  train {len(tr_i):,}  test {len(te_i):,}", flush=True)

        net = UNet(4).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=3)
        best, bad, state = 1e9, 0, None
        rg = np.random.default_rng(7)

        def batch(pool, n):
            k = rg.choice(pool, size=n, replace=False)
            X = chans(k)
            y = np.asarray(DP[k], np.float32) / ZS[k][:, None, None]
            Y = np.log(np.maximum(y, 1e-3))[:, None]
            M = np.asarray(VM[k])[:, None]
            return (torch.from_numpy(np.ascontiguousarray(X)).to(dev),
                    torch.from_numpy(np.ascontiguousarray(Y)).to(dev),
                    torch.from_numpy(np.ascontiguousarray(M)).to(dev))

        def loss_of(p, Y, M):
            d = (p - Y) * M
            n = M.sum().clamp(min=1)
            m1 = d.sum() / n; m2 = (d * d).sum() / n
            return torch.sqrt((m2 - 0.5 * m1 * m1).clamp(min=1e-8))

        for ep in range(35):
            net.train()
            for _ in range(130):
                X, Y, M = batch(tr_i, 14)
                l = loss_of(net(X), Y, M)
                if not torch.isfinite(l): continue
                opt.zero_grad(); l.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            net.eval(); vt = vk = 0
            with torch.no_grad():
                for _ in range(10):
                    X, Y, M = batch(va_i, 14)
                    vt += float(loss_of(net(X), Y, M)); vk += 1
            v = vt / max(vk, 1); sched.step(v)
            if v < best - 1e-4:
                best, bad = v, 0
                state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
            if bad >= 6:
                print(f"  early stop at {ep}, best val {best:.4f}", flush=True); break
        if state: net.load_state_dict(state)
        net.eval()

        order = sorted(te_i, key=lambda i: -float(np.asarray(VM[i]).mean()))[:NKEEP]
        em, tru, pred, val = [], [], [], []
        with torch.no_grad():
            for i in order:
                X = torch.from_numpy(chans(np.array([i]))).to(dev)
                p = np.exp(net(X)[0, 0].cpu().numpy()) * float(ZS[i])
                em.append(np.asarray(EM[i])); tru.append(np.asarray(DP[i], np.float32))
                pred.append(p.astype(np.float32)); val.append(np.asarray(VM[i]))
        np.savez_compressed(f"{CACHE}/qual_{held}.npz",
                            em=np.stack(em), truth=np.stack(tru), pred=np.stack(pred),
                            valid=np.stack(val), px=PX[order], zs=ZS[order])
        out[held] = dict(n=len(order), val=best)
        print(f"  saved {len(order)} planes -> qual_{held}.npz", flush=True)

    json.dump(out, open(f"{ROOT}/cache/qual_dump.json", "w"), indent=1)
    print("\ndone")


if __name__ == "__main__":
    main()
