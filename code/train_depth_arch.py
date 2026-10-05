import argparse, collections, json, math, os, time
import numpy as np

CACHE = ("/private/tmp/claude-501/-Users-hongchulshin/"
         "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache")
ROOT = "/Volumes/One Touch/em-depth-dataset"


def metrics(pred, true):
    r = pred / true
    lg = np.log(np.maximum(pred, 1e-3)) - np.log(np.maximum(true, 1e-3))
    rnd = np.maximum(np.round(pred), 1.0)
    return dict(absrel=float(np.mean(np.abs(pred - true) / true)),
                medrel=float(np.median(np.abs(pred - true) / true)),
                mae_steps=float(np.mean(np.abs(pred - true))),
                d1=float(np.mean(np.maximum(r, 1 / r) < 1.25)),
                d2=float(np.mean(np.maximum(r, 1 / r) < 1.25 ** 2)),
                exact=float(np.mean(rnd == np.round(true))),
                within1=float(np.mean(np.abs(rnd - np.round(true)) <= 1)),
                silog=float(np.sqrt(max(np.mean(lg ** 2) - np.mean(lg) ** 2, 0.0))),
                n=int(true.size))


def stratified_d1(pred, true):
    r = pred / true
    hit = np.maximum(r, 1 / r) < 1.25
    out = {}
    for name, sel in (("1", true < 1.5), ("2", (true >= 1.5) & (true < 2.5)),
                      ("3", (true >= 2.5) & (true < 3.5)),
                      ("4-7", (true >= 3.5) & (true < 7.5)), ("8+", true >= 7.5)):
        out[name] = dict(d1=float(hit[sel].mean()) if sel.any() else None,
                         n=int(sel.sum()))
    return out


def build_net(arch, cin, torch, nn):
    class Block(nn.Module):
        def __init__(s, i, o, dil=1):
            super().__init__()
            g = 8 if o % 8 == 0 else 1
            s.f = nn.Sequential(
                nn.Conv2d(i, o, 3, padding=dil, dilation=dil), nn.GroupNorm(g, o),
                nn.SiLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=dil, dilation=dil), nn.GroupNorm(g, o),
                nn.SiLU(inplace=True))
        def forward(s, x): return s.f(x)

    class UNet(nn.Module):
        def __init__(s, cin, w, levels):
            super().__init__()
            s.levels = levels
            s.enc = nn.ModuleList()
            prev = cin
            for k in range(levels):
                s.enc.append(Block(prev, w[k])); prev = w[k]
            s.bott = Block(prev, w[levels])
            s.up = nn.ModuleList(); s.dec = nn.ModuleList()
            for k in range(levels - 1, -1, -1):
                s.up.append(nn.Conv2d(w[k + 1], w[k], 1))
                s.dec.append(Block(w[k] * 2, w[k]))
            s.out = nn.Conv2d(w[0], 1, 1)
            s.pool = nn.MaxPool2d(2)
        def encode(s, x):
            feats = []
            for k in range(s.levels):
                x = s.enc[k](x); feats.append(x); x = s.pool(x)
            return s.bott(x), feats
        def forward(s, x):
            b, feats = s.encode(x)
            up = nn.functional.interpolate
            for j, k in enumerate(range(s.levels - 1, -1, -1)):
                b = s.dec[j](torch.cat(
                    [up(s.up[j](b), scale_factor=2, mode="nearest"), feats[k]], 1))
            return s.out(b)

    class Dilated(nn.Module):
        def __init__(s, cin, w=64):
            super().__init__()
            layers, prev = [], cin
            for d in (1, 2, 4, 8, 16, 32, 16, 8, 4, 2, 1):
                layers.append(Block(prev, w, dil=d)); prev = w
            s.f = nn.Sequential(*layers); s.out = nn.Conv2d(w, 1, 1)
        def forward(s, x): return s.out(s.f(x))

    class ViT(nn.Module):
        def __init__(s, cin, dim=192, depth=6, heads=6, patch=16, grid=16):
            super().__init__()
            s.patch, s.grid, s.dim = patch, grid, dim
            s.embed = nn.Conv2d(cin, dim, patch, stride=patch)
            s.pos = nn.Parameter(torch.zeros(1, grid * grid, dim))
            nn.init.trunc_normal_(s.pos, std=0.02)
            layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, 0.0,
                                               activation="gelu", batch_first=True,
                                               norm_first=True)
            s.tr = nn.TransformerEncoder(layer, depth)
            s.head = nn.Sequential(
                nn.Conv2d(dim, 128, 3, padding=1), nn.GroupNorm(8, 128), nn.SiLU(),
                nn.Conv2d(128, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.SiLU(),
                nn.Conv2d(64, 1, 1))
        def forward(s, x):
            B, _, H, W = x.shape
            t = s.embed(x).flatten(2).transpose(1, 2) + s.pos
            t = s.tr(t).transpose(1, 2).reshape(B, s.dim, s.grid, s.grid)
            t = nn.functional.interpolate(t, size=(H, W), mode="bilinear",
                                          align_corners=False)
            return s.head(t)

    if arch in ("unet", "mae"):
        return UNet(cin, (32, 64, 128, 192), 3)
    if arch == "deep":
        return UNet(cin, (32, 48, 64, 96, 128, 192), 5)
    if arch == "wide":
        return UNet(cin, (64, 128, 256, 384), 3)
    if arch == "dilated":
        return Dilated(cin)
    if arch == "vit":
        return ViT(cin)
    raise SystemExit(f"unknown arch {arch}")


def pretrain(a, torch, nn, dev, EM, N, CROP):
    net = build_net("unet", 1, torch, nn).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rg = np.random.default_rng(11)
    P = 16
    G = CROP // P
    print(f"pretraining masked autoencoder on {N:,} planes, {a.pre_epochs} epochs",
          flush=True)
    for ep in range(a.pre_epochs):
        tot, k = 0.0, 0
        for _ in range(a.pre_steps):
            idx = rg.choice(N, size=a.batch, replace=False)
            x = np.asarray(EM[idx], np.float32)
            mu = x.mean(axis=(1, 2), keepdims=True)
            sd = np.maximum(x.std(axis=(1, 2), keepdims=True), 1e-3)
            x = (x - mu) / sd
            keep = rg.random((len(idx), G, G)) > a.mask_ratio
            m = np.repeat(np.repeat(keep, P, 1), P, 2).astype(np.float32)
            X = torch.from_numpy((x * m)[:, None]).to(dev)
            Y = torch.from_numpy(x[:, None]).to(dev)
            M = torch.from_numpy(1.0 - m[:, None]).to(dev)
            p = net(X)
            loss = (((p - Y) ** 2) * M).sum() / M.sum().clamp(min=1)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            tot += float(loss); k += 1
        print(f"  epoch {ep + 1:2d}  masked mse {tot / max(k, 1):.4f}", flush=True)
    torch.save(net.state_dict(), a.pre_out)
    print(f"-> {a.pre_out}", flush=True)


def load_pretrained(net, path, cin, torch):
    sd = torch.load(path, map_location="cpu")
    own = net.state_dict()
    copied, skipped = 0, 0
    for k, v in sd.items():
        if k not in own:
            skipped += 1; continue
        if own[k].shape == v.shape:
            own[k] = v; copied += 1
        elif k.endswith("enc.0.f.0.weight") and own[k].shape[1] == cin and v.shape[1] == 1:
            w = own[k].clone(); w[:, :1] = v; own[k] = w; copied += 1
        else:
            skipped += 1
    net.load_state_dict(own)
    return copied, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "pretrain"])
    ap.add_argument("--archs", default="unet,deep,dilated,wide,vit,mae")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--patience", type=int, default=9)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--steps", type=int, default=140)
    ap.add_argument("--edt", type=int, default=1, help="1 feeds the silhouette")
    ap.add_argument("--pre-epochs", type=int, default=12)
    ap.add_argument("--pre-steps", type=int, default=400)
    ap.add_argument("--mask-ratio", type=float, default=0.6)
    ap.add_argument("--pre-out", default=f"{ROOT}/cache/mae_encoder.pt")
    ap.add_argument("--out", default="cache/depth_arch.json")
    a = ap.parse_args()

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
    print(f"device {dev}   {N:,} planes at {CROP} px", flush=True)

    if a.mode == "pretrain":
        pretrain(a, torch, nn, dev, EM, N, CROP)
        return

    counts = collections.Counter(ds.tolist())
    folds = ([v for v, _ in counts.most_common()] if a.folds == 0
             else [v for v, _ in counts.most_common(a.folds)])
    archs = [x.strip() for x in a.archs.split(",") if x.strip()]
    print(f"archs {archs}   folds {len(folds)}   silhouette {'on' if a.edt else 'off'}",
          flush=True)
    cin = 1 + (1 if a.edt else 0) + 2

    def make_batch(idx, rg, aug):
        x = np.asarray(EM[idx], np.float32)
        mu = x.mean(axis=(1, 2), keepdims=True)
        sd = np.maximum(x.std(axis=(1, 2), keepdims=True), 1e-3)
        x = (x - mu) / sd
        m = np.asarray(VM[idx])
        zs = ZS[idx][:, None, None]; px = PX[idx][:, None, None]
        ch = [x]
        if a.edt:
            ch.append(np.asarray(EDT[idx], np.float32) / 40.0)
        ch.append(np.broadcast_to(np.log(px / 16.0), x.shape).astype(np.float32))
        ch.append(np.broadcast_to(np.log(zs / 16.0), x.shape).astype(np.float32))
        X = np.stack(ch, 1)
        y = np.asarray(DP[idx], np.float32) / zs
        Y = np.log(np.maximum(y, 1e-3))[:, None]
        if aug:
            if rg.random() < 0.5:
                X, Y, m = X[..., ::-1, :], Y[..., ::-1, :], m[:, ::-1]
            if rg.random() < 0.5:
                X, Y, m = X[..., ::-1], Y[..., ::-1], m[..., ::-1]
        return (np.ascontiguousarray(X), np.ascontiguousarray(Y),
                np.ascontiguousarray(m)[:, None])

    results = collections.defaultdict(dict)
    outpath = f"{ROOT}/{a.out}"

    for held in folds:
        te_i = np.flatnonzero(ds == held)
        tr_i = np.flatnonzero(ds != held)
        rng = np.random.default_rng(0); rng.shuffle(tr_i)
        va_i, tr_i = tr_i[:240], tr_i[240:]
        T = np.concatenate([np.asarray(DP[i], np.float32)[np.asarray(VM[i])] / ZS[i]
                            for i in te_i])
        print(f"\n{'=' * 78}\nhold out {held}   train {len(tr_i):,}  test {len(te_i):,}  "
              f"scored pixels {T.size:,}", flush=True)

        sub = tr_i[:900]
        dtr = np.concatenate([np.asarray(DP[i], np.float32)[np.asarray(VM[i])] / ZS[i]
                              for i in sub])
        results[held]["constant"] = metrics(np.full_like(T, float(np.median(dtr))), T)
        print(f"  {'constant':<9s} d1 {results[held]['constant']['d1']:.3f}", flush=True)

        for arch in archs:
            t0 = time.time()
            net = build_net(arch, cin, torch, nn).to(dev)
            note = ""
            if arch == "mae":
                if not os.path.exists(a.pre_out):
                    print(f"  {arch:<9s} skipped, no {a.pre_out}", flush=True); continue
                c, s = load_pretrained(net, a.pre_out, cin, torch)
                note = f" init {c} tensors, {s} left fresh"
            opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=3)
            best, bad, state = 1e9, 0, None
            rg = np.random.default_rng(7)

            def loss_of(p, Y, M):
                d = (p - Y) * M
                n = M.sum().clamp(min=1)
                m1 = d.sum() / n
                m2 = (d * d).sum() / n
                return torch.sqrt((m2 - 0.5 * m1 * m1).clamp(min=1e-8))

            for ep in range(a.epochs):
                net.train()
                for _ in range(a.steps):
                    idx = rg.choice(tr_i, size=a.batch, replace=False)
                    X, Y, M = make_batch(idx, rg, True)
                    X = torch.from_numpy(X).to(dev); Y = torch.from_numpy(Y).to(dev)
                    M = torch.from_numpy(M).to(dev)
                    l = loss_of(net(X), Y, M)
                    if not torch.isfinite(l):
                        continue
                    opt.zero_grad(); l.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
                net.eval(); vt = vk = 0
                with torch.no_grad():
                    for _ in range(16):
                        idx = rg.choice(va_i, size=a.batch, replace=False)
                        X, Y, M = make_batch(idx, rg, False)
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
                    X, _, _ = make_batch(np.array([i]), rg, False)
                    p = np.exp(net(torch.from_numpy(X).to(dev))[0, 0].cpu().numpy())
                    P.append(p[np.asarray(VM[i])])
            pred = np.concatenate(P)
            m = metrics(pred, T)
            m["by_depth"] = stratified_d1(pred, T)
            m["epochs_run"] = ep + 1
            m["seconds"] = round(time.time() - t0)
            results[held][arch] = m
            print(f"  {arch:<9s} d1 {m['d1']:.3f}  d2 {m['d2']:.3f}  "
                  f"exact {m['exact']:.3f}  w1 {m['within1']:.3f}  "
                  f"MedRel {m['medrel']:.3f}  MAE {m['mae_steps']:.2f} steps  "
                  f"{m['seconds']}s{note}", flush=True)
            json.dump(results, open(outpath, "w"), indent=1)

    print(f"\n{'=' * 78}\nmean over {len(results)} held-out source volumes")
    print(f"  {'arch':<10s}{'d1':>8s}{'d2':>8s}{'exact':>8s}{'within1':>9s}"
          f"{'MedRel':>9s}{'MAEstep':>9s}")
    for k in ["constant"] + archs:
        vals = [r[k] for r in results.values() if k in r]
        if not vals:
            continue
        print(f"  {k:<10s}"
              f"{np.mean([v['d1'] for v in vals]):>8.3f}"
              f"{np.mean([v['d2'] for v in vals]):>8.3f}"
              f"{np.mean([v['exact'] for v in vals]):>8.3f}"
              f"{np.mean([v['within1'] for v in vals]):>9.3f}"
              f"{np.mean([v['medrel'] for v in vals]):>9.3f}"
              f"{np.mean([v['mae_steps'] for v in vals]):>9.2f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
