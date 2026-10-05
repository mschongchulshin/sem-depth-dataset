import argparse, importlib.util, json, os, sys, time, traceback
import numpy as np

ROOT = "/Volumes/One Touch/em-depth-dataset"
ARMS = f"{ROOT}/code/arms"
sys.path.insert(0, f"{ROOT}/autoresearch")
from harness import Data, metrics
sys.path.insert(0, f"{ROOT}/code")
from relmetrics import per_face, aggregate


def load_arm(path):
    spec = importlib.util.spec_from_file_location(
        "arm_" + os.path.basename(path)[:-3], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def default_loss(p, Y, M, lam=1.0):
    import torch
    d = (p - Y) * M
    n = M.sum().clamp(min=1)
    m1 = d.sum() / n
    m2 = (d * d).sum() / n
    return torch.sqrt((m2 - lam * m1 * m1).clamp(min=1e-8))


def run_arm(arm, data, folds, budget_s, dev, seed=0):
    import torch
    from torch import nn
    loss_fn = getattr(arm, "LOSS", None)
    decode = getattr(arm, "DECODE", None)
    out = {}
    for held in folds:
        te = np.flatnonzero(data.ds == held)
        tr = np.flatnonzero(data.ds != held)
        rng = np.random.default_rng(seed); rng.shuffle(tr)
        va, tr = tr[:240], tr[240:]
        T = data.truth_steps(te)
        Z = data.z_steps_nm(te)

        t0 = time.time()
        net = arm.build(data.cin).to(dev)
        npar = sum(p.numel() for p in net.parameters())
        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=2)
        best, bad, state = 1e9, 0, None
        rg = np.random.default_rng(seed + 7)

        def L(p, Y, M):
            return loss_fn(p, Y, M) if loss_fn is not None else default_loss(p, Y, M)

        while time.time() - t0 < budget_s:
            net.train()
            for _ in range(140):
                if time.time() - t0 > budget_s:
                    break
                idx = rg.choice(tr, size=8, replace=False)
                X, Y, M = data.batch(idx, rg, aug=True)
                X = torch.from_numpy(X).to(dev); Y = torch.from_numpy(Y).to(dev)
                M = torch.from_numpy(M).to(dev)
                l = L(net(X), Y, M)
                if not torch.isfinite(l):
                    continue
                opt.zero_grad(); l.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            net.eval(); vt = vk = 0.0
            with torch.no_grad():
                for _ in range(12):
                    idx = rg.choice(va, size=8, replace=False)
                    X, Y, M = data.batch(idx)
                    X = torch.from_numpy(X).to(dev); Y = torch.from_numpy(Y).to(dev)
                    M = torch.from_numpy(M).to(dev)
                    vt += float(L(net(X), Y, M)); vk += 1
            v = vt / max(vk, 1); sched.step(v)
            if v < best - 1e-4:
                best, bad = v, 0
                state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
            if bad >= 5:
                break
        if state:
            net.load_state_dict(state)
        net.eval()

        P, faces = [], []
        rrng = np.random.default_rng(1234)
        with torch.no_grad():
            for i in te:
                X, _, _ = data.batch(np.array([i]))
                p = net(torch.from_numpy(X).to(dev))
                d = decode(p) if decode is not None else torch.exp(p[:, 0])
                d = d.cpu().numpy()
                vm = np.asarray(data.vm[i])
                pv = np.maximum(d[0], 1e-3)[vm]
                P.append(pv)
                faces.append(per_face(pv, np.asarray(data.dp[i], np.float32)[vm]
                                      / data.zs[i], rrng))
        m = metrics(np.concatenate(P), T, Z)
        m.update(aggregate(faces))
        m["seconds"] = round(time.time() - t0)
        m["params_m"] = round(npar / 1e6, 2)
        out[held] = m
        print(f"    {held:24s} d1_al {m['d1_aligned']:.4f}  logIQR "
              f"{m['log_ratio_iqr']:.3f}  ssi {m['ssi_rel']:.3f}  "
              f"pair {m['pair_acc']:.3f}  |  d1raw {m['d1']:.3f}  "
              f"{m['seconds']}s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma separated slugs, blank means all")
    ap.add_argument("--folds", default="jrc_mus-thymus-1,jrc_macrophage-2,jrc_jurkat-1")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--out", default="cache/sweep_arms.json")
    a = ap.parse_args()

    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    data = Data(silhouette=True)
    folds = [f.strip() for f in a.folds.split(",") if f.strip()]

    files = sorted(f for f in os.listdir(ARMS) if f.endswith(".py"))
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        files = [f for f in files if f[:-3] in want]
    print(f"device {dev}   {len(files)} arms   {len(folds)} folds   "
          f"{a.budget}s per fold\n", flush=True)

    outpath = f"{ROOT}/{a.out}"
    results = json.load(open(outpath)) if os.path.exists(outpath) else {}
    for f in files:
        slug = f[:-3]
        if slug in results and results[slug].get("ok"):
            print(f"{slug:26s} already done, d1 {results[slug]['d1']:.4f}", flush=True)
            continue
        print(f"{slug}", flush=True)
        rec = {}
        try:
            arm = load_arm(f"{ARMS}/{f}")
            rec["paper"] = getattr(arm, "PAPER", "")
            rec["repo"] = getattr(arm, "REPO", None)
            rec["idea"] = getattr(arm, "IDEA", "")
            per = run_arm(arm, data, folds, a.budget, dev)
            rec["per_fold"] = per
            for k in ("d1_aligned", "d2_aligned", "absrel_aligned", "medrel_aligned",
                      "log_ratio_iqr", "scale_k", "pair_acc", "spearman", "ssi_rel",
                      "mae_nm", "medae_nm", "rmse_nm", "mae_pct_of_range",
                      "d1", "exact", "medrel"):
                rec[k] = float(np.mean([v[k] for v in per.values()]))
            rec["params_m"] = list(per.values())[0]["params_m"]
            rec["ok"] = True
            print(f"  -> d1_aligned {rec['d1_aligned']:.4f}  "
                  f"logIQR {rec['log_ratio_iqr']:.3f}  "
                  f"pair {rec['pair_acc']:.3f}\n", flush=True)
        except Exception:
            rec["ok"] = False
            rec["error"] = traceback.format_exc()[-1200:]
            print(f"  -> FAILED\n{rec['error'][-400:]}\n", flush=True)
        results[slug] = rec
        json.dump(results, open(outpath, "w"), indent=1)

    ok = {k: v for k, v in results.items() if v.get("ok")}
    print(f"\n{'=' * 74}\nranked, {len(ok)} arms scored, "
          f"{len(results) - len(ok)} failed")
    print("primary is scale only alignment, one constant multiplier and no offset\n")
    print(f"{'arm':24s}{'d1_al':>7s}{'d2_al':>7s}{'medrel':>8s}{'logIQR':>8s}"
          f"{'ssi':>7s}{'pair':>7s}{'rho':>7s}{'d1raw':>7s}")
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1].get("d1_aligned", 0)):
        print(f"{k:24s}{v.get('d1_aligned',0):7.3f}{v.get('d2_aligned',0):7.3f}"
              f"{v.get('medrel_aligned',0):8.3f}{v.get('log_ratio_iqr',0):8.3f}"
              f"{v.get('ssi_rel',0):7.3f}{v.get('pair_acc',0):7.3f}"
              f"{v.get('spearman',0):+7.3f}{v.get('d1',0):7.3f}")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
