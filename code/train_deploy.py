import argparse, json, os, sys, time
import numpy as np

ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, f"{ROOT}/autoresearch")
sys.path.insert(0, f"{ROOT}/code")
from harness import Data
from relmetrics import per_face, aggregate
from sweep_arms import load_arm, default_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="head_midas_ssi")
    ap.add_argument("--holdout", default="jrc_mus-thymus-1",
                    help="kept out so the shipped model can be sanity checked")
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--no-silhouette", action="store_true",
                    help="the server has no labels, so the deployed model must not need them")
    ap.add_argument("--out", default="server/model.pt")
    a = ap.parse_args()

    import torch
    from torch import nn
    dev = "mps" if torch.backends.mps.is_available() else "cpu"

    data = Data(silhouette=not a.no_silhouette)
    arm = load_arm(f"{ROOT}/code/arms/{a.arm}.py")
    loss_fn = getattr(arm, "LOSS", None)
    decode = getattr(arm, "DECODE", None)

    te = np.flatnonzero(data.ds == a.holdout)
    tr = np.flatnonzero(data.ds != a.holdout)
    rng = np.random.default_rng(0); rng.shuffle(tr)
    va, tr = tr[:300], tr[300:]
    print(f"device {dev}   arm {a.arm}   cin {data.cin}   "
          f"train {len(tr):,}  val {len(va)}  holdout {a.holdout} {len(te)}", flush=True)

    net = arm.build(data.cin).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.4, patience=3)
    rg = np.random.default_rng(7)
    t0 = time.time()
    best, bad, state = 1e9, 0, None

    def L(p, Y, M):
        return loss_fn(p, Y, M) if loss_fn is not None else default_loss(p, Y, M)

    while time.time() - t0 < a.budget:
        net.train()
        for _ in range(140):
            if time.time() - t0 > a.budget:
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
            for _ in range(16):
                idx = rg.choice(va, size=8, replace=False)
                X, Y, M = data.batch(idx)
                X = torch.from_numpy(X).to(dev); Y = torch.from_numpy(Y).to(dev)
                M = torch.from_numpy(M).to(dev)
                vt += float(L(net(X), Y, M)); vk += 1
        v = vt / max(vk, 1); sched.step(v)
        mark = ""
        if v < best - 1e-4:
            best, bad = v, 0
            state = {k: t.detach().cpu().clone() for k, t in net.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        print(f"  {int(time.time()-t0):5d}s  val {v:.4f}{mark}", flush=True)
        if bad >= 8:
            break
    if state:
        net.load_state_dict(state)
    net.eval()

    faces, rrng = [], np.random.default_rng(1234)
    with torch.no_grad():
        for i in te:
            X, _, _ = data.batch(np.array([i]))
            p = net(torch.from_numpy(X).to(dev))
            d = (decode(p) if decode is not None else torch.exp(p[:, 0])).cpu().numpy()
            vm = np.asarray(data.vm[i])
            faces.append(per_face(np.maximum(d[0], 1e-3)[vm],
                                  np.asarray(data.dp[i], np.float32)[vm] / data.zs[i],
                                  rrng))
    rel = aggregate(faces)
    print(f"\nheld out {a.holdout}:  pair_acc {rel['pair_acc']:.4f}  "
          f"spearman {rel['spearman']:+.3f}  ssi_rel {rel['ssi_rel']:.3f}  "
          f"over {rel['faces']} faces", flush=True)

    out = f"{ROOT}/{a.out}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(dict(state=net.state_dict(), arm=a.arm, cin=data.cin,
                    silhouette=not a.no_silhouette, crop=data.crop,
                    holdout=a.holdout, params=npar, val=best, relative=rel,
                    trained_seconds=int(time.time() - t0)), out)
    json.dump(dict(arm=a.arm, cin=data.cin, silhouette=not a.no_silhouette,
                   holdout=a.holdout, params=npar, relative=rel),
              open(out.replace(".pt", ".json"), "w"), indent=1)
    print(f"-> {a.out}  ({npar/1e6:.2f}M params)")


if __name__ == "__main__":
    main()
