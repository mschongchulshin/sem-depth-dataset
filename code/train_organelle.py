import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


GEOMETRIES = {
    "surface": dict(
        subdir="views", image=("se", "bse"), depth="depth_nm", thick="thickness_nm",
        inst="inst_front", mask="mask", clip=None,
        area_field="proj_area_nm2", v2d_field="volume_from_proj_nm3",
        bad_flags=("touches_border",),
        note="rendered SE plus four BSE quadrants",
    ),
    "blockface": dict(
        subdir="faces", image=("em",), depth="depth_below_nm", thick="thickness_below_nm",
        inst="inst_face", mask="mask", clip="clipped",
        area_field="face_area_nm2", v2d_field="volume_from_face_nm3",
        bad_flags=("touches_face_border", "clipped_at_block_bottom"),
        note="the milled block face, detector-derived",
    ),
    "thinsection": dict(
        subdir="sections", image=("image",), depth="extent_above_nm", thick="total_thickness_nm",
        inst="inst_section", mask="mask", clip="clipped",
        area_field="profile_area_nm2", v2d_field="volume_from_profile_nm3",
        bad_flags=("touches_border", "clipped"),
        note="70 nm slab in transmission, detector-derived",
    ),
}


def list_views(root, geometry="surface"):
    sub = GEOMETRIES[geometry]["subdir"]
    root = Path(root)
    hits = list(root.glob(f"{sub}/*.npz")) + list(root.glob(f"*/{sub}/*.npz"))
    return sorted(p for p in hits if not p.name.startswith("._"))

class OrganelleViews(Dataset):
    def __init__(self, files, channels="se", size=256, augment=False, repeat=1, seed=0,
                 px_channel=True, geometry="surface"):
        self.files = list(files) * max(1, int(repeat))
        self.channels, self.size = channels, size
        self.augment = augment
        self.px_channel = px_channel
        self.geometry = geometry
        self.spec = GEOMETRIES[geometry]
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        d = np.load(self.files[i], allow_pickle=False)
        sp = self.spec
        if self.geometry == "surface":
            se, bse = d["se"], d["bse"]
            x = {"se": se[None], "bse": bse,
                 "both": np.concatenate([se[None], bse], 0)}[self.channels]
        else:
            a = np.asarray(d[sp["image"][0]], dtype=np.float32)
            lo, hi = np.percentile(a, [1, 99])
            x = (np.clip((a - lo) / (hi - lo + 1e-6), 0, 1))[None]
        depth = np.nan_to_num(d[sp["depth"]], nan=0.0)
        thick = np.nan_to_num(d[sp["thick"]], nan=0.0)
        mask = d[sp["mask"]].astype(np.float32)
        if sp["clip"] is not None and sp["clip"] in d.files:
            mask = mask * (~d[sp["clip"]].astype(bool)).astype(np.float32)
        inst = d[sp["inst"]].astype(np.int64)
        px = float(json.loads(str(d["meta"]))["pixel_size_nm"])

        if self.augment:
            if self.rng.random() < 0.5:
                x, depth, thick, mask, inst = (a[..., ::-1] for a in (x, depth, thick, mask, inst))
            if self.rng.random() < 0.5:
                x = x[..., ::-1, :]
                depth, thick, mask, inst = (a[::-1] for a in (depth, thick, mask, inst))
            if self.rng.random() < 0.5:
                x = x.transpose(0, 2, 1)
                depth, thick, mask, inst = (a.T for a in (depth, thick, mask, inst))

        H, W = mask.shape
        if self.size and H >= self.size and W >= self.size:
            if self.augment:
                y0 = int(self.rng.integers(0, H - self.size + 1))
                x0 = int(self.rng.integers(0, W - self.size + 1))
            else:
                y0, x0 = (H - self.size) // 2, (W - self.size) // 2
            sl = (slice(y0, y0 + self.size), slice(x0, x0 + self.size))
            x = x[(slice(None),) + sl]
            depth, thick, mask, inst = (a[sl] for a in (depth, thick, mask, inst))
            px_eff = px
        else:
            px_eff = px * (W / float(self.size)) if self.size else px

        t = [torch.from_numpy(np.ascontiguousarray(x)).float(),
             torch.from_numpy(np.ascontiguousarray(depth))[None].float(),
             torch.from_numpy(np.ascontiguousarray(thick))[None].float(),
             torch.from_numpy(np.ascontiguousarray(mask))[None].float(),
             torch.from_numpy(np.ascontiguousarray(inst))[None]]
        if self.size and t[0].shape[-1] != self.size:
            t[0] = F.interpolate(t[0][None], self.size, mode="bilinear", align_corners=False)[0]
            for k in (1, 2, 3):
                t[k] = F.interpolate(t[k][None], self.size, mode="nearest")[0]
            t[4] = F.interpolate(t[4][None].float(), self.size, mode="nearest")[0].long()

        if self.px_channel:
            px_ch = torch.full_like(t[0][:1], float(np.log2(px_eff / 8.0)))
            t[0] = torch.cat([t[0], px_ch], 0)
        return t[0], t[1], t[2], t[3], t[4], str(self.files[i]), float(px_eff)


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU(),
                         nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.GELU())


class TwoHeadUNet(nn.Module):
    def __init__(self, cin=1, w=(32, 64, 128, 256)):
        super().__init__()
        self.enc, c = nn.ModuleList(), cin
        for ww in w:
            self.enc.append(block(c, ww))
            c = ww
        self.dec = nn.ModuleList([block(w[i] + w[i - 1], w[i - 1])
                                  for i in range(len(w) - 1, 0, -1)])
        self.depth_head = nn.Conv2d(w[0], 1, 1)
        self.thick_head = nn.Conv2d(w[0], 1, 1)
        nn.init.zeros_(self.thick_head.weight)
        nn.init.constant_(self.thick_head.bias, 150.0)

    def forward(self, x):
        feats = []
        for i, e in enumerate(self.enc):
            x = e(x if i == 0 else F.max_pool2d(x, 2))
            feats.append(x)
        x = feats[-1]
        for i, dc in enumerate(self.dec):
            skip = feats[-2 - i]
            x = F.interpolate(x, skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dc(torch.cat([x, skip], 1))
        return self.depth_head(x), F.softplus(self.thick_head(x))


def scale_shift_align(pred, target, mask):
    out = torch.zeros_like(pred)
    for i in range(pred.shape[0]):
        m = mask[i] > 0.5
        if m.sum() < 64:
            out[i] = pred[i]
            continue
        p, t = pred[i][m], target[i][m]
        pm, tm = p.mean(), t.mean()
        pc = p - pm
        var = (pc * pc).mean()
        if var < 1e-12:
            out[i] = pred[i] - pm + tm
            continue
        a = (pc * (t - tm)).mean() / var
        b = tm - a * pm
        out[i] = a * pred[i] + b
    return out


def normal_consistency(pred, target, mask, eps=1e-6):
    def normals(d):
        gy = d[..., 1:, :-1] - d[..., :-1, :-1]
        gx = d[..., :-1, 1:] - d[..., :-1, :-1]
        n = torch.stack([-gx, -gy, torch.ones_like(gx)], 1).squeeze(2)
        return n / (n.norm(dim=1, keepdim=True) + eps)

    np_, nt = normals(pred), normals(target)
    m = (mask[..., :-1, :-1] > 0.5).squeeze(1)
    if m.sum() < 16:
        return pred.sum() * 0.0
    cos = (np_ * nt).sum(1)
    return (1.0 - cos)[m].mean()


def losses(pd, pt, depth, thick, mask):
    m = mask > 0.5
    if m.sum() < 64:
        return pd.sum() * 0.0, {}
    aligned = scale_shift_align(pd, depth, mask)

    scale = depth[m].std().detach().clamp(min=1.0)
    l_depth = (aligned - depth).abs()[m].mean() / scale
    gx = ((aligned[..., 1:] - aligned[..., :-1]) - (depth[..., 1:] - depth[..., :-1])).abs()
    gy = ((aligned[..., 1:, :] - aligned[..., :-1, :]) -
          (depth[..., 1:, :] - depth[..., :-1, :])).abs()
    l_grad = (gx.mean() + gy.mean()) / scale
    l_thick = (torch.log1p(pt[m]) - torch.log1p(thick[m])).abs().mean()

    l_norm = normal_consistency(aligned, depth, mask)
    total = l_depth + 0.5 * l_grad + 2.0 * l_thick + 0.5 * l_norm
    return total, dict(depth_rel=l_depth.item(), thick_log_l1=l_thick.item(),
                       normal=l_norm.item())


@torch.no_grad()
def evaluate(model, loader, dev, px_nm_by_file=None, geometry="surface"):
    model.eval()
    depth_absrel, thick_absrel, n = 0.0, 0.0, 0
    rows = []
    for x, depth, thick, mask, inst, paths, px_eff in loader:
        x, depth, thick, mask = x.to(dev), depth.to(dev), thick.to(dev), mask.to(dev)
        pd, pt = model(x)
        pd = scale_shift_align(pd, depth, mask)
        m = (mask > 0.5) & (depth > 1e-3)
        if m.sum() > 0:
            depth_absrel += ((pd[m] - depth[m]).abs() / depth[m]).mean().item()
            n += 1
        mt = (mask > 0.5) & (thick > 1e-3)
        if mt.sum() > 0:
            thick_absrel += ((pt[mt] - thick[mt]).abs() / thick[mt]).mean().item()

        for b in range(x.shape[0]):
            px = float(px_eff[b])
            meta = json.loads(str(np.load(paths[b], allow_pickle=False)["meta"]))
            ids = inst[b, 0].cpu().numpy()
            gspec = GEOMETRIES[geometry]
            for sid, rec in meta["instances"].items():
                if not rec.get("volume_nm3"):
                    continue
                if any(rec.get(f) for f in gspec["bad_flags"]):
                    continue
                sel = torch.from_numpy(ids == int(sid)).to(dev)
                if sel.sum() < 16:
                    continue
                v_true = rec["volume_nm3"]
                v_pred = float(pt[b, 0][sel].sum().item()) * px * px
                v_2d = rec.get(gspec["v2d_field"])
                rows.append(dict(organelle=rec.get("organelle", "?"), px_nm=px,
                                 v_true=v_true,
                                 v_pred=v_pred, v_2d=v_2d,
                                 occluded=rec.get("occluded_fraction"),
                                 err_pred=abs(v_pred - v_true) / v_true,
                                 err_2d=(abs(v_2d - v_true) / v_true) if v_2d else None))
    model.train()
    out = dict(depth_absrel=depth_absrel / max(n, 1),
               thick_absrel=thick_absrel / max(n, 1), n_instances=len(rows))
    if rows:
        out["volume_absrel_model"] = float(np.median([r["err_pred"] for r in rows]))
        e2 = [r["err_2d"] for r in rows if r["err_2d"] is not None]
        if e2:
            out["volume_absrel_2d_only"] = float(np.median(e2))
        bands = [(0.0, 0.3), (0.3, 0.6), (0.6, 0.9), (0.9, 1.01)]
        strat = {}
        for lo, hi in bands:
            sel = [r for r in rows if r.get("occluded") is not None and lo <= r["occluded"] < hi]
            if sel:
                strat[f"occl_{lo:.1f}_{hi:.1f}"] = dict(
                    n=len(sel),
                    model=float(np.median([r["err_pred"] for r in sel])),
                    two_d=float(np.median([r["err_2d"] for r in sel
                                           if r["err_2d"] is not None])) if any(
                        r["err_2d"] is not None for r in sel) else None)
        out["by_occlusion"] = strat

        grid = {}
        for px in sorted({r["px_nm"] for r in rows}):
            per = {}
            at_px = [r for r in rows if r["px_nm"] == px]
            for org in sorted({r["organelle"] for r in at_px}):
                sel = [r for r in at_px if r["organelle"] == org]
                e2 = [r["err_2d"] for r in sel if r["err_2d"] is not None]
                per[org] = dict(n=len(sel),
                                model=float(np.median([r["err_pred"] for r in sel])),
                                two_d=float(np.median(e2)) if e2 else None)
            allm = float(np.median([r["err_pred"] for r in at_px]))
            alle = [r["err_2d"] for r in at_px if r["err_2d"] is not None]
            per["__all__"] = dict(n=len(at_px), model=allm,
                                  two_d=float(np.median(alle)) if alle else None)
            grid[f"px_{px:g}nm"] = per
        out["by_scale_class"] = grid
    return out, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--geometry", default="surface", choices=sorted(GEOMETRIES),
                    help="which acquisition geometry to train on. blockface and thinsection "
                         "carry detector-derived pixels and are 64,147 of the corpus images")
    ap.add_argument("--channels", default="se", choices=["se", "bse", "both"],
                    help="surface only. blockface and thinsection are single-channel")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--balance-datasets", action="store_true",
                    help="sample each dataset equally instead of by natural view count")
    ap.add_argument("--overfit-batch", type=int, default=0,
                    help="fail-fast check: train on this many views only and expect loss -> 0")
    ap.add_argument("--no-px-channel", action="store_true",
                    help="drop the pixel-size input channel; the scale-conditioning ablation")
    ap.add_argument("--train-scales", default=None,
                    help="comma-separated pixel sizes in nm to train on, e.g. 16")
    ap.add_argument("--val-scales", default=None,
                    help="comma-separated pixel sizes in nm to validate on, e.g. 4,8,16")
    ap.add_argument("--split-by-cell", action="store_true",
                    help="hold out whole cells, the only split that measures generalisation")
    ap.add_argument("--holdout-cells", default=None,
                    help="comma-separated dataset names to hold out explicitly")
    ap.add_argument("--repeat", type=int, default=8,
                    help="augmented passes over each training view per epoch")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    roots = [Path(r) for r in a.root.split(",")]
    files = []
    for r in roots:
        files += list_views(r, a.geometry)
        for sub in sorted(p for p in r.iterdir() if p.is_dir()) if r.is_dir() else []:
            files += list_views(sub, a.geometry)
    files = sorted(set(files))
    if not files:
        raise SystemExit(f"no {a.geometry} files under {a.root} "
                         f"(looking for {GEOMETRIES[a.geometry]['subdir']}/*.npz)")
    if a.geometry != "surface" and a.channels != "se":
        raise SystemExit(f"--channels {a.channels} is surface-only; {a.geometry} has a single "
                         f"image array ({GEOMETRIES[a.geometry]['image'][0]})")
    print(f"{a.geometry}: {len(files):,} files  [{GEOMETRIES[a.geometry]['note']}]", flush=True)
    px = {}
    for f in files:
        px[f.name] = float(json.loads(str(np.load(f, allow_pickle=False)["meta"]))["pixel_size_nm"])

    def cell_of(path):
        return Path(path).name.split("__")[0]

    rng = np.random.default_rng(0)
    if a.holdout_cells:
        held = set(a.holdout_cells.split(","))
        val = [f for f in files if cell_of(f) in held]
        trn = [f for f in files if cell_of(f) not in held]
        if not val or not trn:
            raise SystemExit(f"holdout {held} leaves an empty split; cells present: "
                             f"{sorted({cell_of(f) for f in files})}")
    elif a.split_by_cell:
        cells = sorted({cell_of(f) for f in files})
        if len(cells) < 2:
            raise SystemExit(f"--split-by-cell needs 2+ cells, found {cells}")
        nval = max(1, int(round(len(cells) * a.val_frac)))
        held = set(np.array(cells)[rng.permutation(len(cells))[:nval]].tolist())
        val = [f for f in files if cell_of(f) in held]
        trn = [f for f in files if cell_of(f) not in held]
    else:
        idx = rng.permutation(len(files))
        nval = max(1, int(len(files) * a.val_frac))
        val = [files[i] for i in idx[:nval]]
        trn = [files[i] for i in idx[nval:]] or val
        held = {"<random view split>"}
    if a.train_scales:
        keep = {float(v) for v in a.train_scales.split(",")}
        trn = [f for f in trn if px[f.name] in keep]
        if not trn:
            raise SystemExit(f"--train-scales {sorted(keep)} matches nothing; "
                             f"present: {sorted(set(px.values()))}")
    if a.val_scales:
        keep = {float(v) for v in a.val_scales.split(",")}
        val = [f for f in files if px[f.name] in keep and f not in set(trn)]
        if not val:
            raise SystemExit(f"--val-scales {sorted(keep)} matches nothing outside train; "
                             f"present: {sorted(set(px.values()))}")
    if a.train_scales or a.val_scales:
        print(f"train px: {sorted({px[f.name] for f in trn})}  "
              f"val px: {sorted({px[f.name] for f in val})}", flush=True)

    if a.overfit_batch:
        trn = trn[:a.overfit_batch]
        val = trn
        print(f"OVERFIT MODE: {len(trn)} views, expecting loss -> 0", flush=True)
    print(f"{len(trn)} train / {len(val)} val views, channels={a.channels}", flush=True)
    print(f"train cells: {sorted({cell_of(f) for f in trn})}", flush=True)
    print(f"held-out   : {sorted(held)}", flush=True)

    use_px = not a.no_px_channel
    tr_ds = OrganelleViews(trn, a.channels, a.size, augment=True, repeat=a.repeat,
                           px_channel=use_px, geometry=a.geometry)
    if a.balance_datasets:
        import collections
        from torch.utils.data import WeightedRandomSampler
        cells = [cell_of(f) for f in tr_ds.files]
        per = collections.Counter(cells)
        w = torch.tensor([1.0 / per[c] for c in cells], dtype=torch.double)
        sampler = WeightedRandomSampler(w, num_samples=len(tr_ds), replacement=True)
        ltr = DataLoader(tr_ds, batch_size=a.batch, sampler=sampler)
        print(f"dataset-balanced sampling over {len(per)} cells: "
              f"{dict(sorted(per.items()))}", flush=True)
    else:
        ltr = DataLoader(tr_ds, batch_size=a.batch, shuffle=True)
    lva = DataLoader(OrganelleViews(val, a.channels, a.size, px_channel=use_px,
                                    geometry=a.geometry),
                     batch_size=1, shuffle=False)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    n_img = {"se": 1, "bse": 4, "both": 5}[a.channels] if a.geometry == "surface" else 1
    cin = n_img + (1 if use_px else 0)
    model = TwoHeadUNet(cin).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr,
                                                total_steps=a.epochs * max(len(ltr), 1))
    print(f"device={dev} params={sum(p.numel() for p in model.parameters()) / 1e6:.2f}M",
          flush=True)

    hist = []
    for ep in range(a.epochs):
        tot, k = 0.0, 0
        for x, depth, thick, mask, inst, _, _ in ltr:
            x, depth, thick, mask = x.to(dev), depth.to(dev), thick.to(dev), mask.to(dev)
            pd, pt = model(x)
            loss, parts = losses(pd, pt, depth, thick, mask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
            k += 1
        if ep == 0 or ep % 5 == 4 or ep == a.epochs - 1:
            mt, rows = evaluate(model, lva, dev, px, geometry=a.geometry)
            hist.append(dict(epoch=ep + 1, train_loss=tot / max(k, 1), **mt))
            print(f"ep{ep + 1:3d} loss={tot / max(k, 1):7.4f}  "
                  f"depth AbsRel={mt['depth_absrel']:.4f}  "
                  f"thick AbsRel={mt['thick_absrel']:.4f}  "
                  f"volume AbsRel model={mt.get('volume_absrel_model', float('nan')):.3f} "
                  f"vs 2D-only={mt.get('volume_absrel_2d_only', float('nan')):.3f} "
                  f"(n={mt['n_instances']})", flush=True)
            for k, v in (mt.get("by_occlusion") or {}).items():
                print(f"        {k}: n={v['n']:4d} model={v['model']:.3f} "
                      f"2D-only={v['two_d'] if v['two_d'] is None else round(v['two_d'], 3)}",
                      flush=True)
            for px_key, per in (mt.get("by_scale_class") or {}).items():
                print(f"        {px_key}", flush=True)
                for org, v in per.items():
                    two = "n/a" if v["two_d"] is None else f"{v['two_d']:.3f}"
                    win = "" if v["two_d"] is None else (
                        "  WIN" if v["model"] < v["two_d"] else "  lose")
                    print(f"          {org:9s} n={v['n']:4d} model={v['model']:.3f} "
                          f"2D-only={two}{win}", flush=True)

    out = Path(a.out) if a.out else Path(a.root) / f"train_{a.channels}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "history.json").write_text(json.dumps(hist, indent=1))
    mt, rows = evaluate(model, lva, dev, px, geometry=a.geometry)
    (out / "instances_eval.json").write_text(json.dumps(rows, indent=1))
    print(f"saved -> {out}", flush=True)


if __name__ == "__main__":
    main()
