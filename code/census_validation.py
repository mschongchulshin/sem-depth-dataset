import collections, glob, json, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def excluded():
    p = os.environ.get("DEPOSIT_EXCLUSIONS")
    if not p:
        return set(), set()
    e = json.load(open(p))
    return set(e["boxes"]), set(e["planes"])


def faces_dir(box):
    ov = os.environ.get("CENSUS_OVERLAY")
    if ov:
        d = f"{ov}/{box}/faces"
        if os.path.isdir(d):
            return d
    return f"{ROOT}/release/blockface/{box}/faces"


def main():
    skip_box, skip_plane = excluded()
    if os.environ.get("CENSUS_OVERLAY"):
        n = len([d for d in os.listdir(os.environ["CENSUS_OVERLAY"]) if not d.startswith(".")])
        print(f"overlay covers {n} boxes", flush=True)
    raws = sorted(p for p in glob.glob(f"{ROOT}/raw/*.npz")
                  if os.path.basename(p)[:-4] not in skip_box)
    if LIMIT:
        raws = raws[:LIMIT]
    print(f"{len(raws)} boxes, excluding {len(skip_box)}", flush=True)

    C = collections.Counter()
    by_class_clip = collections.defaultdict(lambda: [0, 0])
    ident, ident_clip = [], []
    fail_ident, fail_thick = [], []
    ident_by_class = collections.defaultdict(list)
    thick = []
    sphere = collections.defaultdict(list)
    max_depth = 0
    t0 = time.time()

    for n, p in enumerate(raws):
        if n and n % 50 == 0:
            el = time.time() - t0
            print(f"  {n}/{len(raws)}  {el/60:.1f} min  "
                  f"eta {el/n*(len(raws)-n)/60:.0f} min", flush=True)
        box = os.path.basename(p)[:-4]
        faces = sorted(f for f in glob.glob(f"{faces_dir(box)}/*.npz")
                       if "/._" not in f and os.path.basename(f)[:-4] not in skip_plane)
        if not faces:
            continue
        try:
            with np.load(p, allow_pickle=True) as d:
                if "inst" not in d.files:
                    continue
                inst3 = d["inst"]
                cls3 = d["cls"] if "cls" in d.files else None
                meta3 = json.loads(str(d["meta"]))
        except Exception:
            C["unreadable box"] += 1
            continue
        if inst3.ndim != 3:
            continue
        C["boxes"] += 1
        names = {int(k): v.get("organelle") for k, v in meta3.get("instances", {}).items()}
        trunc = {int(k): bool(v.get("truncated")) for k, v in meta3.get("instances", {}).items()}
        drop_ids = {i for i, o in names.items() if o == "ribo"}
        if drop_ids:
            inst3 = np.where(np.isin(inst3, list(drop_ids)), 0, inst3)
            names = {i: o for i, o in names.items() if o != "ribo"}
            trunc = {i: t for i, t in trunc.items() if i not in drop_ids}
        Z3, H3, W3 = inst3.shape

        walls = np.concatenate([np.unique(inst3[0]), np.unique(inst3[-1]),
                                np.unique(inst3[:, 0]), np.unique(inst3[:, -1]),
                                np.unique(inst3[:, :, 0]), np.unique(inst3[:, :, -1])])
        wall_ids = set(int(v) for v in np.unique(walls) if v)
        present = set(int(v) for v in np.unique(inst3) if v)
        for i, t in trunc.items():
            if t or i not in present:
                continue
            C["unbiased instances"] += 1
            if i in wall_ids:
                C["unbiased touching a wall"] += 1

        clist = meta3.get("classes") or []
        if cls3 is not None and clist:
            fgv = inst3 > 0
            iv = inst3[fgv]; cv = cls3[fgv].astype(np.int64)
            uid, inv = np.unique(iv, return_inverse=True)
            nc = int(cv.max()) + 1
            tally = np.bincount(inv * nc + cv, minlength=len(uid) * nc).reshape(len(uid), nc)
            tally[:, 0] = 0
            modal = tally.argmax(axis=1)
            for k, i in enumerate(uid):
                o = names.get(int(i))
                v = int(modal[k]) - 1
                if not o or o not in clist or not 0 <= v < len(clist):
                    continue
                C["instances named"] += 1
                if clist[v] != o:
                    C["name disagrees"] += 1

        for fp in faces:
            try:
                z = np.load(fp, allow_pickle=False)
                dep = z["depth_below_steps"]; own = z["own_occupancy_below_steps"]
                thk = z["thickness_below_steps"]; msk = z["mask"]
                ef = z["inst_face"]; clp = z["clipped"]
                fm = json.loads(str(z["meta"]))
            except Exception:
                C["unreadable plane"] += 1
                continue
            C["planes"] += 1
            zi = int(os.path.basename(fp).rsplit("_z", 1)[1].split(".")[0])
            px = float(fm.get("pixel_size_nm", 0) or 0)
            zs = float(fm.get("z_step_nm", 0) or 0)
            H, W = dep.shape
            y0 = (H3 - H) // 2 if H3 >= H else 0
            x0 = (W3 - W) // 2 if W3 >= W else 0

            fg = (msk > 0) & (ef > 0)
            nfg = int(fg.sum())
            C["foreground pixels"] += nfg
            if not nfg:
                continue
            max_depth = max(max_depth, int(dep.max()))

            C["depth > own"] += int((dep[fg] > own[fg]).sum())
            C["own > thickness"] += int((own[fg] > thk[fg]).sum())

            cf = ef[fg]
            for i in np.unique(cf):
                o = names.get(int(i))
                if not o:
                    continue
                sl = fg & (ef == i)
                by_class_clip[o][1] += int(sl.sum())
                by_class_clip[o][0] += int((clp & sl).sum())

            if zi >= Z3 or px <= 0 or zs <= 0:
                continue
            sub = inst3[zi:, y0:y0 + H, x0:x0 + W]
            if sub.shape[1:] != dep.shape:
                continue

            match = sub == ef[None, :, :]

            reach = match.all(axis=0) & fg
            c = clp & fg
            C["clip px"] += nfg
            C["clip flagged not reaching"] += int((c & ~reach).sum())
            C["clip reaching not flagged"] += int((~c & reach).sum())

            run = np.zeros(dep.shape, np.int32)
            live = fg.copy()
            for k in range(match.shape[0]):
                live &= match[k]
                if not live.any():
                    break
                run += live
            C["recompute px"] += nfg
            C["recompute mismatch"] += int((np.abs(run[fg] - dep[fg]) > 0).sum())

            lab = (sub > 0) & fg[None, :, :]
            lhs = float(thk[fg].sum()) * px * px * zs
            rhs = float(lab.sum()) * px * px * zs
            if rhs > 0:
                r = lhs / rhs
                thick.append(r)
                if abs(r - 1.0) >= 1e-6 and len(fail_thick) < 400:
                    fail_thick.append([box, zi, round(r, 6)])

            depth_col = match.sum(axis=0)
            for i in np.unique(cf):
                sel = fg & (ef == i)
                if sel.sum() < 40:
                    continue
                swept = float(own[sel].sum()) * px * px * zs
                under = float(depth_col[sel].sum()) * px * px * zs
                if under <= 0:
                    continue
                r = swept / under
                C["identity instances"] += 1
                if abs(r - 1.0) < 1e-6:
                    C["identity exact"] += 1
                elif len(fail_ident) < 400:
                    fail_ident.append([box, zi, int(i), names.get(int(i)), round(r, 6)])
                (ident_clip if clp[sel].any() else ident).append(r)
                o = names.get(int(i))
                if o and not clp[sel].any():
                    ident_by_class[o].append(r)
                    rec = meta3["instances"].get(str(i)) or meta3["instances"].get(i)
                    v = float(rec.get("volume_nm3") or 0) if rec else 0.0
                    if v > 0:
                        d_eq = (6.0 * v / np.pi) ** (1.0 / 3.0)
                        sphere[o].append((float(dep[sel].sum()) * zs, float(sel.sum()) * d_eq))

    def pct(a, b):
        return 100.0 * a / b if b else float("nan")

    out = dict(
        boxes=C["boxes"], planes=C["planes"], foreground_pixels=C["foreground pixels"],
        ordering=dict(pixels=C["foreground pixels"], depth_gt_own=C["depth > own"],
                      own_gt_thickness=C["own > thickness"]),
        clipped_flag=dict(pixels=C["clip px"],
                          flagged_not_reaching=C["clip flagged not reaching"],
                          reaching_not_flagged=C["clip reaching not flagged"],
                          agreement_pct=100.0 - pct(C["clip flagged not reaching"]
                                                    + C["clip reaching not flagged"],
                                                    C["clip px"])),
        recompute=dict(pixels=C["recompute px"], mismatch=C["recompute mismatch"]),
        identity=dict(instances=C["identity instances"], exact=C["identity exact"],
                      exact_pct=pct(C["identity exact"], C["identity instances"]),
                      median=float(np.median(ident)) if ident else None,
                      within_1e6_pct=pct(sum(abs(x - 1) < 1e-6 for x in ident), len(ident)),
                      clipped_n=len(ident_clip)),
        thickness=dict(planes=len(thick),
                       median=float(np.median(thick)) if thick else None,
                       within_1e6_pct=pct(sum(abs(x - 1) < 1e-6 for x in thick), len(thick))),
        truncation=dict(unbiased=C["unbiased instances"],
                        touching_wall=C["unbiased touching a wall"]),
        naming=dict(instances=C["instances named"], disagree=C["name disagrees"],
                    agree_pct=100.0 - pct(C["name disagrees"], C["instances named"])),
        clipped_by_class={k: round(pct(v[0], v[1]), 4) for k, v in by_class_clip.items()},
        sphere_cut={k: dict(n=len(v), ratio=round(sum(a for a, _ in v) / sum(b for _, b in v), 4))
                    for k, v in sphere.items() if sum(b for _, b in v) > 0},
        max_depth_steps=max_depth,
        unreadable=dict(boxes=C["unreadable box"], planes=C["unreadable plane"]),
        failures=dict(identity=fail_ident, thickness=fail_thick),
        _census=True, _minutes=round((time.time() - t0) / 60, 1),
    )
    dst = os.environ.get("CENSUS_OUT", f"{ROOT}/cache/census_validation.json")
    json.dump(out, open(dst, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("clipped_by_class", "sphere_cut")}, indent=1))
    print(f"-> {dst}")


if __name__ == "__main__":
    main()
