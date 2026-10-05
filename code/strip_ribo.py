import glob, json, os, shutil, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEP = f"{ROOT}/release/blockface"
OUT = os.environ.get("RIBO_OUT", f"{ROOT}/staging/ribo_removed")
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def main():
    boxes = sorted(json.load(open(os.environ["RIBO_BOXES"])))
    if LIMIT:
        boxes = boxes[:LIMIT]
    skip_plane = set()
    if os.environ.get("DEPOSIT_EXCLUSIONS"):
        skip_plane = set(json.load(open(os.environ["DEPOSIT_EXCLUSIONS"]))["planes"])
    print(f"{len(boxes)} boxes -> {OUT}", flush=True)

    n_plane = n_touched = 0
    px_cleared = thick_changed = 0

    for nb, box in enumerate(boxes):
        raw = f"{ROOT}/raw/{box}.npz"
        if not os.path.exists(raw):
            print(f"  no raw for {box}"); continue
        with np.load(raw, allow_pickle=True) as d:
            cls3 = d["cls"][:]
            meta3 = json.loads(str(d["meta"]))
        clist3 = meta3.get("classes") or []
        if "ribo" not in clist3:
            continue
        ribo3 = clist3.index("ribo") + 1
        Z3, H3, W3 = cls3.shape
        ribo_vox = (cls3 == ribo3)

        od = f"{OUT}/{box}/faces"
        os.makedirs(od, exist_ok=True)
        for fp in sorted(glob.glob(f"{DEP}/{box}/faces/*.npz")):
            name = os.path.basename(fp)
            if name[:-4] in skip_plane or name.startswith("."):
                continue
            z = np.load(fp, allow_pickle=False)
            a = {k: z[k] for k in z.files if k != "meta"}
            m = json.loads(str(z["meta"]))
            cl = m.get("classes") or []
            n_plane += 1
            if "ribo" not in cl:
                shutil.copyfile(fp, f"{od}/{name}")
                continue
            k = cl.index("ribo")
            idx = k + 1
            cf = a["cls_face"]
            sel = cf == idx

            for key in ("inst_face", "mask", "clipped",
                        "depth_below_steps", "own_occupancy_below_steps"):
                if key in a:
                    a[key] = np.where(sel, 0, a[key]).astype(a[key].dtype)
            a["cls_face"] = np.where(sel, 0, np.where(cf > idx, cf - 1, cf)).astype(cf.dtype)

            zi = int(name.rsplit("_z", 1)[1].split(".")[0])
            th = a["thickness_below_steps"]
            H, W = th.shape
            y0 = (H3 - H) // 2 if H3 >= H else 0
            x0 = (W3 - W) // 2 if W3 >= W else 0
            if zi < Z3:
                sub = ribo_vox[zi:, y0:y0 + H, x0:x0 + W]
                if sub.shape[1:] == th.shape:
                    drop = sub.sum(axis=0).astype(th.dtype)
                    thick_changed += int((drop > 0).sum())
                    a["thickness_below_steps"] = np.maximum(
                        th.astype(np.int64) - drop, 0).astype(th.dtype)
            a["thickness_below_steps"] = np.where(
                a["mask"] > 0, a["thickness_below_steps"], 0).astype(th.dtype)

            m["classes"] = [c for c in cl if c != "ribo"]
            inst = m.get("instances") or {}
            m["instances"] = {i: v for i, v in inst.items()
                              if (v.get("organelle") if isinstance(v, dict) else v) != "ribo"}
            m["n_instances"] = len(m["instances"])
            if "exposed_fraction" in m:
                m["exposed_fraction"] = float((a["mask"] > 0).mean())
            px_cleared += int(sel.sum())
            n_touched += 1
            np.savez_compressed(f"{od}/{name}", meta=json.dumps(m), **a)
        if (nb + 1) % 20 == 0:
            print(f"  {nb + 1}/{len(boxes)}  {n_touched} planes rewritten", flush=True)

    print(f"\nplanes seen {n_plane}, rewritten {n_touched}")
    print(f"ribo pixels cleared {px_cleared:,}")
    print(f"pixels whose thickness changed {thick_changed:,}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
