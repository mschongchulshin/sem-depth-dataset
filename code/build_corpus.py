import argparse
import json
import zlib
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import (FIRST_COMPLETED, ProcessPoolExecutor,
                                wait as fut_wait)
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("SEM_DEPTH_ROOT", Path(__file__).resolve().parent.parent))
CODE = ROOT / "code"

STAGE = Path("/private/tmp/em-depth-stage")
MOVE_LOCK = STAGE / ".move.lock"

MAX_LOAD = 40.0
NICE = 10


def load1():
    return os.getloadavg()[0]


def wait_for_load(limit=MAX_LOAD, poll=15.0, max_wait=1800.0):
    waited = 0.0
    while load1() > limit and waited < max_wait:
        time.sleep(poll)
        waited += poll
    return load1()

CELLS = [
    ("jrc_hela-2", "HeLa"),
    ("jrc_hela-3", "HeLa"),
    ("jrc_hela-1", "HeLa"),
    ("jrc_hela-bfa", "HeLa + brefeldin A"),
    ("jrc_jurkat-1", "Jurkat T"),
    ("jrc_macrophage-2", "macrophage"),
    ("aic_desmosome-2", "keratinocyte"),
    ("aic_desmosome-3", "keratinocyte"),
    ("jrc_cos7-11", "COS-7 monkey"),
    ("jrc_mus-liver", "mouse hepatocyte"),
    ("jrc_choroid-plexus-2", "choroid plexus"),
]

TISSUE = [
    ("jrc_mus-heart-1", "mouse heart"),
    ("jrc_mus-hippocampus-1", "mouse hippocampus"),
    ("jrc_mus-kidney-3", "mouse kidney"),
    ("jrc_mus-liver-3", "mouse liver"),
    ("jrc_mus-pancreas-4", "mouse pancreas"),
    ("jrc_mus-skin-1", "mouse skin"),
    ("jrc_mus-thymus-1", "mouse thymus"),
]
TISSUE_FINE = [
    ("jrc_dauer-larva", "C. elegans dauer"),
]

TIERS = {
    "cell": (64.0, [384, 384, 384],
             ["nucleus", "ne", "chrom", "nucleolus", "pm", "mito", "er"], CELLS),
    "wide": (32.0, [448, 448, 448],
             ["nucleus", "ne", "chrom", "nucleolus", "pm", "mito", "er", "golgi"], CELLS),
    "coarse": (16.0, [512, 512, 512],
               ["mito", "er", "golgi", "lyso", "ld", "endo", "vesicle", "eres",
                "nucleolus", "ne", "chrom", "pm"], CELLS),
    "mid": (8.0, [448, 448, 448],
            ["mito", "er", "golgi", "lyso", "ld", "endo", "vesicle", "eres",
             "nucleolus", "ne", "pm"], CELLS),
    "fine": (4.0, [448, 448, 448],
             ["lyso", "ld", "endo", "vesicle", "eres", "nucleolus"], CELLS),
    "tissue": (128.0, [320, 320, 320], ["nucleus"], TISSUE),
    "tissue_fine": (64.0, [320, 320, 320], ["nucleus"], TISSUE_FINE),
}

LEVEL_CACHE = ROOT / "cache" / "levels.json"


def pyramid_levels(dataset, organelle):
    cache = json.loads(LEVEL_CACHE.read_text()) if LEVEL_CACHE.exists() else {}
    key = f"{dataset}/{organelle}"
    if key in cache:
        return {k: float(v) for k, v in cache[key].items()}
    from n5 import N5Array
    out = {}
    for lvl in ["s0", "s1", "s2", "s3", "s4", "s5"]:
        try:
            a = N5Array(seg_url(dataset, organelle), lvl)
            v = a.voxel_size_nm()
            out[lvl] = float(v[2])
        except Exception:
            break
    cache[key] = out
    LEVEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    LEVEL_CACHE.write_text(json.dumps(cache, indent=1))
    return out


def level_for_nm(dataset, organelle, target_nm, tol=1.6):
    lv = pyramid_levels(dataset, organelle)
    if not lv:
        return None, None
    best = min(lv.items(), key=lambda kv: abs(np.log2(kv[1] / target_nm)))
    if abs(np.log2(best[1] / target_nm)) > np.log2(tol):
        return None, best[1]
    return best[0], best[1]


def seg_url(dataset, name):
    return ("https://janelia-cosem-datasets.s3.amazonaws.com/"
            f"{dataset}/{dataset}.n5/labels/{name}_seg")


def dataset_classes(dataset, wanted):
    avail = ROOT / "cache" / "availability.json"
    if avail.exists():
        m = json.loads(avail.read_text()).get(dataset, {})
        keep = [c for c in wanted if m.get(c, {}).get("ok")]
        if keep:
            return keep
    survey = ROOT / "cache" / "survey.json"
    if not survey.exists():
        return wanted
    rows = {r["dataset"]: {s.replace("_seg", "") for s in r["segs"]}
            for r in json.loads(survey.read_text())}
    have = rows.get(dataset, set())
    keep = [c for c in wanted if c in have]
    if keep:
        return keep
    for c in wanted:
        alt = [h for h in have if h.endswith(c) or c in h]
        if alt:
            return [sorted(alt, key=len)[0]]
    return sorted(have)[:1] if have else wanted[:1]


def spread_starts(n_boxes, box, shape_hint, seed):
    z, y, x = shape_hint
    phi = 0.618033988749895
    off = ((abs(seed) % 10_000) / 10_000.0)
    out = []
    for k in range(n_boxes):
        fz = ((k + 1) * phi + off) % 1.0
        fx = ((k + 1) * phi * phi + off * 0.5) % 1.0
        out.append((int(fz * max(z - box[0], 1)), 0, int(fx * max(x - box[2], 1))))
    return out


SHAPE_HINT = {"s0": (6368, 1600, 12000), "s1": (3184, 800, 6000), "s2": (1592, 400, 3000)}


CELL_INDEX = ROOT / "cache" / "cells.json"


def _cell_starts(dataset, n_boxes, box, level_vox_nm, shape, seed):
    if not CELL_INDEX.exists():
        return None
    rec = json.loads(CELL_INDEX.read_text()).get(dataset)
    if not rec or not rec.get("cells"):
        return None
    cells = rec["cells"]
    vox = np.array(level_vox_nm, float)
    shape = np.array(shape, int)
    box = np.array(box, int)
    rng = np.random.default_rng(seed)
    starts, ids, taken = [], [], set()
    for k in range(n_boxes):
        c = cells[k % len(cells)]
        centre = np.array(c["centroid_nm"], float) / vox
        jitter = 0.0 if k < len(cells) else 0.35
        off = rng.uniform(-jitter, jitter, 3) * box
        start = np.round(centre - box / 2.0 + off).astype(int)
        start = np.clip(start, 0, np.maximum(shape - box, 0))
        t = tuple(int(v) for v in start)
        if t in taken:
            continue
        taken.add(t)
        starts.append(t)
        ids.append(int(c["id"]))
    return starts, ids


def make_plan(tiers, boxes, views, faces=16, sections=24):
    from n5 import N5Array
    plan, skipped = [], []
    for tier in tiers:
        target_nm, box, classes, sources = TIERS[tier]
        sources = CELLS[:sources] if isinstance(sources, int) else sources
        for ds, label in sources:
            cl = dataset_classes(ds, classes)
            level, got_nm = level_for_nm(ds, cl[0], target_nm)
            if level is None:
                skipped.append((tier, ds, got_nm))
                continue
            try:
                shape = N5Array(seg_url(ds, cl[0]), level).shape
            except Exception:
                skipped.append((tier, ds, None))
                continue
            seed = zlib.crc32(f"{ds}{tier}".encode())
            arr = N5Array(seg_url(ds, cl[0]), level)
            byc = _cell_starts(ds, boxes, box, arr.voxel_size_nm(), shape, seed)
            if byc is None:
                starts, cell_ids = spread_starts(boxes, box, shape, seed=seed), [None] * boxes
                how = "positional"
            else:
                starts, cell_ids = byc
                how = "per-cell"
            for b, (start, cid) in enumerate(zip(starts, cell_ids)):
                plan.append(dict(tier=tier, dataset=ds, label=label, level=level,
                                 voxel_nm=got_nm, box=box, classes=cl, start=list(start),
                                 views=views, name=f"{ds}__{tier}_b{b:03d}",
                                 cell_id=cid, sampling=how, faces=faces,
                                 sections=sections))
    groups = {}
    for e in plan:
        groups.setdefault((e["tier"], e["dataset"]), []).append(e)
    order, keys = [], list(groups)
    while any(groups[k] for k in keys):
        for k in keys:
            if groups[k]:
                order.append(groups[k].pop(0))
    plan = order

    bycell = sum(1 for e in plan if e.get("sampling") == "per-cell")
    ncell = len({(e["dataset"], e["cell_id"]) for e in plan if e.get("cell_id") is not None})
    print(f"  sampling: {bycell}/{len(plan)} boxes placed on identified cells, "
          f"{ncell} distinct cells", flush=True)

    for tier, ds, nm in skipped:
        print(f"  skip {ds} from tier {tier}: finest voxel is "
              + (f"{nm:.1f} nm" if nm else "unreadable"), flush=True)
    return plan


def _spawn(cmd, out, env=None, cwd=None):
    proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    try:
        return proc.wait()
    except BaseException:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        raise


def run(cmd, log, tag):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ [{tag}] " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=CODE)


def _raw_matches(raw, p):
    try:
        with np.load(raw, allow_pickle=False) as d:
            m = json.loads(str(d["meta"]))
            if "em" not in d.files:
                return False
            if d["em"].shape != d["inst"].shape:
                return False
            if m.get("em_norm") != "interior-p0.2-p99.8":
                e = d["em"][::4, ::4, ::4].astype(np.float32)
                interior = e[(e > e.min()) & (e < e.max())]
                ref = interior if interior.size > 1000 else e
                lo, hi = np.percentile(ref, [0.2, 99.8])
                if (hi - lo) < 200.0:
                    return False
    except Exception:
        return False
    if list(m.get("box_start_zyx", [])) != list(p["start"]):
        return False
    if m.get("level") != p["level"]:
        return False
    return set(p["classes"]) <= set(m.get("classes", []))


def do_fetch(p):
    raw = ROOT / "raw" / f"{p['name']}.npz"
    if raw.exists():
        if _raw_matches(raw, p):
            return p["name"], "cached", 0
        stale = ROOT / "raw_stale" / f"{p['name']}.npz"
        stale.parent.mkdir(parents=True, exist_ok=True)
        raw.replace(stale)
        rc = _refetch(p)
        if raw.exists():
            stale.unlink(missing_ok=True)
            return p["name"], "ok", rc
        return p["name"], "refetch-failed", rc
    rc = _refetch(p)
    return p["name"], ("ok" if raw.exists() else "failed"), rc


def _refetch(p):
    return run([sys.executable, "-u", "capped.py", "fetch_wholecell.py",
              "--dataset", p["dataset"], "--organelles", *p["classes"],
              "--level", p["level"], "--table-level", "s4", "--fine-level", p["level"],
              "--box", *map(str, p["box"]), "--start", *map(str, p["start"]),
               "--name", p["name"]], ROOT / "corpus" / f"{p['name']}.log", "fetch")


def _acquire_move_lock(timeout=1800):
    STAGE.mkdir(parents=True, exist_ok=True)
    waited = 0.0
    while True:
        try:
            fd = os.open(MOVE_LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            if waited > timeout:
                try:
                    os.unlink(MOVE_LOCK)
                except OSError:
                    pass
                waited = 0.0
            time.sleep(0.5)
            waited += 0.5


def _release_move_lock():
    try:
        os.unlink(MOVE_LOCK)
    except OSError:
        pass


def external_mounted():
    return ROOT.is_dir() and (ROOT / "code").is_dir()


def do_render(p):
    raw = ROOT / "raw" / f"{p['name']}.npz"
    final = ROOT / "corpus" / p["name"]
    if not external_mounted():
        return p["name"], "volume-gone", 1
    if not raw.exists():
        return p["name"], "no-raw", 1
    idx = final / "index.json"
    if idx.exists():
        if idx.stat().st_mtime >= raw.stat().st_mtime:
            return p["name"], "cached", 0
        shutil.rmtree(final, ignore_errors=True)

    shutil.rmtree(final, ignore_errors=True)
    (final / "views").mkdir(parents=True, exist_ok=True)
    (final / "preview").mkdir(parents=True, exist_ok=True)

    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    wait_for_load()
    log = ROOT / "corpus" / f"{p['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ [render] {p['name']} {p['views']} views -> {final}\n")
        f.flush()
        rc = _spawn(
            ["nice", "-n", str(NICE), sys.executable, "-u", "capped.py",
             "render_organelles.py",
             "--npz", str(raw), "--out", str(final), "--views", str(p["views"]),
             "--crop-size", "320", "--domain-random",
             "--min-valid", "0.08", "--min-usable", "1"],
            out=f, cwd=CODE, env=env)

    if not (final / "index.json").exists():
        n = len([q for q in (final / "views").glob("*.npz") if not q.name.startswith("._")])
        return p["name"], f"incomplete({n} views kept)", rc
    return p["name"], "ok", rc


def do_blockface(p):
    raw = ROOT / "raw" / f"{p['name']}.npz"
    final = ROOT / "blockface" / p["name"]
    if not external_mounted():
        return p["name"], "volume-gone", 1
    if not raw.exists():
        return p["name"], "no-raw", 1
    try:
        with np.load(raw, allow_pickle=False) as _d:
            if "em" not in _d.files:
                return p["name"], "no-image", 0
    except Exception:
        return p["name"], "unreadable-raw", 1
    idx = final / "index.json"
    if idx.exists() and idx.stat().st_mtime >= raw.stat().st_mtime:
        return p["name"], "cached", 0

    (final / "faces").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    log = ROOT / "blockface" / f"{p['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ [blockface] {p['name']}\n")
        f.flush()
        rc = _spawn(
            ["nice", "-n", str(NICE), sys.executable, "-u", "capped.py", "make_blockface.py",
             "--npz", str(raw), "--out", str(final),
             "--slices", str(p.get("faces", 16)), "--crop-size", "512",
             "--margin", "0.25", "--png"],
            out=f, cwd=CODE, env=env)
    return p["name"], ("ok" if (final / "index.json").exists() else "failed"), rc


def do_thinsection(p):
    raw = ROOT / "raw" / f"{p['name']}.npz"
    final = ROOT / "thinsection" / p["name"]
    if not external_mounted():
        return p["name"], "volume-gone", 1
    if not raw.exists():
        return p["name"], "no-raw", 1
    try:
        with np.load(raw, allow_pickle=False) as _d:
            if "em" not in _d.files:
                return p["name"], "no-image", 0
    except Exception:
        return p["name"], "unreadable-raw", 1
    idx = final / "index.json"
    if idx.exists() and idx.stat().st_mtime >= raw.stat().st_mtime:
        return p["name"], "cached", 0
    (final / "sections").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    log = ROOT / "thinsection" / f"{p['name']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        f.write(f"\n$ [thinsection] {p['name']}\n")
        f.flush()
        rc = _spawn(
            ["nice", "-n", str(NICE), sys.executable, "-u", "capped.py", "make_thinsection.py",
             "--npz", str(raw), "--out", str(final),
             "--sections", str(p.get("sections", 24)),
             "--thickness-nm", "70", "--crop-size", "512", "--png"],
            out=f, cwd=CODE, env=env)
    return p["name"], ("ok" if (final / "index.json").exists() else "failed"), rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="+",
                    default=["cell", "wide", "coarse", "mid", "fine",
                             "tissue", "tissue_fine"])
    ap.add_argument("--boxes", type=int, default=20, help="boxes per cell per tier")
    ap.add_argument("--views", type=int, default=8, help="views per box")
    ap.add_argument("--fetch-workers", type=int, default=4, help="network bound")
    ap.add_argument("--render-workers", type=int, default=6,
                    help="CPU bound; far below the core count on purpose, see MAX_LOAD note")
    ap.add_argument("--blockface-workers", type=int, default=4,
                    help="block-face cutting is IO bound and cheap, it needs few workers")
    ap.add_argument("--faces", type=int, default=16, help="block-face images per box")
    ap.add_argument("--sections", type=int, default=24, help="thin sections per box")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] planning: tiers={a.tiers} boxes={a.boxes} "
          f"views={a.views} faces={a.faces} sections={a.sections}", flush=True)
    plan = make_plan(a.tiers, a.boxes, a.views, a.faces, a.sections)
    print(f"[{time.strftime('%H:%M:%S')}] planning done", flush=True)
    n_views = sum(p["views"] for p in plan)
    cells = {p["dataset"] for p in plan}
    print(f"plan: {len(plan)} boxes, {len(cells)} cells, {len(a.tiers)} tiers, "
          f"{n_views} views, ~{n_views * 2.1 / 1024:.1f} GB", flush=True)
    for tier in a.tiers:
        sub = [p for p in plan if p["tier"] == tier]
        if not sub:
            continue
        nm = sub[0]["voxel_nm"]
        print(f"  {tier:7s} {len(sub):4d} boxes  {sum(p['views'] for p in sub):5d} views  "
              f"{nm:.0f} nm/voxel  extent={sub[0]['box'][2] * nm / 1000:.1f} um  "
              f"cells={sorted({p['dataset'] for p in sub})}", flush=True)
    print(f"  est wall clock: fetch ~{len(plan) * 2 / a.fetch_workers:.0f} min, "
          f"render ~{n_views / a.render_workers:.0f} min", flush=True)
    if a.dry_run:
        (ROOT / "corpus").mkdir(exist_ok=True)
        (ROOT / "corpus" / "plan_dryrun.json").write_text(json.dumps(plan, indent=1))
        print(f"  plan written to corpus/plan_dryrun.json", flush=True)
        return

    (ROOT / "corpus").mkdir(exist_ok=True)
    (ROOT / "corpus" / "plan.json").write_text(json.dumps(plan, indent=1))

    print(f"\n--- pipelined: fetch x{a.fetch_workers} feeding render x{a.render_workers} ---",
          flush=True)
    done = failed = 0
    n = len(plan)
    with ProcessPoolExecutor(max_workers=a.fetch_workers) as fex, \
         ProcessPoolExecutor(max_workers=a.render_workers) as rex, \
         ProcessPoolExecutor(max_workers=a.blockface_workers) as bex:
        import queue as _queue
        events = _queue.Queue()
        window = max(a.fetch_workers * 4, 16)
        pending = list(plan)
        fetch_futs, render_futs, bf_futs = {}, {}, {}
        bf_done = bf_failed = 0

        def submit(pool, fn, pl, book):
            fut = pool.submit(fn, pl)
            book[fut] = pl
            fut.add_done_callback(events.put)
            return fut

        def top_up():
            while pending and len(fetch_futs) < window:
                submit(fex, do_fetch, pending.pop(0), fetch_futs)

        top_up()
        while pending or fetch_futs or render_futs or bf_futs:
            try:
                fut = events.get(timeout=120)
            except _queue.Empty:
                continue

            if fut in fetch_futs:
                pl = fetch_futs.pop(fut)
                name, status, rc = fut.result()
                if status in ("ok", "cached"):
                    submit(bex, do_blockface, pl, bf_futs)
                    submit(bex, do_thinsection, pl, bf_futs)
                    submit(rex, do_render, pl, render_futs)
                else:
                    failed += 1
                    print(f"  fetch {name}: {status} rc={rc}", flush=True)
                top_up()

            elif fut in bf_futs:
                bf_futs.pop(fut)
                name, status, rc = fut.result()
                if status in ("ok", "cached"):
                    bf_done += 1
                    if bf_done % 200 == 0:
                        print(f"  [cut {bf_done}] {name}", flush=True)
                elif status != "no-image":
                    bf_failed += 1
                    print(f"  cut {name}: {status} rc={rc}", flush=True)

            elif fut in render_futs:
                render_futs.pop(fut)
                name, status, rc = fut.result()
                if status in ("ok", "cached"):
                    done += 1
                else:
                    failed += 1
                if done % 25 == 0 or status not in ("ok", "cached"):
                    print(f"  [{done + failed}/{n}] render {name}: {status}"
                          + (f" rc={rc}" if rc else ""), flush=True)

    print(f"\ncorpus build done: {done} rendered, {failed} failed, of {n} boxes", flush=True)
    print(f"cut subsets (block-face + thin section): {bf_done} jobs done, "
          f"{bf_failed} failed", flush=True)


if __name__ == "__main__":
    main()
