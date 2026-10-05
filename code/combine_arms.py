import argparse, importlib.util, itertools, json, os, sys, time, traceback
import numpy as np

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARMS = f"{ROOT}/code/arms"
sys.path.insert(0, f"{ROOT}/autoresearch")
from harness import Data, metrics
from sweep_arms import load_arm, run_arm


class Combined:

    def __init__(self, backbone, head):
        self.backbone, self.head = backbone, head
        self.NAME = f"{backbone.NAME}+{head.NAME}"
        self.PAPER = f"{backbone.PAPER}  ||  {head.PAPER}"
        self.REPO = f"{backbone.REPO} || {head.REPO}"
        self.IDEA = "backbone from the first arm, output head and loss from the second"
        self.OUT = getattr(head, "OUT", 1)
        if hasattr(head, "LOSS"):
            self.LOSS = head.LOSS
        if hasattr(head, "DECODE"):
            self.DECODE = head.DECODE

    def build(self, cin, **kw):
        return self.backbone.build(cin, out=self.OUT, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbones", default="unireplk_lka,segformer_mlp,convnext_unet")
    ap.add_argument("--heads", default="head_lognormal_mdn,head_hazard_survival,head_conor")
    ap.add_argument("--folds", default="jrc_mus-thymus-1,jrc_mus-pancreas-4,jrc_macrophage-2")
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--out", default="cache/combine_arms.json")
    a = ap.parse_args()

    import torch
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    data = Data(silhouette=True)
    folds = [f.strip() for f in a.folds.split(",") if f.strip()]
    bs = [b.strip() for b in a.backbones.split(",") if b.strip()]
    hs = [h.strip() for h in a.heads.split(",") if h.strip()]
    print(f"device {dev}   {len(bs)} backbones x {len(hs)} heads   "
          f"{len(folds)} folds   {a.budget}s per fold\n", flush=True)

    outpath = f"{ROOT}/{a.out}"
    results = json.load(open(outpath)) if os.path.exists(outpath) else {}
    for b, h in itertools.product(bs, hs):
        slug = f"{b}+{h}"
        if slug in results and results[slug].get("ok"):
            print(f"{slug:44s} already done, d1 {results[slug]['d1']:.4f}", flush=True)
            continue
        print(slug, flush=True)
        rec = {}
        try:
            arm = Combined(load_arm(f"{ARMS}/{b}.py"), load_arm(f"{ARMS}/{h}.py"))
            per = run_arm(arm, data, folds, a.budget, dev)
            rec["per_fold"] = per
            rec["d1"] = float(np.mean([v["d1"] for v in per.values()]))
            rec["exact"] = float(np.mean([v["exact"] for v in per.values()]))
            rec["medrel"] = float(np.mean([v["medrel"] for v in per.values()]))
            rec["params_m"] = list(per.values())[0]["params_m"]
            rec["ok"] = True
            print(f"  -> d1 {rec['d1']:.4f}\n", flush=True)
        except Exception:
            rec["ok"] = False
            rec["error"] = traceback.format_exc()[-1200:]
            print(f"  -> FAILED\n{rec['error'][-350:]}\n", flush=True)
        results[slug] = rec
        json.dump(results, open(outpath, "w"), indent=1)

    ok = {k: v for k, v in results.items() if v.get("ok")}
    print(f"\n{'=' * 74}\n{len(ok)} combinations scored, {len(results) - len(ok)} failed")
    print(f"{'combination':46s}{'d1':>8s}{'exact':>8s}{'MedRel':>9s}")
    for k, v in sorted(ok.items(), key=lambda kv: -kv[1]["d1"]):
        print(f"{k:46s}{v['d1']:8.4f}{v['exact']:8.3f}{v['medrel']:9.3f}")


if __name__ == "__main__":
    main()
