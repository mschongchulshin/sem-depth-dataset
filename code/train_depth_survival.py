import argparse
import collections
import itertools
import json
import os
import time

import numpy as np


ROOT = os.environ.get("SEM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE = ("/private/tmp/claude-501/-Users-hongchulshin/"
                 "1c78f3b9-27cc-4bf5-ad58-b616b56ba9e3/scratchpad/depthcache")
ARM_NAMES = ("M", "P", "S", "B", "Pperm")


def make_edges(exact, max_steps, log_bins):
    if exact < 1 or max_steps <= exact:
        raise ValueError("max steps must exceed exact steps")
    shallow = np.arange(1, exact + 2, dtype=np.int64)
    deep = np.rint(np.geomspace(exact + 1, max_steps + 1, log_bins + 1)).astype(np.int64)
    edges = np.unique(np.r_[shallow, deep])
    edges = edges[(edges >= 1) & (edges <= max_steps + 1)]
    if edges[-1] != max_steps + 1:
        edges = np.r_[edges, max_steps + 1]
    return edges


def bin_steps(steps, edges):
    clipped = np.clip(np.rint(steps).astype(np.int64), 1, edges[-1] - 1)
    return np.searchsorted(edges[1:], clipped, side="right").astype(np.int64)


def write_json(path, value):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(tmp, path)


def d1_strata(pred, truth):
    groups = (("1", truth == 1), ("2", truth == 2), ("3", truth == 3),
              ("4-7", (truth >= 4) & (truth <= 7)), ("8+", truth >= 8))
    out = {}
    for name, mask in groups:
        if mask.any():
            ratio = pred[mask] / truth[mask]
            out[name] = {"d1": float(np.mean(np.maximum(ratio, 1 / ratio) < 1.25)),
                         "n": int(mask.sum())}
        else:
            out[name] = {"d1": None, "n": 0}
    return out


def metrics(pred, truth, nll_sum, nll_n):
    pred = np.maximum(pred.astype(np.float64), 1.0)
    truth = np.maximum(truth.astype(np.float64), 1.0)
    ratio = pred / truth
    rel = np.abs(pred - truth) / truth
    return {
        "d1": float(np.mean(np.maximum(ratio, 1 / ratio) < 1.25)),
        "d2": float(np.mean(np.maximum(ratio, 1 / ratio) < 1.25 ** 2)),
        "exact_step_accuracy": float(np.mean(pred == truth)),
        "within_one_step_accuracy": float(np.mean(np.abs(pred - truth) <= 1)),
        "absrel": float(np.mean(rel)),
        "medrel": float(np.median(rel)),
        "mae_steps": float(np.mean(np.abs(pred - truth))),
        "nll": float(nll_sum / max(nll_n, 1)),
        "n": int(truth.size),
        "d1_by_true_depth": d1_strata(pred, truth),
    }


def paired_bootstrap(folds, arms, draws, seed):
    rng = np.random.default_rng(seed)
    answer = {}
    for left, right in itertools.combinations(arms, 2):
        delta = np.array([folds[k]["arms"][left]["d1"] - folds[k]["arms"][right]["d1"]
                          for k in folds
                          if left in folds[k].get("arms", {})
                          and right in folds[k].get("arms", {})], dtype=np.float64)
        if not len(delta):
            continue
        sampled = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(1)
        answer[f"{left}-{right}"] = {
            "mean_difference_d1": float(delta.mean()),
            "ci95": [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))],
            "volumes": int(len(delta)),
        }
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=DEFAULT_CACHE)
    parser.add_argument("--arms", default=",".join(ARM_NAMES))
    parser.add_argument("--folds", type=int, default=0, help="zero evaluates every source volume")
    parser.add_argument("--fold-names", default="", help="comma separated volumes, overrides folds")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--steps", type=int, default=140)
    parser.add_argument("--batch", type=int, default=6)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--exact-steps", type=int, default=8)
    parser.add_argument("--log-bins", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--decision-grid", type=int, default=512)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--limit-planes", type=int, default=0,
                        help="debug limit for each split and zero keeps every plane")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="cache/depth_survival.json")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    arms = [item.strip() for item in args.arms.split(",") if item.strip()]
    unknown = sorted(set(arms) - set(ARM_NAMES))
    if unknown:
        raise ValueError(f"unknown arms {unknown}")
    if not arms:
        raise ValueError("at least one arm is required")
    if args.width % 8:
        raise ValueError("width must be divisible by eight for group normalization")

    import torch
    from torch import nn
    import torch.nn.functional as functional

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    meta_path = os.path.join(args.cache, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"missing staged metadata {meta_path}")
    with open(meta_path) as handle:
        staged = json.load(handle)
    n, crop = int(staged["n"]), int(staged["crop"])
    if len(staged["meta"]) < n:
        raise ValueError("staged metadata has fewer planes than n")
    em = np.load(os.path.join(args.cache, "em.npy"), mmap_mode="r")[:n]
    depth = np.load(os.path.join(args.cache, "depth.npy"), mmap_mode="r")[:n]
    valid = np.load(os.path.join(args.cache, "valid.npy"), mmap_mode="r")[:n]
    if em.shape != (n, crop, crop) or depth.shape != em.shape or valid.shape != em.shape:
        raise ValueError("staged arrays do not match metadata")
    records = staged["meta"][:n]
    volume = np.asarray([str(row["dataset"]) for row in records])
    px_nm = np.asarray([float(row["px_nm"]) for row in records], dtype=np.float32)
    z_nm = np.asarray([float(row.get("z_step_nm") or row["px_nm"]) for row in records],
                      dtype=np.float32)
    if np.any(px_nm <= 0) or np.any(z_nm <= 0):
        raise ValueError("pixel and axial spacings must be positive")
    edges = make_edges(args.exact_steps, args.max_steps, args.log_bins)
    actions = np.unique(np.rint(np.linspace(1, args.max_steps, args.decision_grid)).astype(np.int64))
    counts = collections.Counter(volume.tolist())
    if args.fold_names:
        wanted = [f.strip() for f in args.fold_names.split(",") if f.strip()]
        missing = [f for f in wanted if f not in counts]
        if missing:
            raise SystemExit(f"unknown source volumes {missing}")
        held_out = wanted
    else:
        held_out = [name for name, _ in counts.most_common()]
        if args.folds > 0:
            held_out = held_out[:args.folds]
    out_path = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    result = {"configuration": {"arms": arms, "exact_steps": args.exact_steps,
              "log_bins": args.log_bins, "max_steps": args.max_steps,
              "decision_grid": actions.tolist(), "cache": args.cache,
              "epochs": args.epochs, "steps": args.steps, "batch": args.batch,
              "width": args.width, "seed": args.seed, "limit_planes": args.limit_planes}, "folds": {}}
    if not args.no_resume and os.path.exists(out_path):
        with open(out_path) as handle:
            old = json.load(handle)
        if old.get("configuration") == result["configuration"]:
            result = old
    result.setdefault("folds", {})
    print(f"device {device} staged {n} planes crop {crop} bins {len(edges) - 1}", flush=True)
    print(f"arms {arms} held out {len(held_out)}", flush=True)

    class FilmBlock(nn.Module):
        def __init__(self, width, dilation):
            super().__init__()
            self.one = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
            self.two = nn.Conv2d(width, width, 3, padding=dilation, dilation=dilation)
            self.norm_one = nn.GroupNorm(8, width, affine=False)
            self.norm_two = nn.GroupNorm(8, width, affine=False)

        def forward(self, x, film):
            scale_one, shift_one, scale_two, shift_two = film.chunk(4, dim=1)
            y = self.norm_one(self.one(x))
            y = functional.silu(y * (1 + scale_one[:, :, None, None]) + shift_one[:, :, None, None])
            y = self.norm_two(self.two(y))
            y = functional.silu(y * (1 + scale_two[:, :, None, None]) + shift_two[:, :, None, None])
            return x + y

    class SurvivalTower(nn.Module):
        def __init__(self, width, bins):
            super().__init__()
            self.width = width
            self.stem = nn.Conv2d(2, width, 3, padding=1)
            self.blocks = nn.ModuleList([FilmBlock(width, dilation) for dilation in (1, 2, 4, 8, 4, 2)])
            self.meta = nn.Sequential(nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, len(self.blocks) * 4 * width))
            self.head = nn.Conv2d(width, bins, 1)

        def forward(self, image_mask, metadata):
            film = self.meta(metadata).view(-1, len(self.blocks), 4 * self.width)
            feature = self.stem(image_mask)
            for number, block in enumerate(self.blocks):
                feature = block(feature, film[:, number])
            return self.head(feature)

        def metadata_only(self, metadata, height, width):
            blank = metadata.new_zeros((metadata.shape[0], 2, height, width))
            return self(blank, metadata)

    def log_probabilities(logits):
        hazards = torch.sigmoid(logits)
        hazards = torch.cat([hazards[:, :-1], torch.ones_like(hazards[:, -1:])], dim=1)
        log_hazard = torch.log(hazards.clamp_min(1e-7))
        log_survival = torch.cumsum(torch.log1p((-hazards).clamp(min=-1 + 1e-7)), dim=1)
        before = torch.cat([torch.zeros_like(log_survival[:, :1]), log_survival[:, :-1]], dim=1)
        return log_hazard + before, before

    def survival_nll(logits, event_bin, event_mask, censor_bin=None, censor_mask=None):
        event_logp, log_before = log_probabilities(logits)
        chosen = event_logp.gather(1, event_bin[:, None]).squeeze(1)
        total = -(chosen * event_mask).sum()
        count = event_mask.sum()
        if censor_bin is not None and censor_mask is not None:
            lower = log_before.gather(1, censor_bin[:, None]).squeeze(1)
            total = total - (lower * censor_mask).sum()
            count = count + censor_mask.sum()
        return total / count.clamp_min(1), total.detach(), count.detach()

    utility = np.empty((len(actions), len(edges) - 1), dtype=np.float32)
    for column, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        support = np.arange(low, high, dtype=np.float64)
        utility[:, column] = np.mean((support[None, :] >= actions[:, None] / 1.25) &
                                     (support[None, :] <= actions[:, None] * 1.25), axis=1)

    def decisions(mass):
        flat = mass.transpose(0, 2, 3, 1).reshape(-1, mass.shape[1])
        answer = np.empty(len(flat), dtype=np.int64)
        for start in range(0, len(flat), 8192):
            score = flat[start:start + 8192] @ utility.T
            answer[start:start + len(score)] = actions[np.argmax(score, axis=1)]
        return answer.reshape(mass.shape[0], mass.shape[2], mass.shape[3])

    def make_permutation(indices, seed):
        mapping = np.arange(n, dtype=np.int64)
        rng = np.random.default_rng(seed)
        groups = {}
        for index in indices:
            key = (volume[index], float(px_nm[index]), float(z_nm[index]))
            groups.setdefault(key, []).append(int(index))
        for members in groups.values():
            shuffled = np.asarray(members, dtype=np.int64).copy()
            rng.shuffle(shuffled)
            mapping[np.asarray(members, dtype=np.int64)] = shuffled
        return mapping

    def make_batch(indices, arm, photo_map, augment, rng):
        indices = np.asarray(indices, dtype=np.int64)
        photo = np.asarray(em[photo_map[indices]], dtype=np.float32)
        photo = (photo - photo.mean((1, 2), keepdims=True)) / np.maximum(photo.std((1, 2), keepdims=True), 1e-3)
        silhouette = np.asarray(valid[indices], dtype=np.float32)
        if arm in ("M", "S"):
            photo.fill(0)
        if arm in ("M", "P", "Pperm"):
            silhouette.fill(0)
        steps = np.asarray(depth[indices], dtype=np.float32) / z_nm[indices, None, None]
        labels = bin_steps(steps, edges)
        mask = np.asarray(valid[indices], dtype=np.float32)
        if augment and rng.random() < .5:
            photo, silhouette, labels, mask = (photo[:, ::-1], silhouette[:, ::-1],
                                                labels[:, ::-1], mask[:, ::-1])
        if augment and rng.random() < .5:
            photo, silhouette, labels, mask = (photo[:, :, ::-1], silhouette[:, :, ::-1],
                                                labels[:, :, ::-1], mask[:, :, ::-1])
        image_mask = np.stack([photo, silhouette], axis=1)
        metadata = np.stack([np.log(px_nm[indices] / 16), np.log(z_nm[indices] / 16)], axis=1).astype(np.float32)
        return (np.ascontiguousarray(image_mask), np.ascontiguousarray(metadata),
                np.ascontiguousarray(labels), np.ascontiguousarray(mask), steps)

    def evaluate(net, indices, arm, photo_map, rng):
        net.eval()
        pred_all, truth_all = [], []
        nll_sum, nll_n = 0.0, 0
        with torch.no_grad():
            for index in indices:
                x, metadata, labels, mask, truth = make_batch([index], arm, photo_map, False, rng)
                logits = net(torch.from_numpy(x).to(device), torch.from_numpy(metadata).to(device))
                label_t = torch.from_numpy(labels).to(device)
                mask_t = torch.from_numpy(mask).to(device)
                _, total, count = survival_nll(logits, label_t, mask_t)
                logp, _ = log_probabilities(logits)
                mass = torch.exp(logp).cpu().numpy()
                predicted = decisions(mass)[0]
                keep = mask[0].astype(bool)
                pred_all.append(predicted[keep])
                truth_all.append(np.rint(truth[0][keep]).astype(np.int64))
                nll_sum += float(total.cpu())
                nll_n += int(count.cpu())
        return metrics(np.concatenate(pred_all), np.concatenate(truth_all), nll_sum, nll_n)

    for fold_number, held in enumerate(held_out):
        test_index = np.flatnonzero(volume == held)
        train_full = np.flatnonzero(volume != held)
        if len(test_index) == 0 or len(train_full) < 2:
            print(f"skip {held} because the fold has too few planes", flush=True)
            continue
        fold_rng = np.random.default_rng(args.seed + fold_number)
        fold_rng.shuffle(train_full)
        val_count = min(180, max(1, len(train_full) // 10))
        val_index, train_index = train_full[:val_count], train_full[val_count:]
        if not len(train_index):
            train_index, val_index = train_full, train_full[:1]
        if args.limit_planes:
            if args.limit_planes < 1:
                raise ValueError("limit planes must be positive")
            test_index = test_index[:args.limit_planes]
            train_index = train_index[:args.limit_planes]
            val_index = val_index[:args.limit_planes]
        fold_data = result["folds"].setdefault(held, {"n_test_planes": int(len(test_index)), "arms": {}})
        print(f"hold out {held} train {len(train_index)} test {len(test_index)}", flush=True)
        for arm_number, arm in enumerate(arms):
            if arm in fold_data["arms"]:
                print(f"  {arm} resumed", flush=True)
                continue
            permutation = (make_permutation(train_index, args.seed + fold_number * 101)
                           if arm == "Pperm" else np.arange(n, dtype=np.int64))
            val_permutation = (make_permutation(val_index, args.seed + fold_number * 103)
                               if arm == "Pperm" else np.arange(n, dtype=np.int64))
            test_permutation = (make_permutation(test_index, args.seed + fold_number * 107)
                                if arm == "Pperm" else np.arange(n, dtype=np.int64))
            torch.manual_seed(args.seed + fold_number * 1000)
            net = SurvivalTower(args.width, len(edges) - 1).to(device)
            optimizer = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.4, patience=3)
            best_loss, best_state, stale = float("inf"), None, 0
            rng = np.random.default_rng(args.seed + fold_number * 100)
            started = time.time()
            for epoch in range(args.epochs):
                net.train()
                for _ in range(args.steps):
                    batch_index = rng.choice(train_index, size=args.batch,
                                             replace=len(train_index) < args.batch)
                    x, metadata, labels, mask, _ = make_batch(batch_index, arm, permutation, True, rng)
                    loss, _, _ = survival_nll(net(torch.from_numpy(x).to(device), torch.from_numpy(metadata).to(device)),
                                               torch.from_numpy(labels).to(device), torch.from_numpy(mask).to(device))
                    if torch.isfinite(loss):
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(net.parameters(), 5)
                        optimizer.step()
                net.eval()
                validation = []
                with torch.no_grad():
                    for start in range(0, len(val_index), args.batch):
                        subset = val_index[start:start + args.batch]
                        x, metadata, labels, mask, _ = make_batch(subset, arm, val_permutation, False, rng)
                        loss, _, _ = survival_nll(net(torch.from_numpy(x).to(device), torch.from_numpy(metadata).to(device)),
                                                   torch.from_numpy(labels).to(device), torch.from_numpy(mask).to(device))
                        validation.append(float(loss.cpu()))
                value = float(np.mean(validation))
                scheduler.step(value)
                if value < best_loss - 1e-5:
                    best_loss, stale = value, 0
                    best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
                else:
                    stale += 1
                if stale >= 7:
                    break
            if best_state is not None:
                net.load_state_dict(best_state)
            scored = evaluate(net, test_index, arm, test_permutation, rng)
            fold_data["arms"][arm] = scored
            write_json(out_path, result)
            print(f"  {arm} d1 {scored['d1']:.4f} exact {scored['exact_step_accuracy']:.4f} "
                  f"nll {scored['nll']:.4f} seconds {time.time() - started:.0f}", flush=True)

    completed = [fold for fold in result["folds"].values() if fold.get("arms")]
    means = {}
    for arm in arms:
        rows = [fold["arms"][arm] for fold in completed if arm in fold["arms"]]
        if rows:
            means[arm] = {key: float(np.mean([row[key] for row in rows]))
                          for key in ("d1", "d2", "exact_step_accuracy", "within_one_step_accuracy",
                                      "absrel", "medrel", "mae_steps", "nll")}
            means[arm]["d1_by_true_depth"] = {}
            for stratum in ("1", "2", "3", "4-7", "8+"):
                values = [row["d1_by_true_depth"][stratum]["d1"] for row in rows
                          if row["d1_by_true_depth"][stratum]["d1"] is not None]
                means[arm]["d1_by_true_depth"][stratum] = {
                    "d1": float(np.mean(values)) if values else None,
                    "folds": len(values),
                }
            means[arm]["folds"] = len(rows)
    result["mean_over_folds"] = means
    result["paired_bootstrap_d1"] = paired_bootstrap(result["folds"], arms, args.bootstrap, args.seed)
    write_json(out_path, result)
    print(f"results {out_path}", flush=True)
    for arm, row in means.items():
        print(f"mean {arm} d1 {row['d1']:.4f} nll {row['nll']:.4f} folds {row['folds']}", flush=True)


if __name__ == "__main__":
    main()
