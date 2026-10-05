import json, os, subprocess, sys, time

ROOT = os.environ.get("EM_DEPTH_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WINDOWS = (30, 60)


def _empty(base):
    n = 0
    for d in os.listdir(base):
        if d.startswith("._"):
            continue
        try:
            with open(os.path.join(base, d, "index.json")) as f:
                if not json.load(f):
                    n += 1
        except (OSError, ValueError):
            pass
    return n


CACHE = os.path.join(ROOT, "cache", "status_cache.json")


def counts():
    try:
        with open(CACHE) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    out, dirty = {}, False
    sealed = cache.get("__sealed__", {})
    for sub in ("blockface", "thinsection", "corpus"):
        if sub in sealed:
            out[sub] = tuple(sealed[sub])
            continue
        base = os.path.join(ROOT, sub)
        n = b = 0
        for d in os.listdir(base):
            if d.startswith("._"):
                continue
            idx = os.path.join(base, d, "index.json")
            try:
                mt = os.stat(idx).st_mtime
            except OSError:
                continue
            key = sub + "/" + d
            hit = cache.get(key)
            if hit and hit[0] == mt:
                k = hit[1]
            else:
                try:
                    with open(idx) as f:
                        k = len(json.load(f))
                except (OSError, ValueError):
                    continue
                cache[key] = [mt, k]
                dirty = True
            n += k
            b += 1 if k else 0
        out[sub] = (n, b)
        if sub in ("blockface", "thinsection") and b + _empty(base) >= 2377:
            sealed[sub] = [n, b]
            cache["__sealed__"] = sealed
            dirty = True
        out[sub] = (n, b)
    if dirty:
        tmp = CACHE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cache, f)
        os.replace(tmp, CACHE)
    return out


def throughput(now):
    base = os.path.join(ROOT, "corpus")
    widest = max(WINDOWS) * 60
    hits = {w: 0 for w in WINDOWS}
    attempted = scanned = 0
    for d in os.listdir(base):
        if d.startswith("._"):
            continue
        v = os.path.join(base, d, "views")
        try:
            dm = os.stat(v).st_mtime
        except OSError:
            continue
        attempted += 1
        if now - dm > widest:
            continue
        scanned += 1
        for f in os.listdir(v):
            if not f.endswith(".npz") or f.startswith("._"):
                continue
            try:
                age = (now - os.path.getmtime(os.path.join(v, f))) / 60
            except OSError:
                continue
            for w in WINDOWS:
                if age < w:
                    hits[w] += 1
    return attempted, scanned, hits


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def main():
    t0 = time.time()
    pgid = sh("cat '%s/.build.pgid'" % ROOT)
    alive = sh("ps -o etime= -p 67614").strip()
    workers = sh("ps -o pgid=,command= -ax | grep render_organelles | grep -v grep | "
                 "awk '$1==%s' | wc -l" % pgid)
    orphans = sh("ps -o pgid=,command= -ax | grep render_organelles | grep -v grep | "
                 "awk '$1!=%s' | wc -l" % pgid)
    tiers = sh("ps -o pgid=,command= -ax | grep render_organelles | grep -v grep | "
               "awk '$1==%s' | grep -oE 'raw/[^ ]+\\.npz' | sed 's|raw/||;s|\\.npz||;"
               "s/.*__//;s/_b[0-9]*$//' | sort | uniq -c | tr '\\n' ' '" % pgid)

    print(f"build 67614 uptime {alive or 'DEAD'}   workers {workers.strip()} in pgid {pgid}"
          f"   orphans {orphans.strip()}   aggguard {sh('pgrep -f aggregate_guard.py | wc -l').strip()}")
    print(f"tier mix: {tiers}")
    print(sh("uptime | sed 's/.*load/load/'"))

    c = counts()
    tot = sum(v[0] for v in c.values())
    for sub in ("blockface", "thinsection", "corpus"):
        n, b = c[sub]
        print(f"{sub:12s} images={n:7d} nonempty_boxes={b:5d}")
    print(f"{'TOTAL':12s} images={tot:7d}")

    att, scanned, hits = throughput(time.time())
    print(f"surface attempted {att}/2480   (scanned {scanned} recently-touched box dirs)")
    for w in WINDOWS:
        print(f"  last {w}m: {hits[w]}")

    print("failures:", sh("grep -h -E 'Traceback|MemoryError|Killed|FAIL' '%s'/corpus/*.log "
                          "2>/dev/null | wc -l" % ROOT).strip())
    print(sh("ps -o rss= -p $(pgrep -f render_organelles | tr '\\n' ',' | sed 's/,$//') | "
             "awk '{s+=$1} END {printf \"workers RSS %.1f GB\", s/1048576}'"))
    print(sh("top -l 1 -n 0 | grep PhysMem"))
    print("swap:", sh("sysctl -n vm.swapusage"))
    print("guard events:", max(0, int(sh("wc -l < '%s/aggregate_guard.log'" % ROOT) or 1) - 1))
    print(f"[status took {time.time()-t0:.1f}s]")


if __name__ == "__main__":
    main()
