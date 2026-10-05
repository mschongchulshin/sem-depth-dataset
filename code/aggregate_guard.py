import os, re, signal, subprocess, sys, time
from collections import deque

ROOT = "/Volumes/One Touch/em-depth-dataset"
FREE_GB_FLOOR = float(os.environ.get("AG_FREE_FLOOR_GB", 6))
SWAP_GROWTH_MB = float(os.environ.get("AG_SWAP_GROWTH_MB", 200))
HARD_FLOOR_GB = float(os.environ.get("AG_HARD_FLOOR_GB", 4))
POLL_S = float(os.environ.get("AG_POLL_S", 10))
WINDOW = 4
COOLDOWN_S = 300
LOG = os.path.join(ROOT, "aggregate_guard.log")


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def free_gb():
    out = subprocess.run(["top", "-l", "1", "-n", "0"], capture_output=True, text=True).stdout
    m = re.search(r"PhysMem:.*?(\d+)([GM]) unused", out)
    if not m:
        return None
    v = float(m.group(1))
    return v if m.group(2) == "G" else v / 1024


def swap_mb():
    out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    m = re.search(r"used\s*=\s*([\d.]+)M", out)
    return float(m.group(1)) if m else None


def _etime_s(et):
    if not et:
        return 0
    days = 0
    if "-" in et:
        d, et = et.split("-", 1)
        days = int(d)
    parts = [int(x) for x in et.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    return days * 86400 + h * 3600 + m * 60 + sec


def renders():
    out = subprocess.run(["pgrep", "-f", "render_organelles"], capture_output=True, text=True).stdout
    rows = []
    for pid in [p for p in out.split() if p.isdigit()]:
        cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
        m = re.search(r"raw/([^ ]+)\.npz", cmd)
        if not m:
            continue
        box = m.group(1)
        vdir = os.path.join(ROOT, "corpus", box, "views")
        try:
            n = len([f for f in os.listdir(vdir) if f.endswith(".npz") and not f.startswith("._")])
        except OSError:
            n = 0
        et = subprocess.run(["ps", "-o", "etime=", "-p", pid], capture_output=True, text=True).stdout.strip()
        rss = subprocess.run(["ps", "-o", "rss=", "-p", pid], capture_output=True, text=True).stdout.strip()
        gb = (int(rss) / 1048576) if rss.strip().isdigit() else 0.0
        rows.append((int(pid), n, _etime_s(et), box, gb))
    return rows


def main():
    say(f"aggregate guard up: hard floor {HARD_FLOOR_GB} GB free fires alone; "
        f"soft floor {FREE_GB_FLOOR} GB needs +{SWAP_GROWTH_MB} MB swap growth over "
        f"{WINDOW} polls of {POLL_S}s")
    hist = deque(maxlen=WINDOW)
    last_kill = 0.0
    while True:
        f, s = free_gb(), swap_mb()
        if f is None or s is None:
            time.sleep(POLL_S)
            continue
        hist.append(s)
        grew = len(hist) == WINDOW and (hist[-1] - hist[0]) >= SWAP_GROWTH_MB
        critical = f < HARD_FLOOR_GB
        if (critical or (f < FREE_GB_FLOOR and grew)) and (time.time() - last_kill) > COOLDOWN_S:
            rows = renders()
            if rows:
                pid, n, et, box, gb = sorted(rows, key=lambda r: (r[1], -r[4], r[2]))[0]
                why = "CRITICAL free memory" if critical else "low memory with swap growth"
                say(f"{why}: free={f:.1f}GB swap {hist[0]:.0f}->{hist[-1]:.0f}MB. "
                    f"killing pid {pid} on {box} ({n}/12 views, {et}s, {gb:.1f}GB) "
                    f"to reclaim memory")
                try:
                    os.kill(pid, signal.SIGKILL)
                    last_kill = time.time()
                    hist.clear()
                except OSError as e:
                    say(f"  kill failed: {e}")
            else:
                say(f"DISTRESS free={f:.0f}GB but no render worker to shed")
        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        say("aggregate guard down")
