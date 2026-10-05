import os, sys, signal, threading, time
import psutil

_state = {"peak": 0.0, "armed": False}


def _tree_rss_gb(proc):
    total = proc.memory_info().rss
    for c in proc.children(recursive=True):
        try:
            total += c.memory_info().rss
        except psutil.Error:
            pass
    return total / 1024 ** 3


def _watch(cap_gb, warn_gb, interval, log):
    me = psutil.Process()
    stop = threading.Event()
    while not stop.wait(interval):
        try:
            rss = _tree_rss_gb(me)
        except psutil.Error:
            return
        if rss > _state["peak"]:
            _state["peak"] = rss
            if rss > warn_gb:
                print(f"[guard] rss {rss:.1f} GB of {cap_gb:.0f} GB cap", flush=True)
        if rss > cap_gb:
            msg = (f"[guard] ABORT: rss {rss:.1f} GB exceeded the {cap_gb:.0f} GB cap. "
                   f"Reduce --box or the pyramid level.")
            print(msg, file=sys.stderr, flush=True)
            if log:
                with open(log, "a") as fh:
                    fh.write(msg + "\n")
            for c in me.children(recursive=True):
                try:
                    c.kill()
                except psutil.Error:
                    pass
            os.kill(os.getpid(), signal.SIGKILL)


def preflight(nbytes, what="allocation", cap_gb=None):
    cap = float(cap_gb or os.environ.get("EMD_MEM_CAP_GB", 48))
    gb = nbytes / 1024 ** 3
    head = psutil.Process().memory_info().rss / 1024 ** 3
    if head + gb > cap:
        raise MemoryError(f"[guard] {what} needs {gb:.1f} GB on top of {head:.1f} GB resident, "
                          f"over the {cap:.0f} GB cap")
    print(f"[guard] preflight ok: {what} {gb:.1f} GB, resident {head:.1f} GB, cap {cap:.0f} GB",
          flush=True)
    return True


def arm(cap_gb=None, threads=None):
    if _state["armed"]:
        return
    cap_gb = float(cap_gb or os.environ.get("EMD_MEM_CAP_GB", 48))
    threads = int(threads or os.environ.get("EMD_THREADS", 8))
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, str(threads))
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass
    log = os.environ.get("EMD_GUARD_LOG")
    interval = float(os.environ.get('EMD_POLL_S', 0.05))
    t = threading.Thread(target=_watch, args=(cap_gb, cap_gb * 0.6, interval, log), daemon=True)
    t.start()
    _state["armed"] = True
    print(f"[guard] armed: {cap_gb:.0f} GB rss cap, {threads} compute threads, pid {os.getpid()}",
          flush=True)

    import atexit
    atexit.register(lambda: print(f"[guard] peak rss {_state['peak']:.2f} GB", flush=True))
