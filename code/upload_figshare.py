import concurrent.futures as cf, hashlib, json, os, sys, time
import urllib.request, urllib.error

API = "https://api.figshare.com/v2"
TOK = os.environ["FIGSHARE_TOKEN"]
DIST = os.path.join(os.environ.get("SEM_DEPTH_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "dist")


def call(method, url, body=None, raw=False):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {TOK}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        b = r.read()
    if raw:
        return b
    try:
        return json.loads(b) if b else {}
    except ValueError:
        return {}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def put_part(url, path, part, retries=4):
    for _ in range(retries):
        with open(path, "rb") as fh:
            fh.seek(part["startOffset"])
            chunk = fh.read(part["endOffset"] - part["startOffset"] + 1)
        req = urllib.request.Request(f"{url}/{part['partNo']}", data=chunk, method="PUT")
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                r.read()
            return len(chunk)
        except Exception as e:
            err = e
    raise err


def one(fname, workers=8, article=None):
    path = os.path.join(DIST, fname)
    size = os.path.getsize(path)

    if article is None:
        a = call("POST", f"{API}/account/articles", {
            "title": "A dataset for depth estimation from single block-face electron "
                     "micrographs of cells and tissues",
            "defined_type": "dataset"})
        article = a["location"].rstrip("/").split("/")[-1]
        print("article", article, flush=True)

    loc = call("POST", f"{API}/account/articles/{article}/files",
               {"name": fname, "md5": md5(path), "size": size})["location"]
    info = call("GET", loc)
    up = call("GET", info["upload_url"])
    parts = up["parts"]
    print(f"{fname}  {size/1e6:.0f} MB in {len(parts)} parts, {workers} workers", flush=True)

    t0 = time.time()
    done = [0]
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(put_part, info["upload_url"], path, p): p for p in parts}
        for f in cf.as_completed(futs):
            done[0] += f.result()
            el = time.time() - t0
            print(f"  {done[0]/1e6:7.1f}/{size/1e6:.0f} MB  {done[0]/el/1024:7.0f} KB/s", flush=True)

    call("POST", loc)
    print(f"complete in {time.time()-t0:.0f} s, {size/(time.time()-t0)/1024:.0f} KB/s", flush=True)
    return article


def main():
    article = sys.argv[1] if len(sys.argv) > 1 else None
    man = json.load(open(os.path.join(DIST, "MANIFEST.json")))
    want = {e["file"]: e["bytes"] for e in man}
    if article:
        req = urllib.request.Request(
            f"{API}/account/articles/{article}/files?page_size=100",
            headers={"Authorization": f"token {TOK}"})
        have = {f["name"]: f["size"] for f in json.load(urllib.request.urlopen(req))}
    else:
        have = {}
    todo = [f for f, s in want.items() if have.get(f) != s]
    print(f"{len(have)} already there, {len(todo)} to upload", flush=True)
    for f in todo:
        article = one(f, 8, article)
    print("all uploaded, article", article, flush=True)


main()
